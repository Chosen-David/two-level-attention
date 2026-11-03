"""
iter91：Qwen3RMSNorm patch 注入层（仿 sparse_attn/patches/patch.py 的实例级
MethodType 注入约定）。

注入方式：**实例级** forward 替换（不动 Qwen3RMSNorm 类本身，不影响其它
会话/进程已加载的模型）；每个模块保存 `_iter91_original_forward` 原引用，
可完整撤销。

覆盖范围（一个 Qwen3 模型内的全部 Qwen3RMSNorm 实例）：
  - 每层 attention 的 `q_norm` / `k_norm`（head_dim=128，pre-transpose 输入 (B,1,H,D)）
  - 每层 decoder 的 `input_layernorm` / `post_attention_layernorm`（(B, 1, H)）
  - 末层 `model.norm`（同为 (B, 1, H)，语义一致顺带覆盖）
预期实例数 = num_hidden_layers * 4 + 1（如 8B 为 36*4+1=145）。

行为：
  - forward 时先过 `is_fusable`（纯元数据判断，无 host sync）：仅
    seq_len==1 的 BF16 contiguous CUDA 输入走三 kernel 融合；
  - prefill（seq_len>1）/ fp32 weight / 非连续布局 → 原样调用 HF 原
    forward（保底语义不变）；
  - 仿 TLI patch 的 B3 纪律：调用方显式传 expected 时，count 不符即 raise，
    杜绝「零匹配却静默跑 dense 还打上融合标签」的实验结论污染。
"""

from types import MethodType

from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from ..ops.qwen3_rmsnorm_triton import (
    fused_rmsnorm_decode, fused_rmsnorm_decode_single, is_fusable)

_ORIG_ATTR = "_iter91_original_forward"


def _iter91_rmsnorm_forward(self, hidden_states):
    """替换后的 Qwen3RMSNorm.forward：decode 融合 / prefill 回退。"""
    # gate 全部为 shape/dtype/布局元数据判断：无 .item()、无 device sync，
    # CUDA graph capture 期间路径确定（同一静态 shape 恒走同一路径）。
    if is_fusable(hidden_states, self.weight):
        if self.weight.numel() == 128:
            return fused_rmsnorm_decode_single(
                hidden_states, self.weight, self.variance_epsilon)
        return fused_rmsnorm_decode(
            hidden_states, self.weight, self.variance_epsilon)
    # prefill / 其它 dtype / 非连续 → HF 原 forward，语义零改动
    return getattr(self, _ORIG_ATTR)(hidden_states)


def register_rmsnorm_patch(model, expected: int = None, verbose: bool = True) -> int:
    """注册 iter91 RMSNorm patch，返回成功 patch 的模块数。

    expected：调用方可传预期实例数（num_hidden_layers*4+1）；不符即 raise
    （防零匹配静默通过，见模块 docstring 的 B3 纪律说明）。
    """
    n_patched = 0
    for name, module in model.named_modules():
        if isinstance(module, Qwen3RMSNorm):
            if not hasattr(module, _ORIG_ATTR):  # 幂等：重复注册不覆盖原引用
                setattr(module, _ORIG_ATTR, module.forward)
            module.forward = MethodType(_iter91_rmsnorm_forward, module)
            n_patched += 1
            if verbose and (
                name.endswith(("q_norm", "k_norm", "input_layernorm", "post_attention_layernorm"))
                and ".0." in name
            ):
                print(f"register iter91 rmsnorm: [{name}]")
    if expected is not None and n_patched != expected:
        raise RuntimeError(
            f"iter91 RMSNorm patch 数量不符：patched={n_patched}，expected={expected}；"
            f"拒绝静默半 patch（结论会污染实验标签）"
        )
    return n_patched


def unregister_rmsnorm_patch(model) -> int:
    """撤销 iter91 patch，恢复 HF 类级原 forward；返回恢复的模块数。

    恢复方式：删除实例字典里的 `forward`（register 时注入的 MethodType），
    使属性查找回落到 Qwen3RMSNorm 类级 forward —— 与 patch 前完全同态。
    仅当实例 forward 确为 iter91 注入的那个 MethodType 时才删（若期间被
    第三方二次 patch 则不动其 forward，只清 _ORIG_ATTR 标记）。
    """
    n_restored = 0
    for _, module in model.named_modules():
        if getattr(module, _ORIG_ATTR, None) is None or not isinstance(module, Qwen3RMSNorm):
            continue
        fwd = module.__dict__.get("forward")
        if fwd is not None and getattr(fwd, "__func__", None) is _iter91_rmsnorm_forward:
            del module.__dict__["forward"]
        delattr(module, _ORIG_ATTR)
        n_restored += 1
    return n_restored
