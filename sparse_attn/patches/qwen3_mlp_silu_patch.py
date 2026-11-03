"""Qwen3MLP decode SiLU+multiply 的实例级 patch。

仿照 ``qwen3_rmsnorm_patch.py`` 的 MethodType 注入/撤销约定，但使用完全
独立的开关和原 forward 属性；注册本 patch 不会改变 RMSNorm 路径。

仅对 ``x.shape[-2] == 1``、BF16 contiguous CUDA 输入且激活为非 inplace
``nn.SiLU`` 的 Qwen3MLP 改换 elementwise 段。gate/up/down 三个 GEMM 的
调用与顺序保持原样，prefill 和其它几何直接执行保存的 HF 原 forward。
"""

from types import MethodType

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from ..ops.qwen3_mlp_silu_triton import fused_silu_mul_decode, is_fusable

_ORIG_ATTR = "_iter97_mlp_silu_original_forward"


def _can_use_fused_elementwise(module: Qwen3MLP, x: torch.Tensor) -> bool:
    """在 GEMM 前筛选生产几何；全部为无同步的元数据判断。"""
    if not isinstance(module.act_fn, nn.SiLU) or module.act_fn.inplace:
        return False
    if not isinstance(x, torch.Tensor) or x.dim() < 2 or x.shape[-2] != 1:
        return False
    if x.dtype is not torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        return False

    gate_weight = module.gate_proj.weight
    up_weight = module.up_proj.weight
    if gate_weight.shape != up_weight.shape:
        return False
    if gate_weight.dtype is not torch.bfloat16 or up_weight.dtype is not torch.bfloat16:
        return False
    if not gate_weight.is_cuda or not up_weight.is_cuda:
        return False
    return True


def _iter97_mlp_silu_forward(self, x):
    """保持三次投影不变，只替换两次 BF16 elementwise kernel。"""
    if not _can_use_fused_elementwise(self, x):
        return getattr(self, _ORIG_ATTR)(x)

    # 与 HF Qwen3MLP 相同的 gate/up 调用顺序；不在此融合或改写 GEMM。
    gate = self.gate_proj(x)
    up = self.up_proj(x)
    if is_fusable(gate, up):
        activated = fused_silu_mul_decode(gate, up)
    else:
        # 防御第三方 Linear patch 产生异常布局：复用已算出的 gate/up，
        # elementwise 段严格回退 HF，避免重复执行 GEMM。
        activated = self.act_fn(gate) * up
    return self.down_proj(activated)


def register_mlp_silu_patch(model, expected: int = None, verbose: bool = True) -> int:
    """注册 Qwen3MLP patch；expected 不符即拒绝静默半 patch。"""
    n_patched = 0
    for name, module in model.named_modules():
        if isinstance(module, Qwen3MLP):
            if not hasattr(module, _ORIG_ATTR):
                setattr(module, _ORIG_ATTR, module.forward)
            module.forward = MethodType(_iter97_mlp_silu_forward, module)
            n_patched += 1
            if verbose and name.endswith(".0.mlp"):
                print(f"register iter97 MLP SiLU+mul: [{name}]")
    if expected is not None and n_patched != expected:
        raise RuntimeError(
            f"iter97 Qwen3MLP patch 数量不符：patched={n_patched}，"
            f"expected={expected}；拒绝静默半 patch"
        )
    return n_patched


def unregister_mlp_silu_patch(model) -> int:
    """撤销本候选的实例 forward，不触碰 RMSNorm 或第三方 patch。"""
    n_restored = 0
    for _, module in model.named_modules():
        if getattr(module, _ORIG_ATTR, None) is None or not isinstance(module, Qwen3MLP):
            continue
        fwd = module.__dict__.get("forward")
        if fwd is not None and getattr(fwd, "__func__", None) is _iter97_mlp_silu_forward:
            del module.__dict__["forward"]
        delattr(module, _ORIG_ATTR)
        n_restored += 1
    return n_restored
