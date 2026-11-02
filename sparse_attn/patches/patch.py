import torch.nn as nn
from types import MethodType
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from .qwen3_attn_patch import qwen3_attn_forward
from ..indexer import indexer_type_dict
from ..indexer.nfa_indexer import NFAIndexer

def _reject_removed_cache_args(args):
    backend = getattr(args, "nfa_far_skip_backend", "legacy")
    cache = getattr(args, "nfa_far_skip_cache", "retain")
    if backend != "legacy" or cache != "retain":
        raise ValueError(
            "far-skip/static/batch cache paths have been removed; "
            "only GraphReadyPagedRegionCache is supported "
            "(nfa_far_skip_backend=legacy, nfa_far_skip_cache=retain)")



def _is_glm4_moe_lite_attn(module) -> bool:
    """Real HF GLM-4.7-Flash attention structure gate."""
    return (
        type(module).__name__ == "Glm4MoeLiteAttention"
        and all(hasattr(module, name) for name in (
            "kv_a_proj_with_mqa", "qk_head_dim", "qk_nope_head_dim",
            "qk_rope_head_dim", "v_head_dim", "scaling", "layer_idx"))
    )


def register_patch(model: nn.Module, args, snapshot=None) -> int:
    """注册稀疏 attention patch，返回成功 patch 的模块数。

    当前生产路径仅支持 Qwen3：dense prefill + frozen-region gate ->
    near/far 粗筛 -> 选中细筛 -> 选中 token main attention。

    【B3 修复（kimi3 清单 F3，2026-10-08）】method≠none 而零模块匹配时，
    原实现静默返回——未支持模型跑 method≠none → 零匹配 → 实际
    dense 输出被打上稀疏方法标签（实验结论污染）。改为：
      ① 返回 patch 计数（调用方可显式校验）；
      ② method≠none 且 patched=0 时显式 raise 拒绝。

    【081（TL-E121-OUTPUT-SNAPSHOT-081）】snapshot：生产入口在模型
    加载前冻结的 TreatmentSnapshot（sparse_attn.info.
    resolve_treatment_snapshot 产出）。nfa 臂逐层建 indexer 实例时
    注入同一 snapshot 对象——层实例仍逐层建（layer_idx 各异），但
    D′ 掩码 / 投影基内容不再按层重开文件（旧路径每层各 open 一次，
    patch 中途换文件 → 跨层混代可达）。仅 NFAIndexer 接受注入，
    其他 indexer 类型忽略 snapshot（其无文件型配置）。
    """
    _reject_removed_cache_args(args)
    IndexerType = indexer_type_dict.get(args.method, None)
    if args.method != 'none' and IndexerType is None:
        raise ValueError(f"not support {args.method}")
    glm_targets = [m for m in model.modules() if _is_glm4_moe_lite_attn(m)]
    glm_forward = None
    if glm_targets:
        if args.method != 'nfa':
            raise ValueError('GLM-4.7-Flash 仅支持 method=nfa 或原生 method=none')
        expected_layers = getattr(model.config, 'num_hidden_layers', None)
        if (getattr(model.config, 'model_type', None) != 'glm4_moe_lite'
                or expected_layers != 47 or len(glm_targets) != expected_layers):
            raise ValueError(
                f'GLM NFA 要求真实 glm4_moe_lite 且恰好 47 个 attention 层；'
                f'当前 model_type={getattr(model.config, "model_type", None)!r}、'
                f'num_hidden_layers={expected_layers!r}、匹配层数={len(glm_targets)}')
        if any(
            m.qk_head_dim != 256 or m.qk_nope_head_dim != 192
            or m.qk_rope_head_dim != 64 or m.v_head_dim != 256
            for m in glm_targets
        ):
            raise ValueError('GLM NFA 要求 20 heads 且 K=V=256=[NoPE192|RoPE64]')
        from .glm4_moe_lite_attn_patch import glm4_moe_lite_attn_forward
        glm_forward = glm4_moe_lite_attn_forward

    n_patched = 0
    if IndexerType is not None:
        for name, module in model.named_modules():
            if isinstance(module, Qwen3Attention):
                module.indexer = IndexerType(args, snapshot=snapshot) \
                    if (snapshot is not None and IndexerType is NFAIndexer) \
                    else IndexerType(args)
                module.indexer.head_dim = module.head_dim
                if hasattr(module, "layer_idx"):
                    module.indexer.layer_idx = module.layer_idx  # NFA D' 层掩码用
                if not hasattr(module, '_nfa_original_forward'):
                    module._nfa_original_forward = module.forward
                module.forward = MethodType(qwen3_attn_forward, module)
                n_patched += 1
                if hasattr(module, "layer_idx") and module.layer_idx == 0:
                    print(f"register qwen3 attn: [{name}]")
            elif glm_forward is not None and _is_glm4_moe_lite_attn(module):
                module.indexer = IndexerType(args, snapshot=snapshot)
                module.indexer.layer_idx = module.layer_idx
                module.indexer.head_dim = 256
                module.indexer._idx_sub_cache = None
                module.indexer._qat_idx_cache = None
                module.indexer._inc = None
                if not hasattr(module, '_nfa_original_forward'):
                    module._nfa_original_forward = module.forward
                module.forward = MethodType(glm_forward, module)
                n_patched += 1
                if module.layer_idx == 0:
                    print(f"register glm4_moe_lite attn (D=256): [{name}]")
    if args.method != 'none' and n_patched == 0:
        mod_types = sorted({type(m).__name__ for _, m in model.named_modules()})
        hint = ""
        if any('glm' in t.lower() for t in mod_types):
            hint = ("（检测到 GLM 系模块，但并非完整匹配的 47 层 "
                    "Glm4MoeLiteAttention 生产结构）")
        raise RuntimeError(
            f"method={args.method!r} 但模型中没有任何可 patch 的 "
            f"Qwen3Attention/Glm4MoeLiteAttention 模块（patched=0）——继续运行会"
            f"静默产出 dense 结果却被标记为稀疏方法{hint}。请确认模型架构"
            f"或改 --method none。模块类型样本: {mod_types[:8]}"
        )
    return n_patched
