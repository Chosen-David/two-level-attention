import os

import torch
import torch.nn as nn
from types import MethodType
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.llama.modeling_llama import LlamaAttention


from .qwen3_attn_patch import qwen3_attn_forward
from .llama3_attn_patch import llama3_attn_forward
from .deepseek_v2_attn_patch import deepseek_v2_attn_forward
from ..indexer import indexer_type_dict
from ..indexer.tli_indexer import TLIIndexer

# 【E127 inplace 合并（用户 2026-10-12 指令）】TLIIndexerFast 子类退役：
# 增量缓存 + Triton fast path 语义已全部并入 TLIIndexer 基类
# （tli_indexer.py），env 实例化切换门随之退役——tli 臂恒实例化基类，
# fast path 由基类内部守卫（GPU/Triton 可用性/布局）自动生效，
# 异常 fail-closed 回退原版表达式。tli_indexer_fast.py 已删除。


def _is_glm4_moe_lite_attn(module) -> bool:
    """真实 HF GLM-4.7-Flash attention 结构门，避免误挂其他 GLM/MLA 层。"""
    return (
        type(module).__name__ == "Glm4MoeLiteAttention"
        and all(hasattr(module, name) for name in (
            "kv_a_proj_with_mqa", "qk_head_dim", "qk_nope_head_dim",
            "qk_rope_head_dim", "v_head_dim", "scaling", "layer_idx"))
    )


def _is_deepseek_attn(module) -> bool:
    """【E142a】MLA 结构门：类名前缀 DeepseekV2* + MLA 投影属性存在
    （kv_a_proj_with_mqa / q_head_dim / qk_rope_head_dim）。防非 attention
    的 DeepseekV2 前缀类（MoE/decoder 层等）误挂 patch。"""
    return (
        hasattr(module, "kv_a_proj_with_mqa")
        and hasattr(module, "q_head_dim")
        and hasattr(module, "qk_rope_head_dim")
        and hasattr(module, "rotary_emb")
    )

def register_patch(model: nn.Module, args, snapshot=None) -> int:
    """注册稀疏 attention patch，返回成功 patch 的模块数。

    【B3 修复（kimi3 清单 F3，2026-10-08）】method≠none 而零模块匹配时，
    原实现静默返回——GLM-4 等未支持模型跑 method≠none → 零匹配 → 实际
    dense 输出被打上稀疏方法标签（实验结论污染）。改为：
      ① 返回 patch 计数（调用方可显式校验）；
      ② method≠none 且 patched=0 时显式 raise 拒绝。
    GLM 说明：仅真实 47 层 Glm4MoeLiteAttention + 已验证 D=256 tail32
    生产子集注册；其余 GLM 架构或配置继续 fail-closed。

    【081（TL-E121-OUTPUT-SNAPSHOT-081）】snapshot：生产入口在模型
    加载前冻结的 TreatmentSnapshot（sparse_attn.info.
    resolve_treatment_snapshot 产出）。tli 臂逐层建 indexer 实例时
    注入同一 snapshot 对象——层实例仍逐层建（layer_idx 各异），但
    D′ 掩码 / 投影基内容不再按层重开文件（旧路径每层各 open 一次，
    patch 中途换文件 → 跨层混代可达）。仅 TLIIndexer 接受注入，
    其他 indexer 类型忽略 snapshot（其无文件型配置）。
    """
    from ..ops.far_skip_runtime import resolve_far_skip_modes
    resolve_far_skip_modes(args, model=model)
    IndexerType = indexer_type_dict.get(args.method, None)
    glm_targets = [m for m in model.modules() if _is_glm4_moe_lite_attn(m)]
    glm_forward = None
    if glm_targets and args.method not in ("none", "tli"):
        raise ValueError("GLM-4.7-Flash 仅支持 method=tli 或原生 method=none")
    if glm_targets and args.method == "tli":
        config = model.config
        expected_layers = getattr(config, "num_hidden_layers", None)
        if (getattr(config, "model_type", None) != "glm4_moe_lite"
                or expected_layers != 47 or len(glm_targets) != expected_layers):
            raise ValueError(
                "GLM TLI 要求真实 glm4_moe_lite 且恰好 47 个 attention 层；"
                f"当前 model_type={getattr(config, 'model_type', None)!r}、"
                f"num_hidden_layers={expected_layers!r}、匹配层数={len(glm_targets)}")
        if getattr(config, "num_attention_heads", None) != 20 or any(
                m.qk_head_dim != 256 or m.qk_nope_head_dim != 192
                or m.qk_rope_head_dim != 64 or m.v_head_dim != 256
                for m in glm_targets):
            raise ValueError("GLM TLI 要求 20 heads 且 K=V=256=[NoPE192|RoPE64]")
        if (getattr(args, "tli_subspace", "full") != "nope"
                or getattr(args, "tli_l2_subspace", "same") != "same"
                or getattr(args, "tia_level2_cmp_ratio", None) != 4
                or getattr(args, "tli_l2_quant", False)
                or getattr(args, "tli_static_pair", False)
                or getattr(args, "tli_proj_basis", None)
                or getattr(args, "tli_proj_basis_all", None)):
            raise ValueError(
                "GLM D=256 仅支持 nope + l2_subspace=same + cmp_ratio=4 "
                "+ l2_quant=False，且禁用 static_pair/投影基；"
                "RoPE-only tail32 已证实丢失 NoPE 检索内容")
        unsupported = []
        if (getattr(args, "tli_far_skip_backend", "legacy") != "legacy"
                or getattr(args, "tli_far_skip_cache", "retain") != "retain"):
            unsupported.append("far-skip/compact")
        if os.environ.get("TLI_SPARSE_PREFILL", "0") == "1":
            unsupported.append("sparse-prefill")
        if os.environ.get("TLI_L2_BYPASS", "0") == "1":
            unsupported.append("L2-bypass")
        if os.environ.get("TLI_BATCH_INDEXER", "legacy") != "legacy":
            unsupported.append("parallel-batch-indexer")
        if unsupported:
            raise ValueError("GLM TLI 不支持配置: " + ", ".join(unsupported))
        if os.environ.get("TLI_CUDA_GRAPH", "0") == "1":
            devices = {
                str(device) for device in getattr(model, "hf_device_map", {}).values()
                if isinstance(device, (int, str)) and str(device) not in ("cpu", "disk")
            }
            if len(devices) > 1:
                raise ValueError("GLM TLI CUDA Graph 仅支持单卡；多卡仅允许 eager")
        # 仅在真实 GLM 且全部预扫通过后局部导入，保持 TF 4.56.2 可导入本模块。
        from .glm4_moe_lite_attn_patch import glm4_moe_lite_attn_forward
        glm_forward = glm4_moe_lite_attn_forward
    if getattr(args, "tli_far_skip_backend", "legacy") == "compact":
        targets = [m for m in model.modules()
                   if isinstance(m, (Qwen3Attention, LlamaAttention)) or _is_deepseek_attn(m)]
        if args.method != "tli" or not targets or any(
                not isinstance(m, Qwen3Attention) or m.head_dim != 128
                or m.sliding_window is not None for m in targets):
            raise ValueError("compact far skip currently supports only TLI Qwen3, "
                             "head_dim=128, without native sliding-window layers")
    if args.method != 'none' and IndexerType is None:
        raise ValueError(f"not support {args.method}")
    if getattr(args, 'tli_far_skip_backend', 'legacy') == 'compact' and any(
        isinstance(m, LlamaAttention) or _is_deepseek_attn(m) for m in model.modules()
    ):
        raise ValueError('compact far-skip storage currently requires the Qwen3 consumer')
    # 082（TL-E121-LLM-EVAL-SNAPSHOT-082）：tli 臂不带 snapshot 走
    # register_patch 是 081 快照契约的旁路（legacy 路径仅测试/兼容
    # 保留）——loudly 警告，防新增入口无意绕过快照。生产入口
    # （pred.py / pred_ruler.py / llm_eval.py）均已注入 snapshot。
    if args.method == 'tli' and snapshot is None:
        print("[TLI][082-WARN] register_patch 未注入 treatment snapshot——"
              "各层将按旧路径重开 D′ 掩码/投影基文件（081 快照契约旁路，"
              "仅测试/兼容路径合法），生产入口须先 "
              "resolve_treatment_snapshot(args) 并透传")
    # A5（TL-E137-LLAMA-NOOP-146）：fail-closed 注册预扫。LlamaAttention
    # 与 Qwen3Attention 注入相同的 TLI/E137 参数，但 llama3_attn_forward
    # 从不调用 calibrate_dyn_dgate——--tli_dyn_dgate 在 Llama 上是静默
    # no-op 且结果仍带稀疏方法标签（实验结论污染）。检测到 Llama 目标类
    # 且声明动态 gate 时显式拒绝（在任何 indexer 构造/patch 动作前）。
    if getattr(args, "tli_dyn_dgate", False) and any(
        isinstance(m, LlamaAttention) for _, m in model.named_modules()
    ):
        raise RuntimeError(
            "Llama patch 未实现 E137 动态标定，拒绝静默 no-op——如需 Llama "
            "动态 gate 请先实现 llama3_attn_patch 的 calibrate 调用"
            "（当前 llama3_attn_forward 不调用 calibrate_dyn_dgate，"
            "--tli_dyn_dgate 在 Llama 上不会生效，继续运行会产出带动态 "
            "gate 标签的 dense 语义结果）"
        )
    # 【E142a】同口径 MLA 预扫：deepseek_v2_attn_forward 不调用
    # calibrate_dyn_dgate（E137 标定是 Qwen3 128 维 rotate_half 布局口径，
    # MLA 重校准是后续课题）——声明动态 gate + DeepSeek 目标类时显式拒绝。
    if getattr(args, "tli_dyn_dgate", False) and any(
        type(m).__name__.startswith("DeepseekV2") and _is_deepseek_attn(m)
        for _, m in model.named_modules()
    ):
        raise RuntimeError(
            "DeepSeek MLA patch 未实现 E137 动态标定（128 维口径不可直接 "
            "迁移，MLA 重校准是后续课题），拒绝静默 no-op——如需 MLA 动态 "
            "gate 请先实现 calibrate 的 MLA 版本"
        )
    n_patched = 0
    if IndexerType is not None:
        for name, module in model.named_modules():
            if isinstance(module, Qwen3Attention):
                module.indexer = IndexerType(args, snapshot=snapshot) \
                    if (snapshot is not None and IndexerType is TLIIndexer) \
                    else IndexerType(args)
                if hasattr(module, "layer_idx"):
                    module.indexer.layer_idx = module.layer_idx  # TLI D' 层掩码用
                module.forward = MethodType(qwen3_attn_forward, module)
                n_patched += 1
                if hasattr(module, "layer_idx") and module.layer_idx == 0:
                    print(f"register qwen3 attn: [{name}]")
            elif glm_forward is not None and _is_glm4_moe_lite_attn(module):
                module.indexer = IndexerType(args, snapshot=snapshot) \
                    if snapshot is not None else IndexerType(args)
                module.indexer.layer_idx = module.layer_idx
                module.indexer.head_dim = 256
                module.forward = MethodType(glm_forward, module)
                n_patched += 1
                if module.layer_idx == 0:
                    print(f"register glm4_moe_lite attn (D=256): [{name}]")
            elif type(module).__name__.startswith("DeepseekV2") and _is_deepseek_attn(module):
                # 【E142a】DeepSeek-V2 MLA（trust_remote_code 动态类，类名
                # 匹配——DeepseekV2Attention / FlashAttention2 / SdpaAttention
                # 三子类同投影布局，forward 均被覆写）。MLA 语义门：
                # kv_a_proj_with_mqa + q_head_dim(192) 存在性即 MLA 结构。
                module.indexer = IndexerType(args, snapshot=snapshot) \
                    if (snapshot is not None and IndexerType is TLIIndexer) \
                    else IndexerType(args)
                if hasattr(module, "layer_idx"):
                    module.indexer.layer_idx = module.layer_idx
                module.forward = MethodType(deepseek_v2_attn_forward, module)
                n_patched += 1
                if hasattr(module, "layer_idx") and module.layer_idx == 0:
                    print(f"register deepseek_v2 attn (MLA): [{name}]")
            elif isinstance(module, LlamaAttention):
                module.indexer = IndexerType(args, snapshot=snapshot) \
                    if (snapshot is not None and IndexerType is TLIIndexer) \
                    else IndexerType(args)
                if hasattr(module, "layer_idx"):
                    module.indexer.layer_idx = module.layer_idx
                module.forward = MethodType(llama3_attn_forward, module)
                n_patched += 1
                if hasattr(module, "layer_idx") and module.layer_idx == 0:
                    print(f"register llama3 attn: [{name}]")
    if glm_targets and args.method == "tli" and n_patched != 47:
        raise RuntimeError(f"GLM TLI 注册层数异常：{n_patched}/47")
    if args.method != 'none' and n_patched == 0:
        mod_types = sorted({type(m).__name__ for _, m in model.named_modules()})
        hint = ""
        if any('glm' in t.lower() for t in mod_types):
            hint = ("（检测到 GLM 系模块，但并非完整匹配的 47 层 "
                    "Glm4MoeLiteAttention 生产结构）")
        raise RuntimeError(
            f"method={args.method!r} 但模型中没有任何可 patch 的 "
            f"Qwen3Attention/LlamaAttention/Glm4MoeLiteAttention 模块（patched=0）——继续运行会"
            f"静默产出 dense 结果却被标记为稀疏方法{hint}。请确认模型架构"
            f"或改 --method none。模块类型样本: {mod_types[:8]}"
        )
    return n_patched

