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
    GLM 说明：glm4_moe_lite_attn_patch.py 是无 indexer 调用的半成品
    （MLA 结构），注册它反而会复现「标稀疏跑 dense」，故不注册、
    显式拒绝（GLM 支持留待补完 indexer 集成后再放开）。

    【081（TL-E121-OUTPUT-SNAPSHOT-081）】snapshot：生产入口在模型
    加载前冻结的 TreatmentSnapshot（sparse_attn.info.
    resolve_treatment_snapshot 产出）。tli 臂逐层建 indexer 实例时
    注入同一 snapshot 对象——层实例仍逐层建（layer_idx 各异），但
    D′ 掩码 / 投影基内容不再按层重开文件（旧路径每层各 open 一次，
    patch 中途换文件 → 跨层混代可达）。仅 TLIIndexer 接受注入，
    其他 indexer 类型忽略 snapshot（其无文件型配置）。
    """
    IndexerType = indexer_type_dict.get(args.method, None)
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
    if args.method != 'none' and n_patched == 0:
        mod_types = sorted({type(m).__name__ for _, m in model.named_modules()})
        hint = ""
        if any('glm' in t.lower() for t in mod_types):
            hint = ("（检测到 GLM 系模块：glm4_moe_lite_attn_patch.py 为无 "
                    "indexer 集成的半成品，未注册支持）")
        raise RuntimeError(
            f"method={args.method!r} 但模型中没有任何可 patch 的 "
            f"Qwen3Attention/LlamaAttention 模块（patched=0）——继续运行会"
            f"静默产出 dense 结果却被标记为稀疏方法{hint}。请确认模型架构"
            f"或改 --method none。模块类型样本: {mod_types[:8]}"
        )
    return n_patched

