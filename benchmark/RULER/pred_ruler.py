# #66 RULER 评测适配器：把 KVCache-Factory 预生成的 RULER 数据
# （data/RULER/{4096,8192,16384,32768}/×11 任务×100 条，零外网；65536/131072
# 由 gen_ruler_long.py 循环体扩展生成）喂进本仓库 sparse_attn 管线
# （TIA/TLI/Quest/FullKV 同一 monkeypatch）。
# 口径对齐官方 RULER：prompt 原样（无 chat template）、max_new=64、
# min_length=context+1 防复读、string_match_all 打分（正式打分入口
# score_ruler_formal.py——legacy + #198 指针协议双通道；score_ruler.py
# 直接入口仅支持 legacy-direct 产物，pointer root 下 fail-closed，069）。
# 用法（run_ruler.sh 批量调度）：
#   python -u -m benchmark.RULER.pred_ruler \
#     --model Qwen3-8B --model_path $MODEL \
#     --task niah_single_1 --context_length 8192 --method tli --t $TS \
#     --data-root /home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER \
#     --output-dir exp/results_ruler/Qwen3-8B \
#     --tia_level2_topk 1024 --tia_level2_cmp_ratio 4 --pred_postfix _1024 \
#     --max-num 500
# ---- E109 三件套 #3：YaRN 长上下文扩展（2026-10-09）----
# Qwen3-8B 原生 max_position_embeddings=40960，65536/131072 档须 YaRN。
# 配方按 Qwen3 官方 model card：rope_scaling={yarn, factor, original 40960,
# beta_fast 32, beta_slow 1}；128K 档 factor=4.0（官方原配方，effective
# 163840），64K 档取半档 factor=2.0（effective 81920，留 25% 余量；与官方
# 整档倍率风格一致，且 ramp 插值更平缓、近程精度损失更小）。
# 设计文档口径（逐层参数求解器 §10.4）：dense 与所有稀疏方法同一位置缩放/
# tokenizer/最大输出；先报告 dense 长度能力再看稀疏退化。
# YaRN 对 TLI 索引透明：indexer 消费的是 post-RoPE 的 K（qwen3_attn_patch
# 在 apply_rotary_pos_emb + cache.update 之后传入），位置缩放自动生效。
# logits 显存账（超 32K 档自动 logits_to_keep=1，等价性审计 0233 已 A/B
# 实测 bitwise 相等）：131072×151936 vocab bf16 全序列 logits ≈ 39.8GB 纯
# 浪费（只消费末 token），32K 及以下档路径逐位不变（零扰动纪律）。
# ---- 057（GPT 2026-10-10 0330 审计 TL-E119-YARN-IDENTITY-057）----
# 本脚本此前只在 stdout 打印 effective factor、不落任何原生配置证据，formal
# 汇总只能消费操作者事后 CLI 声明——128K 批次出现「manifest 声明 2.0 vs
# 生成链自动档 4.0」的身份冲突。现每次运行把 effective yarn 配置原子写入
# 预测产物旁挂 receipt（{pred 基名}-yarn_receipt.json，见 yarn_receipt.py），
# 生成路径/行格式/自动档语义零改动。
# ---- 059（GPT 2026-10-10 0428 审计 TL-E119-YARN-RECEIPT-BINDING）----
# v1 回执在生成循环之前落最终旁挂路径、且不含预测内容 SHA/行数/run ID/
# 完成标记——「A 进程的预测配 B 进程的回执」可达（同名重跑/中断重跑/
# 并发/事后改写）。升级 producer-yarn-config-v2：预测写不可变 generation，
# 循环结束关闭后算 SHA256+行数，构建 status=complete 完成回执，经
# 原子提交点落位；全生命周期持规范化输出路径 flock。
# ---- 066/crash-recovery（GPT 2026-10-10 1130 审计，#198）----
# 059 的两步 os.replace（预测先替换最终路径 → 回执后替换旁挂）在
# 「B 预测与 A 字节完全相同」的死亡中间态下不可检（三方 SHA 全等，
# formal 接受 A config 标 verified_same_generation=true）。升级为
# 不可变 generation + 单指针原子切换：预测与完成回执同置
# {out}.gen-{attempt_id}/ 目录，提交 = 单次 os.replace 切指针
# {out}.tli_gen（内容 = gen 目录名；指针切换 = 唯一提交信号，E116f
# 同语义）。崩溃三阶段：指针未切 → 旧完整代可见；指针已切 → 新完整
# 代可见；混合代不可达（预测与回执同目录，消费侧从指针同时读两文件）。
import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime

# 057：生产者原生 effective-config receipt（yarn_receipt.py 零重依赖，
# 生成/消费/测试三方共享同一口径）。REPO 入 sys.path 前置于一切仓库内
# 导入——保证以普通脚本方式（python benchmark/RULER/pred_ruler.py）被
# 调用时 sparse_attn 与包内导入同样可达。
REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig, AutoModelForCausalLM, AutoTokenizer)

from benchmark.RULER.yarn_receipt import (  # noqa: E402
    _file_sha256, acquire_output_lock, build_yarn_receipt,
    commit_yarn_generation, generation_pointer_path, release_output_lock,
    resolve_yarn_config, stage_yarn_generation, stage_yarn_receipt,
)
from sparse_attn.baseline_runtime import baseline_request, load_ffd, baseline_preflight
from sparse_attn.arguments import add_sparse_attn_args  # noqa: E402
from sparse_attn.patches import register_patch  # noqa: E402
from sparse_attn.metrics import get_metrics  # noqa: E402
# 【095 修复】deferred error flag 的请求级消费点（guard 语义/限制见其
# docstring）——RULER 生成的逐行生成段接线。
from sparse_attn.ops.eager_decoding import (  # noqa: E402
    eager_decoding_request_guard)
from sparse_attn.ops.quality_graph import from_dynamic, graph_policy
from sparse_attn.ops.far_skip_runtime import resolve_far_skip_modes  # noqa: E402
from sparse_attn.info import (  # noqa: E402
    get_method_name_with_info, get_treatment_manifest_json,
    resolve_treatment_snapshot)
# 142（TL-FASA-CALIBRATION-IDENTITY）：FASA agreement 源/目标身份的
# 唯一 staging 口径（与 LongBench 入口、treatment 名三方同源）。
from sparse_attn.info import stage_fasa_agreement  # noqa: E402
# 149（TL-MINFERENCE-PATTERN-IDENTITY）：MInference pattern 内容身份
# 的入口冻结（与 LongBench 入口、treatment 名三方同源）。
from sparse_attn.info import freeze_minference_pattern_sha8  # noqa: E402

RULER_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt",
]
MAX_GEN = 64  # 官方 RULER 口径（全部任务统一）
SEED = 42     # seed_everything 默认值（receipt 身份字段，057）
# Qwen3-8B 原生上下文上限（config.max_position_embeddings）
QWEN3_NATIVE_MPE = 40960
# YaRN factor 自动档位（官方 128K 配方 factor 4.0 的等比半档）
YARN_FACTOR_AUTO = {65536: 2.0, 131072: 4.0}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen3-8B")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task", type=str, choices=RULER_TASKS, required=True)
    parser.add_argument("--context_length", type=int,
                        choices=[4096, 8192, 16384, 32768, 65536, 131072],
                        required=True)
    parser.add_argument("--data-root", type=str, required=True,
                        help="KVCache-Factory data/RULER 目录")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--t", type=str, default="ruler run")
    parser.add_argument("--pred_postfix", type=str, default="")
    parser.add_argument("--max-num", type=int, default=500,
                        help="每任务样本数（0=全量）")
    parser.add_argument("--yarn", action="store_true",
                        help="启用 YaRN 位置扩展（Qwen3 官方配方；"
                             "65536/131072 档必开，32768 及以下档保持关闭"
                             "以维持既有口径零扰动）")
    parser.add_argument("--yarn_factor", type=float, default=None,
                        help="YaRN factor（默认按档位自动：65536→2.0、"
                             "131072→4.0；显式传入可覆盖做消融）")
    add_sparse_attn_args(parser)
    return parser.parse_args()


def seed_everything(seed=SEED):   # 057：默认值单源化（receipt 身份字段）
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _extended_receipt_identity(args, data_file):
    """可选的 run 级身份闭包；未设置环境变量时保持既有 RULER 路径。"""
    config_path = os.environ.get("TLI_RECEIPT_RUN_CONFIG")
    if not config_path:
        return {}
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    def checked_files(root, expected, label):
        if not isinstance(expected, dict) or not expected:
            raise SystemExit(f"[receipt] {label} 缺少冻结文件 SHA 清单")
        actual = {}
        for rel, expected_sha in sorted(expected.items()):
            if os.path.isabs(rel) or ".." in rel.split("/"):
                raise SystemExit(f"[receipt] {label} 含非法相对路径: {rel!r}")
            path = os.path.join(root, rel)
            actual_sha = _file_sha256(path)
            if actual_sha != expected_sha:
                raise SystemExit(
                    f"[receipt] {label}/{rel} 当前 SHA={actual_sha}，"
                    f"冻结 SHA={expected_sha}，拒绝生成混代结果")
            actual[rel] = actual_sha
        return actual

    source_files = checked_files(
        REPO, config.get("source", {}).get("files"), "source.files")
    weight_files = checked_files(
        args.model_path, config.get("model", {}).get("weight_files_sha256"),
        "model.weight_files_sha256")
    source_rel = f"{args.context_length}/{args.task}.jsonl"
    input_sha = _file_sha256(data_file)
    expected_input = config.get("inputs_sha256", {}).get(source_rel)
    if input_sha != expected_input:
        raise SystemExit(
            f"[receipt] 输入 {source_rel} 当前 SHA={input_sha}，"
            f"冻结 SHA={expected_input}，拒绝生成")
    return {
        "run_config_sha256": _file_sha256(config_path),
        "input_data": {"relative_path": source_rel, "sha256": input_sha},
        "source_files_sha256": source_files,
        "model_weight_files_sha256": weight_files,
    }


def main():
    args = parse_args()
    baseline_preflight(args)
    seed_everything()

    # 身份来自真实 config，GLM 不复用 Qwen 的 40960/YaRN 口径。
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    is_glm = getattr(config, "model_type", None) == "glm4_moe_lite"
    native_mpe = (getattr(config, "max_position_embeddings", None)
                  if is_glm else QWEN3_NATIVE_MPE)
    # ---- YaRN 校验与配置（E109 三件套 #3 + 057 effective 单点解析）----
    # 超过原生 40960 的档位必须显式 --yarn（防误用原生 RoPE 跑超限位置）；
    # 原生档位默认关闭（既有 4K-32K 路径零扰动），显式开也允许（消融用）。
    use_yarn = args.yarn
    if is_glm:
        if native_mpe != 202752:
            raise SystemExit(f"[glm] 原生 MPE 应为 202752，实际 {native_mpe!r}")
        if args.yarn or args.yarn_factor is not None:
            raise SystemExit("[glm] GLM-4.7-Flash 原生覆盖 32K/64K/128K，禁止 --yarn/--yarn_factor")
        use_yarn = False
    # llama 豁免（champ 树 33cac0 原语义，2026-10-22 恢复）：本门禁按
    # Qwen3-8B 原生 40960 硬编码，不适用于 Llama-3.1（原生
    # max_position_embeddings=131072，64K/128K 档无需 YaRN，yarn=0 是
    # Llama 派单正确口径）。10-22 pc40 全 Llama 任务（yarn=0）在此门禁
    # 速死实证——误删豁免 = 部署污染源。
    if (not is_glm and "llama" not in args.model.lower()
            and args.context_length > QWEN3_NATIVE_MPE and not use_yarn):
        raise SystemExit(
            f"[yarn] context_length={args.context_length} 超出 Qwen3-8B 原生 "
            f"{QWEN3_NATIVE_MPE}，必须加 --yarn（run_ruler_e109.sh 已自动）")
    # 057：effective 配置单点解析（修复前 print 与 config 构造两处内联各
    # 算一遍；resolve_yarn_config 与原内联语义逐位一致——显式 factor 覆盖，
    # 否则档位自动表回退），后续 receipt/config 全部消费同一份结果，
    # 「实际生效值」从此有生产者原生证据。
    factor, rope_scaling = resolve_yarn_config(
        use_yarn, args.yarn_factor, args.context_length,
        YARN_FACTOR_AUTO, native_mpe)
    if use_yarn:
        print(f"[yarn] enabled: factor={factor} "
              f"(effective context = {int(QWEN3_NATIVE_MPE * factor)})")

    data_file = os.path.join(args.data_root, str(args.context_length),
                             f"{args.task}.jsonl")
    receipt_identity = _extended_receipt_identity(args, data_file)
    rows = [json.loads(l) for l in open(data_file)]
    if args.max_num and len(rows) > args.max_num:
        rows = rows[:args.max_num]  # topk 口径（确定性）
    print(f"[ruler] task={args.task} L={args.context_length} "
          f"n={len(rows)} method={args.method}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True)
    # YaRN：在模型加载前把 rope_scaling 写进 config（transformers 从
    # config 构造 YarnRotaryEmbedding；同时写 rope_type/type 兼容新旧字段）
    if use_yarn:
        config.rope_scaling = rope_scaling
    resolve_far_skip_modes(args, config=config)
    snapshot = resolve_treatment_snapshot(args) if args.method == "tli" else None
    if args.method == 'ffd_vendor':
        if use_yarn:
            raise ValueError('FFD Llama cannot use Qwen3-specific YaRN defaults')
        model = load_ffd(args.model_path, args, config=config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, config=config, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto").eval()
        print(f"[device-map] {getattr(model, 'hf_device_map', None)}")
    # ---- Quest vendor（B 轨官方源码复用，主表口径）----
    # 不走 register_patch 的 indexer_type_dict（A 轨 quest_indexer 是语义
    # 复现），改为绑定官方 Quest forward 到 Qwen3Attention——与 LongBench
    # pred.py 同口径：quest_attention_official.py 与官方仓库逐字节一致，
    # qwen3_quest_patch.py 仅接口胶水。RULER 长上下文预算独立设定（默认
    # 1024，派单时按 Quest 论文口径显式传 --quest_token_budget）。
    if args.method == "ffd_vendor":
        n_patched = model.config.num_hidden_layers
    elif args.method == "quest_vendor":
        vendor_dir = os.path.join(
            REPO, "exp", "vendor_baselines", "quest_vendor")
        if vendor_dir not in sys.path:
            sys.path.insert(0, vendor_dir)
        from qwen3_quest_patch import enable_quest  # noqa: E402
        n_patched = enable_quest(
            model,
            token_budget=getattr(args, "quest_token_budget", 1024),
            chunk_size=getattr(args, "quest_chunk_size", 16),
        )
    elif args.method == "minference_vendor":
        # MInference（NeurIPS 2024 sparse prefill）B 轨官方源码复用——与
        # LongBench pred.py 同口径：qwen3_minference_patch.py 轻量胶水只
        # 替换 Qwen3Attention.forward（prefill sparse / decode dense），
        # pattern 为 36 层×32 头 best_pattern_qwen3_8b.json（在线搜索产物）。
        vendor_dir = os.path.join(
            REPO, "exp", "vendor_baselines", "minference_vendor")
        if vendor_dir not in sys.path:
            sys.path.insert(0, vendor_dir)
        from qwen3_minference_patch import enable_minference  # noqa: E402
        cfg_path = getattr(args, "minference_config_path", None)
        if not cfg_path or not os.path.isfile(cfg_path):
            raise RuntimeError(
                "method=minference_vendor 需要 --minference_config_path "
                "指向 pattern json（如 best_pattern_qwen3_8b.json）")
        # 149（TL-MINFERENCE-PATTERN-IDENTITY，GPT 审计 8b56bbd + C+ 批
        # 并发残余）：入口加载期单次冻结 pattern 完整内容身份快照（sha8
        # + raw bytes + 解析后 JSON）——treatment 名尾段 _cfg<sha8> 消费
        # 冻结值；adapter（enable_minference）同样消费冻结对象构建
        # best_pattern，不再二次 open 可变 config_path（旧实现窗口内
        # 源被替换则 name 绑 A、runtime 读 B，与 142 同构）。
        freeze_minference_pattern_sha8(args)
        frozen_snap = getattr(args, "_minference_pattern_frozen", None)
        n_patched = enable_minference(
            model,
            is_search=False,
            config_path=cfg_path,
            starting_layer=getattr(args, "minference_starting_layer", -1),
            frozen_pattern=(frozen_snap.pattern
                            if frozen_snap is not None else None),
        )
    elif args.method == "fasa_vendor":
        # FASA（ICLR 2026 frequency-aware sparse attention）B 轨官方源码
        # 复用——qwen3_fasa_patch.py 仅接口胶水，fasa_utils.py 与官方
        # utils.py 逐字节一致（constructe_adaptive_selection /
        # core_module_with_padding_old）。官方解码剪枝相对路径依赖
        # dimension_rankings/<model>/<dataset>_agreement.pkl（相对 CWD
        # 解析）——C+ 批 142 并发残余 + sha8 前缀碰撞残余（P2）修复后，
        # stage 物化目标是内容寻址进程私有目录
        # dimension_rankings_staged/<full_sha256>/（共享固定路径下并发
        # 首 stage 不同内容会后者覆盖前者、先到进程的官方 loader 读到
        # 后者内容；sha8 前缀碰撞档同构，故目录段用完整 SHA256，sha8
        # 只作名字显示标识），adapter（enable_fasa）调官方函数前把 CWD 切
        # 到 staged 根（try/finally 恢复），官方相对路径即解析到内容
        # 寻址副本；旧 REPO 根 dimension_rankings/ 不再被读写。
        vendor_dir = os.path.join(
            REPO, "exp", "vendor_baselines", "fasa_vendor")
        if vendor_dir not in sys.path:
            sys.path.insert(0, vendor_dir)
        from qwen3_fasa_patch import enable_fasa  # noqa: E402
        fasa_dataset = getattr(args, "fasa_dataset", "qasper")
        # 142（TL-FASA-CALIBRATION-IDENTITY，GPT 审计 B 批）：--fasa_
        # agreement_dir 此前注册后从未被消费（源/目标硬编码 vendor 内
        # dimension_rankings/ + 仓库根），且「目标已存在时不比较」→
        # 不同校准内容静默共享同一 treatment 身份。现统一走
        # sparse_attn/info.py 的 142 三函数（与 LongBench 入口、
        # treatment 名三方同源）：agreement 源目录 = --fasa_agreement_dir
        # （默认 'dimension_rankings' 相对 vendor 目录解析，向后兼容）；
        # 内容寻址私有目标已存在时源/目标 SHA256 对比（被篡改/损坏的
        # stale 目标）fail-closed。
        staged_root = stage_fasa_agreement(args, REPO)
        n_patched = enable_fasa(
            model,
            agreement_dir=staged_root,
            model_name=args.model,
            dataset_name=fasa_dataset,
            budget=getattr(args, "fasa_budget", 1024),
            threshold_ratio=getattr(args, "fasa_threshold_ratio", 0.8),
            min_k=getattr(args, "fasa_min_k", 16),
            max_k=getattr(args, "fasa_max_k", 16),
            mode=getattr(args, "fasa_mode", "topk"),
        )
    elif args.method == "sparq_vendor":
        # SparQ（bandwidth-efficient sparse attention）B 轨官方源码复用——
        # qwen3_sparq_patch.py 仅接口胶水（GQA repeat_kv 展开 + V_mean
        # dtype/device 迁移 + DynamicCache 4.56 全量 slice 适配），核心
        # sparq_attn 来自 sparq_attention_official.py（与 graphcore
        # sparq-gpt-fast 的 sparq.py 逐字节一致，md5=522a085c）。
        # 主表口径（用户 2026-10-14 拍板）：k 固定 1024；r 随序列长度
        # 由 ratio=context_length/(2*k) 反推，context_length 默认跟随任务
        # --context_length（128K RULER → r=2），可用 --sparq_context_length
        # 显式覆盖。
        vendor_dir = os.path.join(
            REPO, "exp", "vendor_baselines", "sparq_vendor")
        if vendor_dir not in sys.path:
            sys.path.insert(0, vendor_dir)
        from qwen3_sparq_patch import enable_sparq  # noqa: E402
        # --sparq_context_length 注册默认 None：getattr 返回 None 而非默认值，
        # 须显式回退到任务 --context_length（RULER 各 L 档位）。生效 ctx 存
        # _sparq_ctx_effective 供 treatment 名编码（r 随序列长度变，不同
        # ctx = 不同 treatment，防 32K/64K/128K 同名互覆）。
        _sparq_cl = getattr(args, "sparq_context_length", None)
        _sparq_ctx = (_sparq_cl if _sparq_cl else args.context_length)
        setattr(args, "_sparq_ctx_effective", _sparq_ctx)
        n_patched = enable_sparq(
            model,
            k=getattr(args, "sparq_k", 1024),
            context_length=_sparq_ctx,
        )
    elif args.method == "moba_vendor":
        # MoBA（Mixture of Block Attention）B 轨官方源码复用——官方语义：
        # prefill 块级 gate 稀疏（chunk-mean key gate → per-query top-k 选块，
        # 尾 chunk 总是选）+ decode dense（官方 wrapper.py:54-82）。
        # qwen3_moba_patch.py 仅接口胶水（GQA 展开 + 形状搬运），核心
        # moba_attn_varlen 来自 moba_attention_official.py（=
        # third_party/MoBA/moba/moba_efficient.py 逐字节一致，md5=da636efc）。
        # 主表预算口径对齐 K2=1024：topk 默认 None → 换算
        # max(2, ceil(budget/chunk))。官方默认 chunk=4096 → topk=2 → 预算 8192
        # （块粒度下限，官方 efficient 内部 topk-1，topk=1 退化为 dense）；
        # 严格 1024 须改 chunk（如 256→topk=4），论文标注 budget-normalized 消融。
        vendor_dir = os.path.join(
            REPO, "exp", "vendor_baselines", "moba_vendor")
        if vendor_dir not in sys.path:
            sys.path.insert(0, vendor_dir)
        from qwen3_moba_patch import enable_moba  # noqa: E402
        n_patched = enable_moba(
            model,
            chunk_size=getattr(args, "moba_chunk_size", 4096),
            topk=getattr(args, "moba_topk", None),
            budget=getattr(args, "moba_budget", 1024),
        )
    elif args.method == "asynctls_vendor":
        # AsyncTLS（arXiv:2604.07815，人大+美团）C 轨语义复现——官方尚无开源
        # 仓库（用户 2026-10-17 授权「无源码则读论文给出近似实现」），三级流水：
        # Stage-1 块粗筛（B=64 top-kb=128 块符号感知 GEMM 打分）+ Stage-2 channel
        # 细筛（Double-Sparsity 校准 top-dc=32 channel + INT4 key + 组内平均
        # softmax）+ Stage-3 全维 KV sparse attn（kt=token_budget）。
        # qwen3_asynctls_patch.py 全为语义复现（_baseline_identity 已标注非官方）。
        # 校准文件（--asynctls_channel_file）必填：identify_channel.py 离线产出。
        vendor_dir = os.path.join(
            REPO, "exp", "vendor_baselines", "asynctls_vendor")
        if vendor_dir not in sys.path:
            sys.path.insert(0, vendor_dir)
        from qwen3_asynctls_patch import enable_asynctls  # noqa: E402
        _atls_chan = getattr(args, "asynctls_channel_file", None)
        if not _atls_chan or not os.path.isfile(_atls_chan):
            raise RuntimeError(
                "method=asynctls_vendor 需要 --asynctls_channel_file 指向校准产物"
                "（exp/vendor_baselines/asynctls_vendor/identify_channel.py 生成）")
        n_patched = enable_asynctls(
            model,
            channel_file=_atls_chan,
            token_budget=getattr(args, "asynctls_token_budget", 1024),
            block_size=getattr(args, "asynctls_block_size", 64),
            top_kb=getattr(args, "asynctls_top_kb", 128),
        )
    elif args.method == "hisa_vendor":
        # HISA（COLM 2026，arXiv:2603.28458v4）B 轨官方 kernel 语义 torch 等价——
        # 官方 GPU kernels 为 TileLang（tile-ai/tilelang examples/dsa_hisa，
        # pinned 7940102813，MIT，third_party/tilelang_dsa_hisa/ 逐字落地）。
        # 选择器两级：Stage-0/1 块均值池化 + Q·BlockedK 线性打分 + 首/末块 +inf
        # 保护 + top-block_topk；Stage-2 块内全分辨率打分 + top-hisa_budget。
        # Qwen3 适配：GQA 组内平均替代 DSA per-head weights、bf16 均值替代 fp8
        # （identity 标注，统一后端机制轨）。qwen3_hisa_patch.py 按官方 kernel
        # 语义用纯 torch 实现，非官方文件逐字节复用。
        vendor_dir = os.path.join(
            REPO, "exp", "vendor_baselines", "hisa_vendor")
        if vendor_dir not in sys.path:
            sys.path.insert(0, vendor_dir)
        from qwen3_hisa_patch import enable_hisa  # noqa: E402
        n_patched = enable_hisa(
            model,
            hisa_budget=getattr(args, "hisa_budget", 1024),
            k_block_size=getattr(args, "hisa_k_block_size", 128),
            block_topk=getattr(args, "hisa_block_topk", 64),
        )
    elif args.method == "jssa_vendor":
        # JSSA（Joint Subspace Sparse Attention，arXiv:2610.19026v1）B 轨官方
        # vLLM patch 语义 torch 等价——官方实现为 vLLM v1 attention backend
        # （third_party/JSSA/vllm_patch/jssa_vllm.patch）。语义：decode 单级
        # token 稀疏，校准基 M=TopEig_r(G_Qm/tr+G_K/tr)（r=32）投影打分 top-
        # budget/head + 精确注意力，prefill dense。qwen3_jssa_patch.py 按官方
        # 语义纯 torch 等价实现（identity 标注）。basis_path 必填（None=拒绝
        # 启动，GATE-FAIL）。
        vendor_dir = os.path.join(
            REPO, "exp", "vendor_baselines", "jssa_vendor")
        if vendor_dir not in sys.path:
            sys.path.insert(0, vendor_dir)
        from qwen3_jssa_patch import enable_jssa  # noqa: E402
        n_patched = enable_jssa(
            model,
            basis_path=getattr(args, "jssa_basis_path", None),
            budget=getattr(args, "jssa_budget", 1024),
            rank=getattr(args, "jssa_rank", 32),
        )
    elif args.method == "clusterkv_vendor":
        # ClusterKV（KMeans 聚类稀疏）B 轨官方源码复用——官方语义：
        # prefill dense（官方 flash_forward 分支）；decode 后期对 prefill key
        # pylibraft KMeans 建簇（build_cluster）→ query×centroids 最近邻选簇到
        # token_budget−sink → sel_key_indices → +sink 偏移 → gather + 精确
        # matmul 细筛（cluster_attn_out，官方算法逐字节一致）。
        # qwen3_clusterkv_patch.py 仅接口胶水（Qwen3 q/k RMSNorm、GQA 展开、
        # cache 更新）；官方 cluster_attention.py 从
        # third_party/ClusterKV/accuracy 直接 import（md5 冻结）。官方
        # `_clusterkv_knl.search_indices` 编译内核在本机不可用（网络阻塞），
        # 由 vendor 包 clusterkv/_clusterkv_knl.py 纯 torch 等价实现替代
        # （语义与官方 KMeansSearchIndicesKernel 一致），官方文件零改动。
        # 预算口径：官方 token_budget 是「选择预算」（sink 额外），与 TLI
        # K2=1024 已含 sink+swa 的口径不同，接入验收时按 unique token 数核对。
        vendor_dir = os.path.join(
            REPO, "exp", "vendor_baselines", "clusterkv_vendor")
        if vendor_dir not in sys.path:
            sys.path.insert(0, vendor_dir)  # vendor clusterkv/ stub 必须最前
        ckv_root = os.path.join(REPO, "third_party", "ClusterKV")
        if ckv_root not in sys.path:
            sys.path.append(ckv_root)  # 官方树放 vendor 之后
        from qwen3_clusterkv_patch import enable_clusterkv  # noqa: E402
        n_patched = enable_clusterkv(
            model,
            token_budget=getattr(args, "clusterkv_token_budget", 1024),
            sink=getattr(args, "clusterkv_sink", 16),
            nlist=getattr(args, "clusterkv_nlist", 400),
            fit_iter=getattr(args, "clusterkv_fit_iter", 20),
            head_sel=getattr(args, "clusterkv_head_sel", "truc"),
            balance=getattr(args, "clusterkv_balance", False),
            dist_t=getattr(args, "clusterkv_dist_t", "cosine"),
            gqa_policy=getattr(args, "clusterkv_gqa_policy", None),
            topk_stat=getattr(args, "clusterkv_topk_stat", False),
        )
    else:
        n_patched = register_patch(model, args, snapshot)
    if args.method != "none" and n_patched == 0:
        raise RuntimeError(
            f"method={args.method!r} 但成功挂载 0 个 attention 模块，模型"
            f"实际将以 dense 运行——拒绝产出被误标为稀疏方法的结果。"
        )

    # 超原生档位启用 logits_to_keep=1：prefill 只回传末 token logits
    # （131072×151936 bf16 全序列 logits ≈ 39.8GB 纯浪费；等价性由审计
    # 0233 A/B 实测 bitwise 相等保证；≤32K 档不启用，路径逐位不变）
    keep_logits = args.context_length > 32768

    eos_ids = [tokenizer.eos_token_id]
    nl = tokenizer.encode("\n", add_special_tokens=False)
    if nl:
        eos_ids.append(nl[-1])

    out_dir = os.path.join(args.output_dir, f"L{args.context_length}",
                           f"pred{args.pred_postfix}")
    os.makedirs(out_dir, exist_ok=True)
    # 081：method_name/out_path 从冻结 snapshot 派生（tli 臂）——与
    # runtime 各层、receipt 的 treatment manifest 同一内容身份。
    method_name = get_method_name_with_info(args, snapshot)
    out_path = os.path.join(
        out_dir, f"{args.task}-{method_name}-{args.t}.jsonl")
    # 076（TL-E121-OUTPUT-ID）：method_name 尾段含 canonical treatment
    # hash _h<hash10>（sparse_attn/info.py——全部输出相关参数排序键
    # JSON→sha256）→ 逻辑路径与 {out}.tli_gen 指针均为 treatment 单射：
    # 异配置不可能命中同一路径（「不切指针」结构性成立），同路径重跑
    # = 同 treatment 的 066/crash-recovery 设计语义（新 gen + 单次原子
    # 切指针）。残余 10-hex 碰撞由回执 v2 同代绑定（062 三方 SHA 一致）
    # 与消费侧 effective_config_sha256 门禁纵深兜底，不在生成侧重复比对。

    # ---- 059 + 066/crash-recovery：不可变 generation + 单指针原子提交 ----
    # 修复前缺陷链：①（059 前）直接以 open(..., "w") 截断最终预测路径
    #   ——生成中途崩溃毁掉上一代完整 best-file；②（059 前）回执在生成
    #   循环之前落最终旁挂路径且无完成标记/预测绑定；③（066/crash-
    #   recovery）059 的两步 os.replace 在「B 预测与 A 字节完全相同」
    #   的死亡中间态下不可检——第一步后、第二步前死亡 → 盘上为「B
    #   物理写入的预测 + A 旧回执」，三方 SHA 全等，formal 接受 A
    #   config 标 verified_same_generation=true（混合代不可检）。
    # 设计（E116f generation/单指针提交同款语义 + 056 flock 口径）：
    #   ① 全生命周期持规范化输出路径锁（realpath(父目录)+basename 键，
    #      042 加固防锁键漂移）——覆盖「gen 目录写入 → SHA/行数计算 →
    #      完成回执 → 指针切换」全程，后到者阻塞等待，崩溃由内核
    #      自动释放；
    #   ② 预测与完成回执同置不可变 generation 目录
    #      {out}.gen-{attempt_id}/（目录名不以 .jsonl 结尾 → 不进
    #      {task}-*.jsonl glob——中断残留不污染 best-file 仲裁 / SKIP
    #      幂等的行数口径）；
    #   ③ 循环结束 fout.close() 后计算预测 SHA256 + 最终行数，构建
    #      status=complete 完成回执（producer-yarn-config-v2，含
    #      prediction_basename/SHA/行数/run_id）写进同一 gen 目录；
    #   ④ 单次提交点：commit_yarn_generation 单次 os.replace 原子切
    #      指针 {out}.tli_gen（内容 = gen 目录名；指针切换 = 唯一
    #      提交信号）。
    # 崩溃语义（066/crash-recovery 验收口径）：gen 未就绪/指针未切 →
    #   旧完整代可见（指针仍指旧 gen）；指针已切 → 新完整代可见；
    #   混合代不可达（预测与回执同目录，消费侧从指针同时读两文件）。
    #   注：059 时代 docstring 的「两步之间死亡 → SHA 必失配」前提在
    #   同字节场景不成立（066），已废止。
    attempt_id = (datetime.now().strftime("%Y%m%d%H%M%S") +
                  f"-{os.getpid()}-{random.randint(1000, 9999)}")
    lock_fd = acquire_output_lock(out_path)
    try:
        tmp_pred = stage_yarn_generation(out_path, attempt_id)
        fout = open(tmp_pred, "w", encoding="utf-8")
    except OSError as e:
        release_output_lock(lock_fd)
        raise SystemExit(f"[pred] 打开临时 generation 失败（{e}）")

    try:
        for row in tqdm(rows):
            # 【2026-10-22 修复·跨行 KV 泄漏 v2】行结束后 past/output/out
            # 三个名字都可能持有 DynamicCache 引用——尤其 decode 循环的
            # `out`（末次 decode 输出，out.past_key_values 即整条 KV cache）。
            # 任一引用存活则 empty_cache() 无法回收旧 dyn KV——row2 prefill
            # 时旧 dyn(34.4G@32B/128K) + 新 dyn 增长 + 权重 65G 实测
            # 133.79G + MLP 激活 6.25G > 139.7G 物理 OOM（pc32 128K 六任务
            # ×3 attempts 全灭实锤；v1 只清 past/output 漏 out 仍死；8B/Llama
            # KV 小无感）。
            past = None
            output = None
            out = None
            torch.cuda.empty_cache()
            input_ids = tokenizer(row["input"], truncation=False,
                                  return_tensors="pt").input_ids.to("cuda")
            context_length = input_ids.shape[-1]
            # 【095 修复（GPT 审计 2026-10-12_0527，2026-10-12）】请求级
            # fail-closed guard：TLI_E127_FAST=1 的 gather 路径把三类契约
            # 违规（空 mask 行/W 超上界/cu 破坏）延迟到 GPU 端错误标志，
            # 由 guard 在本行生成段边界（写盘前）一次性校验 raise——raise
            # 传播后 finally 只释放 flock，generation 目录不 commit、
            # .tli_gen 指针不切（066/crash-recovery 语义天然吸收：旧完整
            # 代可见，无任何完成外观产物被污染）。进入时清旧 flag 防跨
            # 行继承（上一行未消费的违规不误杀本行）。
            with baseline_request(model) as baseline_receipt, eager_decoding_request_guard():
                with torch.no_grad():
                    output = model(input_ids=input_ids, past_key_values=None,
                                   use_cache=True,
                                   **({"logits_to_keep": 1} if keep_logits else {}))
                    past = output.past_key_values
                    pred_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    gen = [pred_idx.item()]
                    del output
                    graphed = from_dynamic(model, past, gen[0], MAX_GEN, eos_ids)
                    if graphed is not None:
                        gen = graphed
                    else:
                        for _ in range(MAX_GEN - 1):
                            # min_length 语义：禁止在 context 结束前停（RULER 官方口径）
                            if len(gen) < 2 and pred_idx.item() in eos_ids:
                                pass  # 首位不许停（近似 min_length=context+1）
                            out = model(input_ids=pred_idx, past_key_values=past,
                                        use_cache=True)
                            past = out.past_key_values
                            pred_idx = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                            gen.append(pred_idx.item())
                            if pred_idx.item() in eos_ids and len(gen) > 1:
                                break
            pred = tokenizer.decode(gen, skip_special_tokens=True)
            metrics = get_metrics()
            budget = None if (baseline_receipt or getattr(model, "_quality_graph_execution", {}).get("actual", "").startswith("graph")) else metrics.get_select_tokens()
            metrics.clear()
            # E116c（GPT 0826 审计 TL-RULER-SAMPLE-GATE-026）：样本身份绑定源行
            # index（RULER 源 jsonl 自带 0..99 int，比 LB v1 的 data_fp 指纹更直接）
            # + answers canonical SHA256 前 16 位（与 LongBench manifest 口径一致），
            # 评分端据此做 fail-closed 集合闭包校验；其余字段与行为零改动。
            fout.write(json.dumps({
                **({"baseline_receipt": baseline_receipt} if baseline_receipt else {}),
                "pred": pred, "answers": row["outputs"],
                "length": row["length"], "budget": budget,
                "execution": getattr(model, "_quality_graph_execution", {}),
                "_id": f"{args.task}:{row['index']}",
                "_answers_sha": hashlib.sha256(json.dumps(
                    row["outputs"], ensure_ascii=False, sort_keys=True)
                    .encode("utf-8")).hexdigest()[:16],
            }, ensure_ascii=False) + "\n")
            fout.flush()
        fout.close()

        # ---- 059+066：完成回执（v2 同代绑定）→ 同 gen 目录 → 切指针 ----
        # 057 的配置指纹字段全部保留（effective factor/完整 rope_scaling/
        # 模型路径与 config hash/生成参数/生产脚本身份）；v2 新增同代绑定
        # 四件：status=complete + prediction_basename/SHA256/行数 + run_id
        # ——formal 消费侧据指针解析 generation 后同时读预测与回执，
        # 三方一致校验（062 纵深）+ 指针同源保证（066/crash-recovery）。
        pred_sha = _file_sha256(tmp_pred)
        pred_lines = sum(1 for _ in open(tmp_pred, "rb"))
        model_cfg_path = os.path.join(args.model_path, "config.json")
        # 079（TL-E121-OUTPUT-ID-079）：tli 臂把 resolved treatment
        # manifest 写入回执——与 method hash（method_name 尾段）、
        # LongBench sidecar 共用 sparse_attn/info.py 同一份 resolved
        # manifest（单一事实源），本入口不再各自维护治疗身份字段子集；
        # 非 tli 臂（none/quest/twia/tia）treatment 身份由可读名整体
        # 编码，不写。schema 校验 sha256(json) 自洽（yarn_receipt 079）。
        # 081：从冻结 snapshot 派生（与 out_path 的 _h 段、runtime 各层
        # 同一对象）——生成窗口内文件替换不再产生 receipt 混装；yarn_
        # receipt 的 081 闭包校验（tm hash[:10] == basename _h 段）由此
        # 结构性地成立。
        treatment_manifest_json = (
            get_treatment_manifest_json(args, snapshot)
            if args.method == "tli" else None)
        yarn_receipt = build_yarn_receipt(
            yarn_enabled=use_yarn,
            effective_factor=factor,
            yarn_factor_cli=args.yarn_factor,
            rope_scaling=rope_scaling,
            context_length=args.context_length,
            task=args.task,
            model_path=os.path.abspath(args.model_path),
            model_config_sha256=(
                _file_sha256(model_cfg_path)
                if os.path.isfile(model_cfg_path) else None),
            native_mpe=native_mpe,
            generation_params={
                "max_gen": MAX_GEN, "max_num": args.max_num, "seed": SEED,
                "method": args.method, "pred_postfix": args.pred_postfix,
                "t": args.t, "graph_policy": graph_policy(),
                "operator_mode": os.environ.get("TLI_OPERATOR_MODE", "off"),
            },
            producer_script_path="benchmark/RULER/pred_ruler.py",
            producer_script_sha256=_file_sha256(os.path.abspath(__file__)),
            run_id=attempt_id,
            prediction_basename=os.path.basename(out_path),
            prediction_sha256=pred_sha,
            prediction_lines=pred_lines,
            treatment_manifest_json=treatment_manifest_json)
        yarn_receipt.update(receipt_identity)
        tmp_rcp = stage_yarn_receipt(out_path, yarn_receipt, attempt_id)
        commit_yarn_generation(tmp_pred, tmp_rcp, out_path)
        print(f"[yarn-receipt] effective config + 同代绑定 -> {tmp_rcp}"
              f"（generation 目录内）")
        print(f"[pred] 提交：指针 {generation_pointer_path(out_path)} -> "
              f"{os.path.basename(os.path.dirname(tmp_pred))}")
        # 「saved -> {out_path}」行保留（调度端 provenance 对账依赖该
        # 口径）；实体 = 指针所指 generation 内的预测文件
        print(f"saved -> {out_path} (via generation pointer)")
    finally:
        # 崩溃/异常路径：未指向的 gen 目录残留（目录名不以 .jsonl 结尾，
        # 不污染 {task}-*.jsonl glob），指针保持上一完整代；锁由本处
        # 显式释放或进程退出时内核自动释放（flock 不持久，无死锁）。
        release_output_lock(lock_fd)


def _gpu_memory_guard(required_free_mb=2000):
    """启动前检查 GPU 空闲显存，不足则 fail-fast 不加载模型。
    防止在显存已被占用的机器上启动后 OOM 崩溃。"""
    if not torch.cuda.is_available():
        return
    free_mem, total_mem = torch.cuda.mem_get_info()
    free_mb = free_mem / 1024 / 1024
    if free_mb < required_free_mb:
        print(f"[OOM-GUARD] GPU 空闲显存 {free_mb:.0f} MiB < "
              f"阈值 {required_free_mb} MiB，放弃启动。",
              file=sys.stderr)
        os._exit(2)
    print(f"[GPU] 空闲显存 {free_mb:.0f} / {total_mem/1024/1024:.0f} MiB")


if __name__ == "__main__":
    _gpu_memory_guard()
    main()
    os._exit(0)
