"""# RULER batch>1 精度扫描入口（计划 B Phase 1，left-pad 协议）。

与 benchmark/RULER/pred_ruler.py 同数据/同打分口径，差别在 decode 批量：
- prefill：多行 left-pad 打包 [B, S_max] + 显式 position_ids（`j-base_b`
  clamp_min 0，pad 前缀位置恒 0）+ 2D attention_mask（尾部真实长置 1）；
- decode：模块级 BatchCtx（sparse_attn/baseline_runtime.py Phase 0）驱动，
  每步输入 [B,1]，所有行等长 append（transformers 4.56 DynamicLayer.update
  无条件 torch.cat，cache_position 无插入作用），pad 列由显式 4D additive
  mask（尾部 seq_lens[b] 列 0，其余 -inf）排除；停止条件逐行复刻旧循环
  （首位 EOS 不停 ≈ min_length、`len(gen)>1` 才断）。
- 死行（已触发停止）继续吃计算：喂 pad_id filler，其 KV 列被 mask 排除，
  不污染同 batch 其它行（每行因果独立）。

【方法范围】只支持 `none` / `fasa_vendor`；NFA batch/cache 旧路径已移除，
本入口启动即拒绝，生产 NFA 使用 `pred_ruler.py` 的 GraphRunner 质量路径，
或 `benchmark/efficiency/benchmark_nfa_end_to_end.py` 的多 BS 性能路径。
quest_vendor / jssa_vendor / asynctls_vendor 在 batch_size>1 时同样启动即
拒绝，避免把逐行回退或旧 cache 代码夸大为已验证的批量/CUDA Graph 能力。

【同步义务】模型加载 + dispatch 分支内联复制自 pred_ruler.py；本文件只维护
none/fasa_vendor 基线，不再包含 NFA cache 或 decode 实现。
minference/sparq/moba/hisa/clusterkv 不在此入口。
producer_script_sha256 用本文件自身 sha（yarn_receipt v2 同代绑定），
pred_postfix 强制注入 `_bs{B}` 后缀隔离 producer（batch 入口产物不与
旧入口同名互覆）。

【用法】对齐 run_ruler_*.sh，追加 --batch-size N：
  python -u -m benchmark.RULER.pred_ruler_batch \
    --model Qwen3-8B --model_path $MODEL \
    --task niah_single_1 --context_length 32768 --method fasa_vendor --t $TS \
    --data-root .../data/RULER --output-dir exp/results_ruler/Qwen3-8B \
    --fasa_agreement_dir <staged_root> --fasa_budget 1024 \
    --pred_postfix _bs4 --max-num 100 --batch-size 4
"""
import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime

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
from benchmark.RULER.pred_ruler import (  # noqa: E402
    RULER_TASKS, _ruler_eos_ids, _ruler_max_new_tokens)
from sparse_attn.scheduler.scheduler import greedy_next_tokens  # noqa: E402
from sparse_attn.arguments import add_sparse_attn_args  # noqa: E402
from sparse_attn.scheduler.batch_builder import BatchCtx, batch_decode_scope
from sparse_attn.baseline_runtime import baseline_request, baseline_preflight
from sparse_attn.info import (  # noqa: E402
    get_method_name_with_info, get_treatment_manifest_json,
    resolve_treatment_snapshot,
    stage_fasa_agreement)
# 【阶段 1】NFA 主方法挂载（同步义务：pred_ruler.py L81/L184-185/L444）。
from sparse_attn.patches import register_patch  # noqa: E402
# 【阶段 1】budget 采集对齐 pred_ruler.py L549-551（同步义务：read 端
# get_select_tokens + write 端 "budget" 字段逐字节一致）。
from sparse_attn.metrics import get_metrics  # noqa: E402

SEED = 42
QWEN3_NATIVE_MPE = 40960
YARN_FACTOR_AUTO = {65536: 2.0, 131072: 4.0}

# Phase 1 白名单：已 batch 化的方法；其余方法 batch 入口拒绝（Phase 2 逐个开）
# 【阶段 1】nfa 加入（NFA 主方法 batch decode：qwen3_attn_patch 的
# _tl_decode_batched 逐行单序列核心 + BatchCtx 驱动）。
_BATCH_METHODS = {"none", "fasa_vendor"}


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
    parser.add_argument("--t", type=str, default="ruler batch run")
    parser.add_argument("--pred_postfix", type=str, default="")
    parser.add_argument("--max-num", type=int, default=500,
                        help="每任务样本数（0=全量）")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="decode batch（支持 none/fasa_vendor/nfa；"
                             "compact NFA 支持按请求混合 ring/linear cache）")
    parser.add_argument("--yarn", action="store_true",
                        help="启用 YaRN 位置扩展（65536/131072 档必开）")
    parser.add_argument("--yarn_factor", type=float, default=None)
    add_sparse_attn_args(parser)
    return parser.parse_args()


def seed_everything(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _pack_rows(tokenizer, rows, pad_id):
    """left-pad 打包：input_ids/position_ids/2D mask 均为 [B, S_max]。

    行 b 的 pad 前缀长 base_b = S_max - lens[b]；position_ids[b, j] =
    j - base_b（clamp 0，pad 前缀位置恒 0，被 mask 排除）。返回
    (input_ids, position_ids, mask2d, lens)（全为 torch.long / torch 设备）。
    """
    encoded = [tokenizer(r["input"], truncation=False,
                         return_tensors="pt").input_ids[0]
               for r in rows]
    lens = [e.shape[-1] for e in encoded]
    S_max = max(lens)
    B = len(rows)
    input_ids = torch.full((B, S_max), pad_id, dtype=torch.long)
    position_ids = torch.zeros(B, S_max, dtype=torch.long)
    mask = torch.zeros(B, S_max, dtype=torch.long)
    for b, (e, L) in enumerate(zip(encoded, lens)):
        base = S_max - L
        input_ids[b, base:] = e
        position_ids[b, base:] = torch.arange(L)
        mask[b, base:] = 1
    return input_ids, position_ids, mask, lens


def _decode_4d_mask(past_len, seq_lens, neg_inf, dtype, device,
                    left_pad_kv=False):
    """构造本步 decode 的 4D additive mask [B, 1, 1, kv_len]。

    kv_len = past_len（本步 forward 时 KV 总列数；调用方需按「cache 更新
    后」口径传）。行 b 的 seq_lens[b] 个有效 KV 列可见（0）、其余 pad 列
    屏蔽（-inf）。dtype 与 hidden_states 一致（attn_weights + mask 同
    dtype 加法）。

    【阶段 2a】left_pad_kv=True（NFA NFAStaticCache）：有效 KV 在前
    [0, seq_lens[b])、pad 在尾 → 可见列取头部 :L；False（默认，fasa/none
    DynamicCache 有效占尾）→ 可见列取尾部 kv_len-L:。dense/skip 层消费
    本 mask 与 KV 布局必须一致，否则会看零 KV（head 布局错位）。
    """
    B = len(seq_lens)
    kv_len = past_len
    mask = torch.full((B, 1, 1, kv_len), neg_inf, dtype=dtype, device=device)
    for b, L in enumerate(seq_lens):
        if left_pad_kv:
            mask[b, 0, 0, :L] = 0
        else:
            mask[b, 0, 0, kv_len - L:] = 0
    return mask


def _decode_batch_chunk(model, tokenizer, rows, input_ids, position_ids,
                        mask2d, lens, eos_ids, keep_logits, max_new_tokens):
    """none/fasa_vendor 的逐行 DynamicCache prefill/decode 基线路径。"""
    B = len(rows)
    pad_id = getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id
    input_ids = input_ids.to("cuda")
    position_ids = position_ids.to("cuda")
    mask2d = mask2d.to("cuda")
    device = input_ids.device
    neg_inf = torch.finfo(next(model.parameters()).dtype).min
    dtype = next(model.parameters()).dtype
    S_max = input_ids.shape[1]

    ctx = BatchCtx(
        batch=B, seq_lens=list(lens), gen_lens=[0] * B,
        alive=[True] * B, pad_id=pad_id)
    with batch_decode_scope(ctx):

        # 逐行 prefill 保留各自 DynamicCache，避免批量 SDPA 的显存峰值。
        pred_list = []
        row_caches = []
        for b in range(B):
            base = S_max - lens[b]
            row_input = input_ids[b:b+1, base:]
            row_pos = position_ids[b:b+1, base:]
            out = model(input_ids=row_input, position_ids=row_pos,
                        use_cache=True)
            pred_list.append(greedy_next_tokens(
                out.logits[:, -1:, :], eos_ids, generated_counts=0,
                min_new_tokens=1)[0])
            row_caches.append(out.past_key_values)
            del out
        past = row_caches
        pred = torch.tensor(pred_list, dtype=torch.long, device=device)
        gen = [[p] for p in pred.tolist()]            # 每行一个 gen 列表

        for step in range(max_new_tokens - 1):
            if not any(ctx.alive):
                break
            # 本步喂入 token：alive 行 = 当前最后产生 token，dead 行 = pad_id
            nxt = [gen[b][-1] if ctx.alive[b] else pad_id for b in range(B)]
            nxt_ids = torch.tensor(nxt, dtype=torch.long, device=device).view(B, 1)
            # 位置：alive 行 = lens[b] + step（当前绝对位置）；dead 行同式递增
            # （被 mask 排除，无影响）。
            pos = torch.tensor([lens[b] + step for b in range(B)],
                               dtype=torch.long, device=device).view(B, 1)
            # 【2026-10-20 修复】seq_lens/gen_lens 在 forward 前 +1：本步新 token
            # 计入真实上下文长（与 KV append 同步）——否则 4D mask 的「尾部
            # 真实长列」少 1，最长行（无 pad）会把第 0 列（真实首 token）误
            # 屏蔽 → decode 首步后全行输出跑偏。修复前 batch=2 dense 冒烟
            # row0 首 token 对、后续全错即此因。
            for b in range(B):
                if ctx.alive[b]:
                    ctx.seq_lens[b] += 1
                    ctx.gen_lens[b] += 1
            # 各行 DynamicCache 长度不同，decode 也保持逐行独立。
            pred = torch.zeros(B, dtype=torch.long, device=device)
            for b in range(B):
                if not ctx.alive[b]:
                    continue
                kv_len_b = lens[b] + step + 1
                m4b = _decode_4d_mask(
                    kv_len_b, [ctx.seq_lens[b]], neg_inf, dtype, device,
                    left_pad_kv=False)
                out_b = model(input_ids=nxt_ids[b:b+1],
                              position_ids=pos[b:b+1],
                              attention_mask=m4b,
                              past_key_values=past[b],
                              use_cache=True)
                past[b] = out_b.past_key_values
                pred[b] = out_b.logits[0, -1, :].argmax(dim=-1)
                del out_b
            for b in range(B):
                if not ctx.alive[b]:
                    continue
                gen[b].append(pred[b].item())
                if pred[b].item() in eos_ids and len(gen[b]) > 1:
                    ctx.alive[b] = False
        return gen


def main():
    from sparse_attn.ops.eager_decoding import eager_decoding_request_guard
    args = parse_args()
    max_new_tokens = _ruler_max_new_tokens(args.task)
    if args.method == "nfa":
        raise RuntimeError(
            "NFA batch cache path has been removed; use pred_ruler.py for B1 quality "
            "or benchmark/efficiency/benchmark_nfa_end_to_end.py for multi-BS performance")
    if args.batch_size < 1:
        raise SystemExit("--batch-size 必须 >= 1")
    if args.batch_size > 1 and args.method not in _BATCH_METHODS:
        raise SystemExit(
            f"[gate-fail] batch 入口 Phase 1 只支持 {sorted(_BATCH_METHODS)}；"
            f"method={args.method!r} 需等 Phase 2 batch 化后再开（拒绝静默"
            f"产错）。batch_size=1 可走旧 pred_ruler.py 或本入口。")
    # producer 隔离：强制 pred_postfix 带 _bs{B}（batch 产物不覆盖旧入口）
    if f"_bs{args.batch_size}" not in args.pred_postfix:
        args.pred_postfix = f"{args.pred_postfix}_bs{args.batch_size}"
        print(f"[batch] pred_postfix 强制注入 _bs{args.batch_size} -> "
              f"{args.pred_postfix!r}")

    baseline_preflight(args)
    seed_everything()

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    is_glm = getattr(config, "model_type", None) == "glm4_moe_lite"
    native_mpe = config.max_position_embeddings
    use_yarn = args.yarn
    if is_glm:
        if (args.method == "nfa" and getattr(args, "nfa_dyn_dgate", False)
                and args.batch_size > 1):
            raise SystemExit(
                "[glm] dynamic far-skip 当前仅支持单请求入口；"
                "batch_size>1 尚未实现逐行 gate/有效长度标定")
        if native_mpe != 202752:
            raise SystemExit(f"[glm] 原生 MPE 应为 202752，实际 {native_mpe!r}")
        if args.yarn or args.yarn_factor is not None:
            raise SystemExit("[glm] GLM-4.7-Flash 原生覆盖 32K/64K/128K，禁止 --yarn/--yarn_factor")
        use_yarn = False
    if not is_glm and args.context_length > QWEN3_NATIVE_MPE and not use_yarn:
        raise SystemExit(
            f"[yarn] context_length={args.context_length} 超出 Qwen3-8B 原生 "
            f"{QWEN3_NATIVE_MPE}，必须加 --yarn")
    factor, rope_scaling = resolve_yarn_config(
        use_yarn, args.yarn_factor, args.context_length,
        YARN_FACTOR_AUTO, native_mpe)
    if use_yarn:
        print(f"[yarn] enabled: factor={factor}")

    data_file = os.path.join(args.data_root, str(args.context_length),
                             f"{args.task}.jsonl")
    rows = [json.loads(l) for l in open(data_file)]
    if args.max_num and len(rows) > args.max_num:
        rows = rows[:args.max_num]
    print(f"[ruler-batch] task={args.task} L={args.context_length} "
          f"n={len(rows)} method={args.method} batch={args.batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True)
    if use_yarn:
        config.rope_scaling = rope_scaling
    from benchmark.rope_policy import configure_rope, verify_runtime_rope
    rope_policy_record = None
    if config.model_type in ('qwen3', 'glm4_moe_lite'):
        rope_policy_record = configure_rope(config, args.context_length, max_new_tokens,
                                           yarn=use_yarn, factor=args.yarn_factor)
        # This constructs the installed framework's rotary module on CPU and
        # checks actual frequencies, scaling and rotation before loading weights.
        rope_preflight = verify_runtime_rope(config)
    snapshot = resolve_treatment_snapshot(args) if args.method == "nfa" else None
    # 【同步义务】加载路径与 pred_ruler.py 一致（device_map="auto"，
    # 调度须设 CUDA_VISIBLE_DEVICES）。
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto").eval()

    runtime_rope = verify_runtime_rope(model) if rope_policy_record is not None else None
    # 【同步义务】以下 dispatch 分支内联自 pred_ruler.py 对应分支；Phase 1
    # 仅保留 none + fasa_vendor（其余方法由 parse 阶段白名单拒绝）。
    if args.method == "fasa_vendor":
        vendor_dir = os.path.join(
            REPO, "exp", "vendor_baselines", "fasa_vendor")
        if vendor_dir not in sys.path:
            sys.path.insert(0, vendor_dir)
        from qwen3_fasa_patch import enable_fasa  # noqa: E402
        fasa_dataset = getattr(args, "fasa_dataset", "qasper")
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
    elif args.method == "nfa":
        # 【同步义务】内联自 pred_ruler.py（L184-185 snapshot 冻结 +
        # L444 register_patch 挂载）；081 快照契约：模型加载前冻结一次，
        # D′ 掩码/投影基每个文件只 open 一次。
        n_patched = register_patch(model, args, snapshot)
    else:
        n_patched = 0  # method=none：dense，不挂任何 patch
    if args.method != "none" and n_patched == 0:
        raise RuntimeError(
            f"method={args.method!r} 但成功挂载 0 个 attention 模块，模型"
            f"实际将以 dense 运行——拒绝产出被误标为稀疏方法的结果。")

    keep_logits = args.context_length > 32768
    eos_ids = _ruler_eos_ids(model, tokenizer)

    out_dir = os.path.join(args.output_dir, f"L{args.context_length}",
                           f"pred{args.pred_postfix}")
    os.makedirs(out_dir, exist_ok=True)
    method_name = get_method_name_with_info(args, snapshot)
    out_path = os.path.join(
        out_dir, f"{args.task}-{method_name}-{args.t}.jsonl")

    attempt_id = (datetime.now().strftime("%Y%m%d%H%M%S") +
                  f"-{os.getpid()}-{random.randint(1000, 9999)}")
    lock_fd = acquire_output_lock(out_path)
    try:
        tmp_pred = stage_yarn_generation(out_path, attempt_id)
        fout = open(tmp_pred, "w", encoding="utf-8")
    except OSError as e:
        release_output_lock(lock_fd)
        raise SystemExit(f"[pred] 打开临时 generation 失败（{e}）")

    pad_id = getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id
    try:
        for start in tqdm(range(0, len(rows), args.batch_size)):
            chunk = rows[start:start + args.batch_size]
            torch.cuda.empty_cache()
            input_ids, position_ids, mask2d, lens = _pack_rows(
                tokenizer, chunk, pad_id)
            if rope_policy_record is not None and max(lens) + max_new_tokens > config.max_position_embeddings:
                raise ValueError("actual batch input plus output exceeds verified runtime position capacity")
            # 【阶段 1】budget 采集对齐 pred_ruler.py（L549-551 同步义务）：
            # chunk 前 clear 全局 metrics、baseline_request 解包 receipt
            # （nfa 无 _baseline_identity → receipt 空 dict falsy → budget 取
            # metrics.get_select_tokens() 实际值）。batch=1 时 chunk=单请求，
            # 与旧 pred_ruler.py 单序列入口逐字节一致（gate_bs1_equiv 依据）；
            # batch>1 时取该 chunk 全部行×步的 select_tokens 均值（聚合口径）。
            metrics = get_metrics()
            metrics.clear()
            with baseline_request(model) as baseline_receipt, eager_decoding_request_guard():
                with torch.no_grad():
                    gen = _decode_batch_chunk(
                        model, tokenizer, chunk, input_ids, position_ids,
                        mask2d, lens, eos_ids, keep_logits, max_new_tokens)
            budget = None if baseline_receipt else metrics.get_select_tokens()
            for (row, g) in zip(chunk, gen):
                pred = tokenizer.decode(g, skip_special_tokens=True)
                fout.write(json.dumps({
                    **({"baseline_receipt": baseline_receipt}
                       if baseline_receipt else {}),
                    "pred": pred, "answers": row["outputs"],
                    "generated_token_ids": g,
                    "length": row["length"], "budget": budget,
                    "execution": {"requested": "eager", "actual": "eager"},
                    "_id": f"{args.task}:{row['index']}",
                    "_answers_sha": hashlib.sha256(json.dumps(
                        row["outputs"], ensure_ascii=False, sort_keys=True)
                        .encode("utf-8")).hexdigest()[:16],
                }, ensure_ascii=False) + "\n")
                fout.flush()
        fout.close()

        pred_sha = _file_sha256(tmp_pred)
        pred_lines = sum(1 for _ in open(tmp_pred, "rb"))
        model_cfg_path = os.path.join(args.model_path, "config.json")
        treatment_manifest_json = (
            get_treatment_manifest_json(args, snapshot)
            if args.method == "nfa" else None)
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
                "max_gen": max_new_tokens, "min_new_tokens": 1,
                "eos_token_ids": list(eos_ids),
                "max_num": args.max_num, "seed": SEED,
                "method": args.method, "pred_postfix": args.pred_postfix,
                "t": args.t, "batch_size": args.batch_size,
                "graph_policy": "off",
                "operator_mode": os.environ.get("NFA_OPERATOR_MODE", "off"),
            },
            producer_script_path="benchmark/RULER/pred_ruler_batch.py",
            producer_script_sha256=_file_sha256(os.path.abspath(__file__)),
            run_id=attempt_id,
            prediction_basename=os.path.basename(out_path),
            prediction_sha256=pred_sha,
            prediction_lines=pred_lines,
            treatment_manifest_json=treatment_manifest_json)
        if rope_policy_record is not None:
            yarn_receipt["rope_policy"] = rope_policy_record
            yarn_receipt["runtime_rope_verification"] = runtime_rope
        tmp_rcp = stage_yarn_receipt(out_path, yarn_receipt, attempt_id)
        commit_yarn_generation(tmp_pred, tmp_rcp, out_path)
        print(f"[pred-batch] 提交：指针 {generation_pointer_path(out_path)} -> "
              f"{os.path.basename(os.path.dirname(tmp_pred))}")
        print(f"saved -> {out_path} (via generation pointer)")
    finally:
        release_output_lock(lock_fd)


if __name__ == "__main__":
    main()
