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

【方法范围（Phase 1）】只支持 `none` / `fasa_vendor` / `tli`（已 batch
化）。quest_vendor / jssa_vendor / asynctls_vendor 在 batch_size>1 时
启动即拒绝（Phase 2 逐个 batch 化后再开），拒绝而非静默产错
（GATE-FAIL 精神）。batch_size=1 仍走 batch 路径（供 gate_bs1_equiv
门禁：JSONL 与旧 pred_ruler.py 逐字节一致）。

【同步义务】模型加载 + dispatch 分支内联复制自 pred_ruler.py（同步义务
标注在各分支注释；tli 挂载内联自 pred_ruler.py L184-185 + L444，
qwen3_attn_patch 的 batch decode 见 sparse_attn/patches/qwen3_attn_patch.py
`_tl_decode_batched`）。minference/sparq/moba/hisa/clusterkv 不在此入口。
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
import time
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
from sparse_attn.arguments import add_sparse_attn_args  # noqa: E402
from sparse_attn.baseline_runtime import (  # noqa: E402
    baseline_request, baseline_preflight, BatchCtx,
    batch_decode_scope,
)
from sparse_attn.info import (  # noqa: E402
    get_method_name_with_info, get_treatment_manifest_json,
    resolve_treatment_snapshot,
    stage_fasa_agreement)
# 【阶段 1】TLI 主方法挂载（同步义务：pred_ruler.py L81/L184-185/L444）。
from sparse_attn.patches import register_patch  # noqa: E402
# 【阶段 1】budget 采集对齐 pred_ruler.py L549-551（同步义务：read 端
# get_select_tokens + write 端 "budget" 字段逐字节一致）。
from sparse_attn.metrics import get_metrics  # noqa: E402

RULER_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt",
]
MAX_GEN = 64  # 官方 RULER 口径（全部任务统一，同步 pred_ruler.py）
SEED = 42
QWEN3_NATIVE_MPE = 40960
YARN_FACTOR_AUTO = {65536: 2.0, 131072: 4.0}

# Phase 1 白名单：已 batch 化的方法；其余方法 batch 入口拒绝（Phase 2 逐个开）
# 【阶段 1】tli 加入（TLI 主方法 batch decode：qwen3_attn_patch 的
# _tl_decode_batched 逐行单序列核心 + BatchCtx 驱动）。
_BATCH_METHODS = {"none", "fasa_vendor", "tli"}


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
                        help="decode batch（支持 none/fasa_vendor/tli；"
                             "compact TLI 支持按请求混合 ring/linear cache）")
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

    【阶段 2a】left_pad_kv=True（TLI TLIStaticCache）：有效 KV 在前
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


def _decode_batch_graph(model, ctx, past, gen, lens, eos_ids, pad_id,
                        neg_inf, dtype, device, B, S_cap):
    from sparse_attn.ops.quality_graph import static_decode
    return static_decode(model, past, gen, lens, eos_ids, MAX_GEN, ctx)


def _decode_hybrid_batch_chunk(model, tokenizer, input_ids, position_ids, lens, eos_ids):
    """Compact TLI: per-request prefill/gates, batched model decode, optional graph."""
    raise RuntimeError(
        "TLI hybrid batch cache path has been removed; use B1 PagedRegionCache")

    device = next(model.parameters()).device
    input_ids, position_ids = input_ids.to(device), position_ids.to(device)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    batch, width = input_ids.shape
    builder = HybridBatchCacheBuilder(model, max(lens)+MAX_GEN, batch)
    generated = []
    prefill_mode = os.environ.get('TLI_HYBRID_PREFILL', 'auto')
    if prefill_mode not in ('auto', 'direct', 'conversion'):
        raise ValueError('TLI_HYBRID_PREFILL must be auto, direct or conversion')
    # Do not expose the outer B-row BatchCtx to a B=1 unpadded prefill.
    with batch_decode_scope(None):
        for b, length in enumerate(lens):
            row_cache = builder.prepare_row(length) if prefill_mode != 'conversion' else None
            output = model(input_ids=input_ids[b:b+1, width-length:],
                           position_ids=position_ids[b:b+1, width-length:],
                           past_key_values=row_cache, use_cache=True)
            generated.append([output.logits[0, -1].argmax().item()])
            if row_cache is None:
                builder.add(output.past_key_values, length)
            else:
                builder.finalize_row(row_cache, length)
            del output  # Release prefill logits before the next request.
    runner = HybridBatchRunner(model, builder.finish(), pad_id)
    # Compact capture recovery has a distinct cache lifecycle; keep auto on
    # its validated staged path until that recovery is independently checked.
    use_graph = graph_policy() == 'on'
    record = execution(model,'staged_hybrid',
                       'hybrid_auto_capture_not_validated' if graph_policy()=='auto' else None,
                       batch_size=batch)
    alive, lengths = [True]*batch, list(lens)
    for step in range(MAX_GEN-1):
        if not any(alive):
            break
        tokens = [row[-1] if active else pad_id for row, active in zip(generated, alive)]
        lengths = [length+int(active) for length, active in zip(lengths, alive)]
        runner.inject(tokens, lengths, alive)
        if use_graph and step == 0:
            # This path supports short contexts too; capture failures are visible.
            runner.capture()
            record.update(actual="graph_hybrid",captures=1)
        if runner.graph is not None:
            record["replay_steps"] += 1
        predicted = runner.run().logits[:, -1].argmax(-1).tolist()
        for b, token in enumerate(predicted):
            if alive[b]:
                generated[b].append(token)
                if token in eos_ids and len(generated[b]) > 1:
                    alive[b] = False
    print(f'[hybrid-batch] B={batch} graph={runner.graph is not None} '
          f'kv_storage_bytes={runner.kv_storage_bytes()}', flush=True)
    return generated


def _decode_batch_chunk(model, tokenizer, rows, input_ids, position_ids,
                        mask2d, lens, eos_ids, keep_logits,
                        fixed_cache=False):
    """一个 left-pad chunk 的 prefill + 多步 decode，返回逐行 (pred, gen)。

    语义对齐 pred_ruler.py 生成循环（首位 EOS 不停 ≈ min_length、
    `len(gen)>1` 才断）；dead 行喂 pad_id filler 吃计算、由 4D mask 排除。

    fixed_cache=True（TLI 阶段 2a）：prefill 走 DynamicCache（现状零改动），
    prefill 后 fill_static_from_dynamic 逐层搬移到 TLIStaticCache（固定容量
    S_cap、有效在前 left-pad 布局）→ decode 每步 KV 形状恒 [B,Hkv,S_cap,D]
    （CUDA Graph 前置）。decode 4D mask 列数恒 S_cap、可见列取头部
    [0, seq_lens[b])（left_pad_kv=True）。
    """
    from sparse_attn.ops.quality_graph import unsupported
    compact = any(getattr(getattr(m, 'indexer', None), 'far_skip_backend', '') == 'compact'
                  for m in model.modules())
    if fixed_cache or compact:
        raise RuntimeError(
            "TLI fixed/compact batch cache path has been removed; "
            "use B1 PagedRegionCache")
    if compact:
        if not fixed_cache:
            raise ValueError('compact batched decode requires fixed_cache=True')
        return _decode_hybrid_batch_chunk(model, tokenizer, input_ids, position_ids, lens, eos_ids)
    if not fixed_cache:
        unsupported(model, "method_has_no_fixed_cache_graph")
    B = len(rows)
    pad_id = getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id
    input_ids = input_ids.to("cuda")
    position_ids = position_ids.to("cuda")
    mask2d = None if fixed_cache else mask2d.to("cuda")
    device = input_ids.device
    neg_inf = torch.finfo(next(model.parameters()).dtype).min
    dtype = next(model.parameters()).dtype
    S_max = input_ids.shape[1]

    ctx = BatchCtx(
        batch=B, seq_lens=list(lens), gen_lens=[0] * B,
        alive=[True] * B, pad_id=pad_id)
    with batch_decode_scope(ctx):

        # prefill row-by-row（避免 B>1 触发 HF SDPA 构造
        # [B*n_heads, S, S] bool mask OOM：B=8/32K ≈ 256 GB > 140 GB H20）：
        # 每行 B=1 走普通 SDPA（[1*n_heads, S_row, S_row] ≈ 32 GB 单行可容），
        # 先保留各行实际 KV，全部 prefill 后逐层搬入 static 并释放源层，
        # 避免后续 prefill 与完整 static batch 同时驻留。
        # 生成 token = 每行末 position logits argmax（与旧 batch prefill 语义一致：
        # prefill 后立即预测第一 decode token）。
        # 非 TLI（fixed_cache=False）：同样逐行 B=1 prefill，
        # 后接 staged row-by-row decode（batch>1 无 graph 提速，逐行回退零额外开销）。
        pred_list = []
        if fixed_cache:
            S_cap = S_max + MAX_GEN
            static_cache = None
            prefilled = []
            for b in range(B):
                base = S_max - lens[b]
                row_input = input_ids[b:b+1, base:]
                row_pos = position_ids[b:b+1, base:]
                out = model(input_ids=row_input, position_ids=row_pos,
                            use_cache=True)
                pred_list.append(out.logits[0, -1, :].argmax(dim=-1))
                prefilled.append(out.past_key_values)
                del out
            consume_prefill_rows(static_cache, prefilled, lens)
            prefilled.clear()
            past = static_cache
        else:
            # 非 TLI：逐行 B=1 prefill → 独立 DynamicCache → row-by-row staged
            # decode；每行 forward 间 del 释放 DynamicCache 控峰值。
            row_caches = []
            for b in range(B):
                base = S_max - lens[b]
                row_input = input_ids[b:b+1, base:]
                row_pos = position_ids[b:b+1, base:]
                out = model(input_ids=row_input, position_ids=row_pos,
                            use_cache=True)
                pred_list.append(out.logits[0, -1, :].argmax(dim=-1))
                row_caches.append(out.past_key_values)
                del out
            # Multi-row DynamicCache 组装为 left-pad 全宽（有效占尾），
            # row-by-row staged decode 使用此 per-row cache；
            # 下面 staged 循环改为逐行独立 cache、mask2d 逐行。
            past = row_caches
        pred = torch.tensor(pred_list, dtype=torch.long, device=device)
        gen = [[p] for p in pred.tolist()]            # 每行一个 gen 列表

        # 【阶段 2c-5】CUDA Graph decode（env TLI_CUDA_GRAPH=1；仅 TLI
        # fixed_cache、S_cap≥16384 且生产主配置——不满足时 helper 返回
        # None → 落下方 staged 逐行路径，行为零扰动）。
        if fixed_cache:
            gen_g = _decode_batch_graph(
                model, ctx, past, gen, lens, eos_ids, pad_id, neg_inf,
                dtype, device, B, S_cap)
            if gen_g is not None:
                return gen_g

        _t_stage, _n_stage = time.time(), 0
        for step in range(MAX_GEN - 1):
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
            # mask 列数：fixed_cache = 恒 S_cap（TLIStaticCache 固定容量、有效在
            # 前 → left_pad_kv 可见列取头部 [0, seq_lens[b])）；否则 = S_max +
            # step + 1（变长 DynamicCache 有效占尾 → 可见列取尾部）。行 b 可见
            # 长 = 更新后的 seq_lens[b]（= lens[b] + step + 1，含本步新 token）。
            if fixed_cache:
                kv_len = S_cap
                m4 = _decode_4d_mask(kv_len, ctx.seq_lens, neg_inf, dtype,
                                     device, left_pad_kv=True)
                out = model(input_ids=nxt_ids, position_ids=pos,
                            attention_mask=m4, past_key_values=past,
                            use_cache=True)
                pred = out.logits[:, -1, :].argmax(dim=-1)  # [B]
            else:
                # 非 TLI：逐行独立 B=1 decode（DynamicCache 各行长不同，
                # 无法 batch → 逐行 forward + per-row 4D mask）。
                pred = torch.zeros(B, dtype=torch.long, device=device)
                for b in range(B):
                    if not ctx.alive[b]:
                        continue
                    base = S_max - lens[b]
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
            _n_stage += 1
            for b in range(B):
                if not ctx.alive[b]:
                    continue
                gen[b].append(pred[b].item())
                if pred[b].item() in eos_ids and len(gen[b]) > 1:
                    ctx.alive[b] = False
        if fixed_cache and _n_stage:
            _dt = time.time() - _t_stage
            print(f"[2c-wall] staged decode: steps={_n_stage} wall={_dt:.3f}s "
                  f"({1000 * _dt / _n_stage:.2f} ms/step, B={B})", flush=True)
        return gen


def main():
    from sparse_attn.ops.eager_decoding import eager_decoding_request_guard
    from sparse_attn.ops.quality_graph import graph_policy
    args = parse_args()
    if args.method == "tli":
        raise RuntimeError(
            "TLI batch cache path has been removed; use benchmark/RULER/pred_ruler.py "
            "with B1 PagedRegionCache")
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
        if (args.method == "tli" and getattr(args, "tli_dyn_dgate", False)
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
        rope_policy_record = configure_rope(config, args.context_length, MAX_GEN,
                                           yarn=use_yarn, factor=args.yarn_factor)
        # This constructs the installed framework's rotary module on CPU and
        # checks actual frequencies, scaling and rotation before loading weights.
        rope_preflight = verify_runtime_rope(config)
    snapshot = resolve_treatment_snapshot(args) if args.method == "tli" else None
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
    elif args.method == "tli":
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
    eos_ids = [tokenizer.eos_token_id]
    nl = tokenizer.encode("\n", add_special_tokens=False)
    if nl:
        eos_ids.append(nl[-1])

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
            if rope_policy_record is not None and max(lens) + MAX_GEN > config.max_position_embeddings:
                raise ValueError("actual batch input plus output exceeds verified runtime position capacity")
            # 【阶段 1】budget 采集对齐 pred_ruler.py（L549-551 同步义务）：
            # chunk 前 clear 全局 metrics、baseline_request 解包 receipt
            # （tli 无 _baseline_identity → receipt 空 dict falsy → budget 取
            # metrics.get_select_tokens() 实际值）。batch=1 时 chunk=单请求，
            # 与旧 pred_ruler.py 单序列入口逐字节一致（gate_bs1_equiv 依据）；
            # batch>1 时取该 chunk 全部行×步的 select_tokens 均值（聚合口径）。
            metrics = get_metrics()
            metrics.clear()
            with baseline_request(model) as baseline_receipt, eager_decoding_request_guard():
                with torch.no_grad():
                    gen = _decode_batch_chunk(
                        model, tokenizer, chunk, input_ids, position_ids,
                        mask2d, lens, eos_ids, keep_logits,
                        fixed_cache=(args.method == "tli"))
            budget = None if (baseline_receipt or getattr(model, "_quality_graph_execution", {}).get("actual", "").startswith("graph")) else metrics.get_select_tokens()
            for (row, g) in zip(chunk, gen):
                pred = tokenizer.decode(g, skip_special_tokens=True)
                fout.write(json.dumps({
                    **({"baseline_receipt": baseline_receipt}
                       if baseline_receipt else {}),
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

        pred_sha = _file_sha256(tmp_pred)
        pred_lines = sum(1 for _ in open(tmp_pred, "rb"))
        model_cfg_path = os.path.join(args.model_path, "config.json")
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
                "t": args.t, "batch_size": args.batch_size,
                "graph_policy": graph_policy(),
                "operator_mode": os.environ.get("TLI_OPERATOR_MODE", "off"),
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
