"""64K MK2 sample13 的 L1/L2 预算隔离诊断；不作为正式质量结果。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch

WT = Path(os.environ["NFA_QUALITY_WT"])
sys.path.insert(0, str(WT))
os.chdir(WT)

from benchmark.efficiency import benchmark_ruler_quality as quality

OUT = Path(os.environ["NFA_SELECTION_DIAG_OUT"])
K1 = int(os.environ.get("NFA_DIAG_K1", "128"))
K2 = int(os.environ.get("NFA_DIAG_K2", "1024"))
SUBSPACE = os.environ.get("NFA_DIAG_SUBSPACE", "full")
FAR_METHOD = os.environ.get("NFA_DIAG_FAR_METHOD", "")
SAMPLE = os.environ.get("NFA_DIAG_SAMPLE", "13")
MAX_NEW = int(os.environ.get("NFA_DIAG_MAX_NEW", "20"))
MODEL_PATH = Path.home() / "Qwen3-8B"
DATA_ROOT = Path.home() / "sparse-bench/third_party/KVCache-Factory/data/RULER"
TASK = "niah_multikey_2"
CONTEXT_LENGTH = 65536

# 与正式质量入口使用同一源行、tokenizer 和完整 input；目标坐标不得手写。
_rows, _data_sha256, _source_path = quality.load_ruler_rows(
    DATA_ROOT, CONTEXT_LENGTH, TASK)
_source_index = int(SAMPLE)
_selected = quality.select_ruler_rows(_rows, 0, 1, [_source_index])
if len(_selected) != 1:
    raise ValueError(f"诊断要求且只要求一个源样本: {SAMPLE}")
_row = _selected[0][1]
_tokenizer = quality.AutoTokenizer.from_pretrained(
    MODEL_PATH, trust_remote_code=True)
_input_ids = _tokenizer(_row["input"], truncation=False)["input_ids"]
_answer = _row["outputs"][0]
_answer_ids = _tokenizer(_answer, add_special_tokens=False)["input_ids"]
if len(_answer_ids) != 7:
    raise ValueError(f"目标答案必须是 7 个 token: sample={SAMPLE} ids={_answer_ids}")
_spans = []
for _start in range(len(_input_ids) - len(_answer_ids) + 1):
    if _input_ids[_start:_start + len(_answer_ids)] == _answer_ids:
        _spans.append((_start, _start + len(_answer_ids) - 1))
if len(_spans) != 2:
    raise ValueError(
        f"目标答案必须恰好出现两次: sample={SAMPLE} spans={_spans}")
_spans = tuple(_spans)
if SAMPLE == "13" and _spans != ((15951, 15957), (47939, 47945)):
    raise ValueError(f"sample13 真值坐标门禁失败: {_spans}")
_target_blocks = tuple(lo // 64 for lo, _ in _spans)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "target_truth.json").write_text(json.dumps({
    "task": TASK,
    "context_length": CONTEXT_LENGTH,
    "source_index": _source_index,
    "source_path": str(_source_path),
    "source_sha256": _data_sha256,
    "model_path": str(MODEL_PATH),
    "input_tokens": len(_input_ids),
    "input_ids_sha256": quality.input_ids_sha256(_input_ids),
    "answer": _answer,
    "answer_token_ids": _answer_ids,
    "spans": [
        {"lo": lo, "hi": hi, "block": lo // 64,
         "token_ids": _input_ids[lo:hi + 1]}
        for lo, hi in _spans
    ],
}, indent=2) + "\n", encoding="utf-8")
print(f"TARGET_TRUTH sample={SAMPLE} answer={_answer} "
      f"answer_ids={_answer_ids} spans={_spans} blocks={_target_blocks}", flush=True)
if os.environ.get("NFA_DIAG_TARGETS_ONLY", "0") == "1":
    raise SystemExit(0)

DENSE_GATHER = os.environ.get("NFA_DIAG_DENSE_GATHER", "0") == "1"
_original_resolve = quality.resolve_nfa_quality_config


def _resolve_with_diagnostic_budget(*args, **kwargs):
    patch_args, gate_config = _original_resolve(*args, **kwargs)
    patch_args.tia_level1_topk = K1
    patch_args.tia_level2_topk = K2
    if FAR_METHOD:
        patch_args.nfa_far_method = FAR_METHOD
    print(f"RESOLVED K1={patch_args.tia_level1_topk} K2={patch_args.tia_level2_topk} "
          f"near_tokens={patch_args.nfa_near_tokens} far_method={patch_args.nfa_far_method} "
          f"max_new={MAX_NEW} dense_gather={DENSE_GATHER}", flush=True)
    return patch_args, gate_config


quality.resolve_nfa_quality_config = _resolve_with_diagnostic_budget
quality._ruler_max_new_tokens = lambda task: MAX_NEW

if DENSE_GATHER:
    from sparse_attn.indexer import nfa_layer_batch_selector as selector

    def _dense_paged_gather(cache, layer_idx, req_ids, query_states, attention=None,
                            decode_write=None, **kwargs):
        """诊断专用：当前 K/V 写回后保留全部逻辑 token，不执行 L1/L2。"""
        if len(req_ids) != 1 or decode_write is None:
            raise ValueError("dense-gather 诊断仅支持 B1 eager decode")
        key_states, value_states, slot_mapping = decode_write
        cache.write_decode(layer_idx, key_states, value_states, slot_mapping)
        length = cache.planned_lengths_buf[0]
        logical = torch.arange(
            cache.active_visible_bucket, dtype=torch.int32, device=length.device)
        valid = logical < length
        pages = torch.clamp(logical // cache.geometry.page, max=cache.max_blocks - 1)
        physical = cache.block_table_buf[layer_idx, 0, pages.long()]
        slots = physical * cache.geometry.page + logical % cache.geometry.page
        gather = cache.gather_indices_for_layer(layer_idx)[0]
        gather.copy_(torch.where(valid, slots, -1).unsqueeze(0))
        cache.visible_lens_for_layer(layer_idx)[0].copy_(
            length.expand(gather.shape[0]))

    import torch
    selector.select_topk_for_layer_batched = _dense_paged_gather
elif os.environ.get("NFA_DIAG_TRACE_TARGETS", "0") == "1":
    from sparse_attn.indexer import nfa_layer_batch_selector as selector

    _original_select = selector.select_topk_for_layer_batched
    _original_filtered = selector._nfa_filtered_topk
    _filtered_steps = {}
    _select_steps = {}
    _hist_l2 = os.environ.get("NFA_DIAG_HIST_L2", "0") == "1"
    _hist_apply = _hist_l2 and os.environ.get("NFA_DIAG_HIST_L2_APPLY", "0") == "1"
    _torch_l2 = os.environ.get("NFA_DIAG_TORCH_L2", "0") == "1"
    _hist_l1 = os.environ.get("NFA_DIAG_HIST_L1", "0") == "1"
    _stats_oracle = os.environ.get("NFA_DIAG_STATS_ORACLE", "0") == "1"
    _l1_oracle = os.environ.get("NFA_DIAG_L1_ORACLE", "0") == "1"
    _fine_oracle = os.environ.get("NFA_DIAG_FINE_ORACLE", "0") == "1"
    _l1_variants = os.environ.get("NFA_DIAG_L1_VARIANTS", "0") == "1"
    _l1_exact = os.environ.get("NFA_DIAG_L1_EXACT", "0") == "1"
    _l1_apply_variant = os.environ.get("NFA_DIAG_L1_APPLY_VARIANT", "")
    _l2_variants = os.environ.get("NFA_DIAG_L2_VARIANTS", "0") == "1"
    _l2_apply_variant = os.environ.get("NFA_DIAG_L2_APPLY_VARIANT", "")
    _current = {}
    _original_l1_scores = selector._l1_scores
    _original_page_stats = selector._page_stats_all_layers
    _original_candidate = selector._candidate_pack_batched
    _original_fused_fine = selector._fused_fine_scores_paged
    _stats_steps = {}
    _stats_records = []
    _l1_steps = {}
    _l1_records = []
    _fine_steps = {}
    _fine_records = []
    _l1_variant_steps = {}
    _l1_variant_records = []
    _l2_variant_records = []

    def _stats_error(actual, expected):
        actual = actual.float()
        expected = expected.float()
        mismatch = actual != expected
        bad = torch.nonzero(mismatch)
        return {
            "mismatch": int(mismatch.sum().item()),
            "max_abs": float((actual - expected).abs().max().item()),
            "first_bad_bhd": bad[:5].cpu().tolist(),
        }

    def _save_stats_oracle():
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "stats_oracle.json").write_text(
            json.dumps(_stats_records, indent=2), encoding="utf-8")

    def _traced_page_stats(cache, req_ids, n_layers, dims, bs, pad, lengths,
                           layer_offset=0, workspace=None, q_job=None,
                           write_job=None, need_sum=True, region=0,
                           pool_near_len=64, sink_pool_lo=0):
        slot = int(req_ids[0].item())
        stats_len = getattr(cache, "_nfa_page_stats_lengths", None)
        previous_before = (None if stats_len is None else
                           int(stats_len[layer_offset, slot].item()))
        dirty_before = bool(getattr(cache, "_nfa_stats_dirty", False))
        result = _original_page_stats(
            cache, req_ids, n_layers, dims, bs, pad, lengths,
            layer_offset=layer_offset, workspace=workspace, q_job=q_job,
            write_job=write_job, need_sum=need_sum, region=region,
            pool_near_len=pool_near_len, sink_pool_lo=sink_pool_lo)
        if not _stats_oracle:
            return result
        step = _stats_steps.get(layer_offset, 0)
        _stats_steps[layer_offset] = step + 1
        if step >= 3:
            return result
        k_min, k_max, k_sum = result
        length = int(lengths[0].item())
        blocks = (length + bs - 1) // bs
        layer = cache.layers[layer_offset]
        physical = cache.block_table_buf[
            layer_offset, slot, :blocks].long()
        keys = layer.kv_pool_k[physical, 0][..., dims]
        positions = torch.arange(bs, device=keys.device)
        valid = positions[None, None, :, None] < torch.clamp(
            length - bs * torch.arange(blocks, device=keys.device),
            min=0, max=bs)[:, None, None, None]
        finite_keys = keys.float()
        direct_min = finite_keys.masked_fill(
            ~valid, float("inf")).amin(dim=2).to(k_min.dtype)
        direct_max = finite_keys.masked_fill(
            ~valid, float("-inf")).amax(dim=2).to(k_max.dtype)
        direct_sum = (finite_keys * valid).sum(dim=2).to(k_sum.dtype)
        actual = tuple(x[0, slot, :blocks] for x in result)
        expected = (direct_min, direct_max, direct_sum)
        errors = {
            name: _stats_error(a, e)
            for name, a, e in zip(("min", "max", "sum"), actual, expected)
        }
        targets = sorted(set(
            [0, blocks // 2, blocks - 1] + [lo // bs for lo, _ in _spans]))
        target_errors = {}
        for block in targets:
            if not 0 <= block < blocks:
                continue
            target_errors[str(block)] = {
                name: _stats_error(a[block], e[block])
                for name, a, e in zip(("min", "max", "sum"), actual, expected)
            }
        record = {
            "layer": layer_offset, "step": step, "length": length,
            "tail_tokens": length - bs * (blocks - 1), "blocks": blocks,
            "previous_before": previous_before,
            "previous_after": int(cache._nfa_page_stats_lengths[
                layer_offset, slot].item()),
            "dirty_before": dirty_before, "dims": int(dims.numel()),
            "errors": errors, "target_blocks": target_errors,
        }
        _stats_records.append(record)
        _save_stats_oracle()
        print(
            f"TRACE_STATS layer={layer_offset} step={step} length={length} "
            f"previous={previous_before}->{record['previous_after']} "
            f"dirty={dirty_before} blocks={blocks} tail={record['tail_tokens']} "
            + " ".join(
                f"{name}(bad={error['mismatch']},max={error['max_abs']:.9g})"
                for name, error in errors.items()), flush=True)
        return result

    def _torch_l2_selection(cand, logits, quota, width, sink):
        """诊断：同一 grouped-Q logits 的 PyTorch Top-K，绕过超宽 Triton 编译。"""
        token = cand[0].long()
        scores = logits[0].float()
        k2 = int(quota[0].item())
        top = torch.topk(scores, k2, dim=-1).indices
        selected = torch.gather(token, 1, top).sort(dim=-1).values
        selected_scores = torch.gather(scores, 1, top)
        keep = (selected >= sink) & torch.isfinite(selected_scores)
        counts = keep.sum(dim=-1, dtype=torch.int32).unsqueeze(0)
        out = torch.zeros(
            (1, cand.shape[1], width), dtype=torch.int32, device=token.device)
        for head in range(cand.shape[1]):
            retained = selected[head, keep[head]]
            out[0, head, :retained.numel()] = retained
        return selector.TokenSelection(out, counts)

    def _historical_l2(cand, logits, quota, width, sink, layer_offset, step):
        """诊断：复现历史 L2，并可记录同轨迹 GQA 聚合变体。"""
        cache = _current["cache"]
        query = _current["query"]
        attention = _current["attention"]
        if query.shape[0] != 1 or attention.indexer.enable_subspace:
            raise ValueError("历史 L2 诊断仅支持 B1 full128")
        layer = cache.layers[layer_offset]
        hkv = cand.shape[1]
        group = query.shape[1] // hkv
        page = cache.geometry.page
        dim = query.shape[-1]
        token = cand[0].long()
        pages = torch.clamp(token // page, max=cache.max_blocks - 1)
        physical = cache.block_table_buf[layer_offset, 0, pages]
        kv = torch.arange(hkv, device=token.device)[:, None]
        flat_index = ((physical * hkv + kv) * page + token % page)
        flat_k = layer.kv_pool_k.reshape(-1, dim)
        keys = flat_k[flat_index].float()
        q = query[0].view(hkv, group, dim).float() * attention.scaling
        q = q.to(torch.bfloat16).float()
        scores = torch.einsum("hgd,hrd->hgr", q, keys).to(torch.bfloat16)
        valid = torch.isfinite(logits[0])
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        per_head_probabilities = torch.softmax(scores, dim=-1)
        probability_mean = per_head_probabilities.mean(dim=1)
        probability_max = per_head_probabilities.amax(dim=1)
        variants = {
            "historical_mean": probability_mean,
            "probability_max": probability_max,
            "probability_top2_mean": per_head_probabilities.topk(
                2, dim=1).values.mean(dim=1),
            "probability_mean_max_average": (probability_mean + probability_max) / 2,
            "probability_lse": torch.logsumexp(
                per_head_probabilities.clamp_min(1e-45), dim=1),
            "logits_mean": scores.mean(dim=1),
            "logits_max": scores.amax(dim=1),
            "logits_lse": torch.logsumexp(scores, dim=1),
        }
        applied_name = _l2_apply_variant or "historical_mean"
        if applied_name not in variants:
            raise ValueError(f"未知 L2 variant: {applied_name}")
        aggregated = variants[applied_name]
        _current["historical_probabilities"] = aggregated
        _current["historical_valid"] = valid
        k2 = int(quota[0].item())
        if (valid.sum(dim=-1) < k2).any():
            raise ValueError("历史 L2 诊断不覆盖候选补洞分支")

        def _select(values):
            top = torch.topk(values, k2, dim=-1).indices
            selected = torch.gather(token, 1, top).sort(dim=-1).values
            keep = selected >= sink
            counts = keep.sum(dim=-1, dtype=torch.int32).unsqueeze(0)
            out = torch.zeros(
                (1, hkv, width), dtype=torch.int32, device=token.device)
            # 压掉 sink，保持 canonical middle 的升序紧凑布局。
            for head in range(hkv):
                retained = selected[head, keep[head]]
                out[0, head, :retained.numel()] = retained
            return selector.TokenSelection(out, counts)

        # 实际应用的选择只计算一次；decision-only 记录必须复用同一对象，
        # 不能依赖重复 Top-K 再次产生完全相同的边界槽位。
        applied_selection = _select(aggregated)
        if _l2_variants and step < 3:
            record = {"layer": layer_offset, "step": step,
                      "applied_variant": applied_name, "targets": {}}
            coverage = {}
            for lo, hi in _spans:
                block = lo // 64
                block_record = {}
                for name, values in variants.items():
                    variant_selection = (applied_selection if name == applied_name
                                         else _select(values))
                    selected = variant_selection.indices[0]
                    selected_valid = (torch.arange(width, device=token.device)[None, :]
                                      < variant_selection.counts[0, :, None])
                    hits = ((selected[:, None, :] >= lo)
                            & (selected[:, None, :] <= hi)
                            & selected_valid[:, None, :]).any(dim=(1, 2))
                    block_record[name] = {
                        "selected_heads": torch.nonzero(hits).flatten().cpu().tolist(),
                    }
                    coverage.setdefault(name, []).append(int(hits.sum().item()))
                record["targets"][str(block)] = block_record
            _l2_variant_records.append(record)
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "l2_variants.json").write_text(
                json.dumps(_l2_variant_records, indent=2), encoding="utf-8")
            print(f"TRACE_L2_VARIANTS layer={layer_offset} step={step} "
                  f"target_head_coverage={coverage}", flush=True)
        return applied_selection

    def _historical_l1_scores(k_min, k_max, k_sum, b_q_full, group_size,
                              indexer, dtype, **kwargs):
        """诊断：current stats 上复现历史 FP32-Q、逐头 BF16、GQA 均值 L1。"""
        far_orig, near_orig = _original_l1_scores(
            k_min, k_max, k_sum, b_q_full, group_size, indexer, dtype, **kwargs)
        query = _current["query"]
        attention = _current["attention"]
        if query.shape[0] != 1 or attention.indexer.enable_subspace:
            raise ValueError("历史 L1 诊断仅支持 B1 full128")
        q = query[0].float() * attention.scaling
        kmin = k_min[0, 0].float().repeat_interleave(group_size, dim=1)
        kmax = k_max[0, 0].float().repeat_interleave(group_size, dim=1)
        mm = (torch.einsum("hd,thd->ht", q.clamp(max=0), kmin)
              + torch.einsum("hd,thd->ht", q.clamp(min=0), kmax))
        hkv = k_min.shape[3]
        kt = k_min.shape[2]
        mm = mm.to(query.dtype).view(hkv, group_size, kt).mean(dim=1)
        bs = kwargs["bs"]
        kavg = (k_sum[0, 0].float() / bs).to(query.dtype).float()
        kavg = kavg.repeat_interleave(group_size, dim=1)
        avg = torch.einsum("hd,thd->ht", q, kavg)
        avg = avg.to(query.dtype).view(hkv, group_size, kt).mean(dim=1)
        far = torch.where(torch.isfinite(far_orig), mm.unsqueeze(0), far_orig)
        near = torch.where(torch.isfinite(near_orig), avg.unsqueeze(0), near_orig)
        return far, near

    def _record_l1_variants(actual_far, *args, **kwargs):
        """记录同预算 L1 聚合候选；仅显式 apply 时替换实际 far 分数。"""
        (k_min, k_max, k_sum, b_q_full, group_size,
         indexer, dtype) = args[:7]
        cache = kwargs["cache"]
        lengths = kwargs["lengths"]
        bs = kwargs["bs"]
        region = kwargs.get("region", 0)
        layer_offset = kwargs.get("layer_offset", 0)
        step = _l1_variant_steps.get(layer_offset, 0)
        _l1_variant_steps[layer_offset] = step + 1
        if step >= 3 and not _l1_apply_variant:
            return
        query = _current["query"]
        attention = _current["attention"]
        if (query.shape[0] != 1 or k_min.shape[0] != 1 or region != 0
                or attention.indexer.enable_subspace):
            raise ValueError("L1 variants 诊断仅支持 B1 full128 region=0")
        length = int(lengths[0].item())
        kt = k_min.shape[2]
        kt_eff = (length + bs - 1) // bs
        near_start = max(
            (length - cache.geometry.window
             - max(bs, int(indexer.nt_near_len))) // bs,
            int(indexer._sink_pool_lo_blk))
        far_end = min(kt_eff, near_start)
        q = query[0].float() * attention.scaling
        kmin = k_min[0, 0].float().repeat_interleave(group_size, dim=1)
        kmax = k_max[0, 0].float().repeat_interleave(group_size, dim=1)
        mm = (torch.einsum("hd,thd->ht", q.clamp(max=0), kmin)
              + torch.einsum("hd,thd->ht", q.clamp(min=0), kmax))
        hkv = k_min.shape[3]
        mm = mm.to(query.dtype).view(hkv, group_size, kt)
        kavg = (k_sum[0, 0].float() / bs).to(query.dtype).float()
        kavg = kavg.repeat_interleave(group_size, dim=1)
        avg = torch.einsum("hd,thd->ht", q, kavg)
        avg = avg.to(query.dtype).view(hkv, group_size, kt)
        mm_mean = mm.mean(dim=1)
        mm_max = mm.max(dim=1).values
        avg_mean = avg.mean(dim=1)
        avg_max = avg.max(dim=1).values
        variants = {
            "production": actual_far[0],
            "mm_mean": mm_mean,
            "mm_max": mm_max,
            "mm_top2_mean": mm.topk(2, dim=1).values.mean(dim=1),
            "mm_mean_max_average": (mm_mean + mm_max) / 2,
            "avg_mean": avg_mean,
            "avg_max": avg_max,
            "avg_top2_mean": avg.topk(2, dim=1).values.mean(dim=1),
            "avg_mean_max_average": (avg_mean + avg_max) / 2,
            "mm_mean_plus_avg_mean": mm_mean + avg_mean,
            "mm_max_plus_avg_mean": mm_max + avg_mean,
        }
        if _l1_exact:
            # 决策侧上界 oracle：直接读取当前层真实 paged K，逐 token 计算与
            # 历史 fine-score 相同的 BF16 logit，再聚合到页。只用于判断压缩
            # 页统计是否漏掉目标；不作为可生产的 dense L1 实现。
            layer = cache.layers[layer_offset]
            physical = cache.block_table_buf[
                layer_offset, 0, :kt_eff].long()
            keys = layer.kv_pool_k[physical, 0].float()
            q_exact = q.view(hkv, group_size, -1).to(torch.bfloat16).float()
            token_scores = torch.einsum(
                "hgd,thpd->hgtp", q_exact, keys).to(torch.bfloat16).float()
            page = int(cache.geometry.page)
            if token_scores.shape[-1] != page:
                raise ValueError("exact-page 诊断要求 page 与统计块一致")
            token_valid = (torch.arange(kt_eff * page, device=token_scores.device)
                           < length)
            token_scores = token_scores.reshape(hkv, group_size, -1)
            token_scores = token_scores.masked_fill(
                ~token_valid[None, None, :], float("-inf"))
            full_lse = torch.logsumexp(token_scores, dim=-1, keepdim=True)
            token_probabilities = torch.exp(token_scores - full_lse)
            mean_probability = token_probabilities.mean(dim=1)
            max_probability = token_probabilities.amax(dim=1)
            token_scores = token_scores.reshape(
                hkv, group_size, kt_eff, page)
            mean_probability = mean_probability.reshape(hkv, kt_eff, page)
            max_probability = max_probability.reshape(hkv, kt_eff, page)
            page_max = token_scores.amax(dim=-1)
            page_lse = torch.logsumexp(token_scores, dim=-1)

            def _active_exact_score(value):
                score = torch.full_like(actual_far[0], float("-inf"))
                score[:, :kt_eff] = value
                return score

            variants.update({
                "exact_page_max_mean": _active_exact_score(page_max.mean(dim=1)),
                "exact_page_max_max": _active_exact_score(page_max.amax(dim=1)),
                "exact_page_lse_mean": _active_exact_score(page_lse.mean(dim=1)),
                "exact_page_lse_max": _active_exact_score(page_lse.amax(dim=1)),
                # 与 all-page 历史 L2 直接对齐的 dense 上界：先对每个
                # query head 做全序列 softmax，再聚合 GQA 概率并取页内最大。
                "exact_dense_meanprob_max": _active_exact_score(
                    mean_probability.amax(dim=-1)),
                "exact_dense_maxprob_max": _active_exact_score(
                    max_probability.amax(dim=-1)),
                "exact_dense_meanprob_sum": _active_exact_score(
                    mean_probability.sum(dim=-1)),
            })
        valid = torch.isfinite(actual_far[0])
        far_quota = max(
            0, int(indexer.args.tia_level1_topk) - int(indexer.nt_near_pages))
        record = {"layer": layer_offset, "step": step,
                  "length": length, "far_end": far_end,
                  "far_quota": far_quota, "targets": {},
                  "applied_variant": _l1_apply_variant or None}
        coverage = {}
        applied_score = None
        for block in _target_blocks:
            block_record = {}
            for name, score in variants.items():
                score = torch.where(valid, score, actual_far[0]).to(query.dtype)
                if name == _l1_apply_variant:
                    applied_score = score
                if step >= 3:
                    continue
                ranked = _l1_reference_topk(score, kt, 0, far_end)
                rank = (ranked == block).nonzero()[:, 1].reshape(hkv, -1)
                selected = (ranked[:, :far_quota] == block).any(dim=-1)
                # 同一总 far 页预算的跨头上界：全局仍只保留 hkv*quota
                # 页，但允许单个 head 超过原固定 quota。另记录 cap256 是否
                # 足以容纳关键页，避免把全局配额误读成隐式无限扩容。
                active_scores = score[:, :far_end].reshape(-1)
                target_scores = score[:, block]
                flat_indices = torch.arange(
                    hkv * far_end, device=score.device)
                target_indices = (torch.arange(hkv, device=score.device)
                                  * far_end + block)
                greater = active_scores[None, :] > target_scores[:, None]
                tie_before = ((active_scores[None, :] == target_scores[:, None])
                              & (flat_indices[None, :] < target_indices[:, None]))
                global_rank = (greater.sum(dim=1) + tie_before.sum(dim=1))
                global_selected = global_rank < hkv * far_quota
                block_record[name] = {
                    "far_rank": rank.cpu().tolist(),
                    "selected_heads": torch.nonzero(
                        selected).flatten().cpu().tolist(),
                    "global_rank": global_rank.cpu().tolist(),
                    "global_selected_heads": torch.nonzero(
                        global_selected).flatten().cpu().tolist(),
                    "global_cap256_selected_heads": torch.nonzero(
                        global_selected & (rank[:, 0] < 256)).flatten().cpu().tolist(),
                    "far_score": score[:, block].float().cpu().tolist(),
                }
                coverage.setdefault(name, []).append(int(selected.sum().item()))
            record["targets"][str(block)] = block_record
        if _l1_apply_variant:
            if applied_score is None:
                raise ValueError(f"未知 L1 variant: {_l1_apply_variant}")
            _current["l1_applied_far"] = applied_score
        if step < 3:
            _l1_variant_records.append(record)
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "l1_variants.json").write_text(
                json.dumps(_l1_variant_records, indent=2), encoding="utf-8")
            print(f"TRACE_L1_VARIANTS layer={layer_offset} step={step} "
                  f"target_head_coverage={coverage}", flush=True)

    def _l1_scores_diagnostic(*args, **kwargs):
        result = (_historical_l1_scores(*args, **kwargs) if _hist_l1
                  else _original_l1_scores(*args, **kwargs))
        if _l1_variants or _l1_apply_variant:
            _record_l1_variants(result[0], *args, **kwargs)
        if _l1_apply_variant:
            return _current["l1_applied_far"].unsqueeze(0), result[1]
        return result

    def _head_lists(hits):
        return (torch.nonzero(hits.any(dim=1)).flatten().cpu().tolist(),
                torch.nonzero(hits.all(dim=1)).flatten().cpu().tolist())

    def _l1_reference_topk(scores, k, active_lo=0, active_hi=None):
        kt = scores.shape[-1]
        if active_hi is None:
            active_hi = kt
        offs = torch.arange(kt, device=scores.device)
        active = (offs >= active_lo) & (offs < active_hi)
        # 生产 score 只提交 [0, kt_eff)；容量尾部是未初始化 workspace，
        # reference 必须复现 active mask，不能把 poison 当作候选分数。
        active_scores = torch.where(
            active, scores, torch.full_like(scores, float("-inf")))
        bits = active_scores.view(torch.int16).to(torch.int32) & 0xFFFF
        negative = (bits & 0x8000) != 0
        ordered = torch.where(
            negative, (~bits) & 0xFFFF, bits | 0x8000) - 0x8000
        packed = ((ordered.to(torch.int64) << 16)
                  | ((kt - 1 - offs) & 0xFFFF))
        top = torch.topk(packed, k, dim=-1).values
        return (kt - 1 - (top & 0xFFFF)).to(torch.int32)

    def _save_l1_oracle():
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "l1_oracle.json").write_text(
            json.dumps(_l1_records, indent=2), encoding="utf-8")

    def _candidate_reference(i_f, i_n, far, near, length, cache, indexer, bs,
                             capacity, out):
        """由实际 L1 索引独立展开 candidate，包含 bypass 与 protected。"""
        hkv, k_f = i_f[0].shape
        k_n = i_n.shape[-1]
        sink = int(indexer.sink_tokens)
        swa = int(indexer.sliding_window_size)
        head = min(sink, capacity)
        tail = min(swa, capacity)
        pack_r = (k_f + k_n + 2) * bs
        if out.shape[-1] != pack_r + head + tail:
            raise ValueError("candidate oracle 当前只支持 full-pack region=0")
        device = out.device
        r = torch.arange(out.shape[-1], device=device)
        lo = max(0, length - swa)
        kt = far.shape[-1]
        kt_eff = (length + bs - 1) // bs
        far_slot = r < k_f * bs
        near_slot = (r >= k_f * bs) & (r < (k_f + k_n) * bs)
        l1 = far_slot | near_slot
        far_idx = torch.clamp(r // bs, max=max(0, k_f - 1))
        near_idx = torch.clamp((r - k_f * bs) // bs, min=0, max=max(0, k_n - 1))
        blk_f = i_f[0][:, far_idx]
        blk_n = i_n[0][:, near_idx]
        blk = torch.where(far_slot[None, :], blk_f, blk_n)
        safe_f = torch.clamp(blk_f, min=0, max=kt - 1)
        safe_n = torch.clamp(blk_n, min=0, max=kt - 1)
        score_f = far[0].gather(1, safe_f.long())
        score_n = near[0].gather(1, safe_n.long())
        score_f = torch.where(
            far_slot[None, :] & (blk_f < kt_eff), score_f,
            torch.full_like(score_f, float("-inf")))
        score_n = torch.where(
            near_slot[None, :] & (blk_n < kt_eff), score_n,
            torch.full_like(score_n, float("-inf")))
        score = torch.where(far_slot[None, :], score_f, score_n)
        k1_width = (k_f + k_n) * bs
        byp0 = (r >= k1_width) & (r < k1_width + bs)
        byp1 = (r >= k1_width + bs) & (r < pack_r)
        swa_byp_blk = lo // bs
        byp_blk = torch.where(
            byp0, torch.full_like(r, int(indexer._sink_pool_lo_blk)),
            torch.full_like(r, swa_byp_blk))
        byp0_keep = sink % bs != 0
        byp1_keep = ((lo % bs != 0)
                     and swa_byp_blk >= int(indexer._sink_pool_lo_blk))
        if sink % bs != 0:
            byp1_keep = byp1_keep and swa_byp_blk != int(indexer._sink_pool_lo_blk)
        token = torch.where(
            l1[None, :], blk * bs + r[None, :] % bs,
            byp_blk[None, :] * bs + r[None, :] % bs)
        bypass_keep = torch.where(
            byp0, torch.full_like(r, byp0_keep, dtype=torch.bool),
            torch.full_like(r, byp1_keep, dtype=torch.bool))
        keep = torch.where(
            l1[None, :], score != float("-inf"), bypass_keep[None, :])
        active = ((r[None, :] < pack_r) & keep & (token >= sink)
                  & (token < lo) & (token >= 0) & (token < length))
        head_slot = (r >= pack_r) & (r < pack_r + head)
        tail_slot = (r >= pack_r + head) & (r < pack_r + head + tail)
        new_id = torch.where(
            head_slot, r - pack_r, lo + r - pack_r - head)
        new_valid = ((head_slot & (new_id < length))
                     | (tail_slot & (new_id < length) & (new_id >= sink)))
        token = torch.where((r < pack_r)[None, :], token, new_id[None, :])
        active = torch.where((r < pack_r)[None, :], active, new_valid[None, :])
        return token.to(out.dtype), active

    def _traced_candidate(*args, **kwargs):
        result = _original_candidate(*args, **kwargs)
        _current["candidate_valid"] = result[1]
        if not _l1_oracle:
            return result
        i_f, i_n, far, near, lengths, cache, valid_req, indexer, bs, capacity = args[:10]
        layer_offset = kwargs.get("layer_offset", 0)
        step = _l1_steps.get(layer_offset, 0)
        _l1_steps[layer_offset] = step + 1
        if step >= 3 or i_f.shape[0] != 1:
            return result
        length = int(lengths[0].item())
        kt = far.shape[-1]
        kt_eff = (length + bs - 1) // bs
        pool_near_len = max(bs, int(indexer.nt_near_len))
        near_start = max(
            (length - cache.geometry.window - pool_near_len) // bs,
            int(indexer._sink_pool_lo_blk))
        near_end = max(near_start, kt_eff - max(1, cache.geometry.window // bs))
        far_end = min(kt_eff, near_start)
        near_width = (pool_near_len + bs - 1) // bs + 2
        compact = (near_start >= i_n.shape[-1]
                   and near_start + near_width >= near_end)
        reference_far = _l1_reference_topk(
            far[0], i_f.shape[-1], 0, far_end)
        reference_near = _l1_reference_topk(
            near[0], i_n.shape[-1], near_start, near_end)
        legacy_reference_near = _l1_reference_topk(near[0], i_n.shape[-1])
        bad_far = torch.nonzero(
            (i_f[0] != reference_far).any(dim=-1)).flatten().cpu().tolist()
        bad_near = torch.nonzero(
            (i_n[0] != reference_near).any(dim=-1)).flatten().cpu().tolist()
        legacy_bad_near = torch.nonzero(
            (i_n[0] != legacy_reference_near).any(dim=-1)).flatten().cpu().tolist()
        expected_out, expected_valid = _candidate_reference(
            i_f, i_n, far, near, length, cache, indexer, bs, capacity, result[0])
        candidate_bad = torch.nonzero(
            ((result[0][0] != expected_out)
             | (result[1][0] != expected_valid)).any(dim=-1)).flatten().cpu().tolist()
        targets = {}
        for block in _target_blocks:
            actual_far = (i_f[0] == block).any(dim=-1)
            actual_near = (i_n[0] == block).any(dim=-1)
            ref_far = (reference_far == block).any(dim=-1)
            ref_near = (reference_near == block).any(dim=-1)
            far_packed = _l1_reference_topk(far[0], far.shape[-1], 0, far_end)
            near_packed = _l1_reference_topk(
                near[0], near.shape[-1], near_start, near_end)
            # 全量 active 排序仅用于取得确定性 rank，不改变实际候选。
            far_rank = (far_packed == block).nonzero()[:, 1].reshape(8, -1)
            near_rank = (near_packed == block).nonzero()[:, 1].reshape(8, -1)
            targets[str(block)] = {
                "actual_far_heads": torch.nonzero(actual_far).flatten().cpu().tolist(),
                "reference_far_heads": torch.nonzero(ref_far).flatten().cpu().tolist(),
                "actual_near_heads": torch.nonzero(actual_near).flatten().cpu().tolist(),
                "reference_near_heads": torch.nonzero(ref_near).flatten().cpu().tolist(),
                "far_rank": far_rank.cpu().tolist(),
                "near_rank": near_rank.cpu().tolist(),
                "far_score": far[0, :, block].float().cpu().tolist(),
                "near_score": near[0, :, block].float().cpu().tolist(),
            }
        near_examples = {}
        for head in bad_near[:2]:
            near_examples[str(head)] = {
                "actual": i_n[0, head].cpu().tolist(),
                "reference": reference_near[head].cpu().tolist(),
            }
        record = {
            "layer": layer_offset, "step": step, "length": length,
            "kt_eff": kt_eff, "far_end": far_end,
            "near_start": near_start, "near_end": near_end,
            "near_compact": compact,
            "far_mismatch_heads": bad_far, "near_mismatch_heads": bad_near,
            "legacy_full_array_near_mismatch_heads": legacy_bad_near,
            "near_examples": near_examples,
            "candidate_mismatch_heads": candidate_bad,
            "targets": targets,
        }
        _l1_records.append(record)
        _save_l1_oracle()
        print(f"TRACE_L1_ORACLE layer={layer_offset} step={step} "
              f"far_mismatch={bad_far} near_mismatch={bad_near} "
              f"legacy_near_mismatch={legacy_bad_near} "
              f"candidate_mismatch={candidate_bad}", flush=True)
        return result

    def _save_fine_oracle():
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "fine_oracle.json").write_text(
            json.dumps(_fine_records, indent=2), encoding="utf-8")

    def _json_float(value):
        value = float(value)
        return value if torch.isfinite(torch.tensor(value)) else str(value)

    def _fine_reference(cache, req_ids, dims, q_fine, cand, keep, owners,
                        generations, capacity, lower, upper, quota,
                        layer_offset, bs, pack_width, region):
        """独立重放 grouped-Q、owner 标签和真实 paged K gather。"""
        if cand.shape[0] != 1 or len(req_ids) != 1 or region != 0:
            raise ValueError("fine oracle 当前仅支持 B1 full-pack region=0")
        query = _current["query"]
        attention = _current["attention"]
        hkv = cand.shape[1]
        group = query.shape[1] // hkv
        if query.shape[0] != 1 or q_fine.shape[:2] != (1, hkv):
            raise ValueError("fine oracle 的 Q/request 几何不匹配")
        q_rows = []
        for head in range(hkv):
            grouped = torch.zeros(
                dims.numel(), dtype=torch.float32, device=query.device)
            for j in range(group):
                q = query[0, head * group + j, dims].float()
                grouped = grouped + (q * attention.scaling).to(
                    torch.bfloat16).float()
            q_rows.append(grouped)
        q_ref = torch.stack(q_rows).unsqueeze(0)
        r = torch.arange(cand.shape[-1], device=cand.device)
        token = cand[0].long()
        valid = keep[0] & (token >= 0) & (token < capacity)
        valid &= ((token >= lower[0]) & (token < upper[0]))
        page = cache.geometry.page
        safe = torch.where(valid, token, torch.zeros_like(token))
        logical_page = safe // page
        valid &= logical_page < cache.max_blocks
        block_last_r = (r // bs) * bs + bs - 1
        owner_slot = (safe // bs).clamp(0, owners.shape[-1] - 1)
        owner = owners[0].gather(1, owner_slot)
        expected_tag = (generations[0, :, None] * pack_width
                        + block_last_r[None, :])
        unique = valid & (owner == expected_tag)
        if int(quota[0].item()) == 0:
            unique.zero_()
        reference = torch.full(
            cand.shape, float("-inf"), dtype=torch.float32, device=cand.device)
        pool = cache.layers[layer_offset].kv_pool_k
        for head in range(hkv):
            slots = torch.nonzero(unique[head]).flatten()
            if slots.numel() == 0:
                continue
            selected = token[head, slots]
            keys = pool[
                selected // page, 0, head, selected % page][:, dims].float()
            scores = (keys * q_ref[0, head][None, :]).sum(dim=-1)
            scores = scores.to(torch.bfloat16).float()
            scores = torch.where(
                scores == scores, scores,
                torch.full_like(scores, float("-inf")))
            reference[0, head, slots] = scores
        return reference.to(torch.bfloat16), q_ref, unique

    def _traced_fine(*args, **kwargs):
        logits = _original_fused_fine(*args, **kwargs)
        _current["fine_record"] = None
        if not _fine_oracle:
            return logits
        (cache, req_ids, n_layers, dims, q_fine, cand, keep, owners,
         generations, capacity, lower, upper, quota) = args[:13]
        layer_offset = kwargs.get("layer_offset", 0)
        step = _fine_steps.get(layer_offset, 0)
        _fine_steps[layer_offset] = step + 1
        if step >= 3 or cand.shape[0] != 1:
            return logits
        bs = kwargs.get("bs", 64)
        pack_width = kwargs.get("pack_width", 0) or cand.shape[-1]
        region = kwargs.get("region", 0)
        reference, q_ref, unique = _fine_reference(
            cache, req_ids, dims, q_fine, cand, keep, owners, generations,
            capacity, lower, upper, quota, layer_offset, bs, pack_width, region)
        q_mismatch = q_fine != q_ref
        finite_pair = torch.isfinite(logits) & torch.isfinite(reference)
        numeric_mismatch = finite_pair & (logits != reference)
        finite_disagreement = torch.isfinite(logits) != torch.isfinite(reference)
        mismatch = numeric_mismatch | finite_disagreement
        max_abs = 0.0
        if finite_pair.any():
            max_abs = float(
                (logits.float() - reference.float()).abs()[finite_pair].max().item())
        examples = []
        for head, slot in torch.nonzero(mismatch[0])[:5].cpu().tolist():
            examples.append({
                "head": head, "slot": slot,
                "token": int(cand[0, head, slot].item()),
                "actual": _json_float(logits[0, head, slot].item()),
                "reference": _json_float(reference[0, head, slot].item()),
                "reference_unique": bool(unique[head, slot].item()),
            })
        targets = []
        for lo, hi in _spans:
            for position in range(lo, hi + 1):
                for head in range(cand.shape[1]):
                    hits = torch.nonzero(
                        (cand[0, head] == position) & keep[0, head]).flatten()
                    for slot in hits.cpu().tolist():
                        row_logits = logits[0, head]
                        row_token = cand[0, head]
                        good = (torch.isfinite(row_logits)
                                & (row_token >= lower[0]) & (row_token < upper[0]))
                        bits = row_logits.view(torch.int16).to(torch.int32) & 0xFFFF
                        negative = (bits & 0x8000) != 0
                        ordered = torch.where(
                            negative, (~bits) & 0xFFFF, bits | 0x8000) - 0x8000
                        packed = ((ordered.to(torch.int64) << 32)
                                  + capacity - row_token.to(torch.int64))
                        target_key = packed[slot]
                        raw_rank = int((good & (packed > target_key)).sum().item()) + 1
                        targets.append({
                            "token": position, "head": head, "slot": slot,
                            "actual": _json_float(logits[0, head, slot].item()),
                            "reference": _json_float(reference[0, head, slot].item()),
                            "reference_unique": bool(unique[head, slot].item()),
                            "raw_rank": raw_rank,
                            "raw_quota": int(quota[0].item()),
                            "raw_selected_by_rank": raw_rank <= int(quota[0].item()),
                            "middle": None,
                        })
        record = {
            "layer": layer_offset, "step": step,
            "q_mismatch": int(q_mismatch.sum().item()),
            "q_max_abs": float((q_fine - q_ref).abs().max().item()),
            "logit_mismatch": int(mismatch.sum().item()),
            "numeric_mismatch": int(numeric_mismatch.sum().item()),
            "finite_disagreement": int(finite_disagreement.sum().item()),
            "max_abs": max_abs,
            "mismatch_heads": torch.nonzero(
                mismatch[0].any(dim=-1)).flatten().cpu().tolist(),
            "reference_unique_counts": unique.sum(dim=-1).cpu().tolist(),
            "examples": examples, "targets": targets,
        }
        _fine_records.append(record)
        _current["fine_record"] = record
        _save_fine_oracle()
        print(f"TRACE_FINE_ORACLE layer={layer_offset} step={step} "
              f"q_mismatch={record['q_mismatch']} "
              f"logit_mismatch={record['logit_mismatch']} "
              f"numeric={record['numeric_mismatch']} "
              f"finite_disagreement={record['finite_disagreement']} "
              f"max_abs={max_abs} mismatch_heads={record['mismatch_heads']}",
              flush=True)
        return logits

    def _traced_filtered(logits, cand_idx2, lower, upper, quota, capacity, width,
                         sink, layer_offset=0, workspace=None, name="middle"):
        if _torch_l2 and name == "middle" and cand_idx2.shape[0] == 1:
            result = _torch_l2_selection(cand_idx2, logits, quota, width, sink)
        else:
            result = _original_filtered(
                logits, cand_idx2, lower, upper, quota, capacity, width, sink,
                layer_offset=layer_offset, workspace=workspace, name=name)
        historical = None
        step = _filtered_steps.get(layer_offset, 0)
        if _hist_l2 and name == "middle" and cand_idx2.shape[0] == 1:
            historical = _historical_l2(
                cand_idx2, logits, quota, width, sink, layer_offset, step)
        _filtered_steps[layer_offset] = step + 1
        if step < 3 and name == "middle" and cand_idx2.shape[0] == 1:
            import triton

            hkv, k = result.indices.shape[1:]
            columns = torch.arange(k, device=result.indices.device)
            valid = columns[None, :] < result.counts[0, :, None]
            fine_record = _current.get("fine_record")
            if fine_record is not None:
                for item in fine_record["targets"]:
                    head = item["head"]
                    count = int(result.counts[0, head].item())
                    item["middle"] = bool(
                        (result.indices[0, head, :count] == item["token"]).any().item())
                    if historical is not None:
                        probability = _current["historical_probabilities"][head]
                        historical_candidate_valid = _current["historical_valid"][head]
                        slot = item["slot"]
                        target_probability = probability[slot]
                        item["historical_probability"] = _json_float(
                            target_probability.item())
                        item["historical_raw_rank_strict_greater"] = int(
                            (historical_candidate_valid
                             & (probability > target_probability)).sum().item()) + 1
                        item["historical_tie_count"] = int(
                            (historical_candidate_valid
                             & (probability == target_probability)).sum().item())
                        item["historical_selected"] = bool(
                            (historical.indices[0, head, :int(
                                historical.counts[0, head].item())]
                             == item["token"]).any().item())
                _save_fine_oracle()
            if historical is not None:
                historical_valid = columns[None, :] < historical.counts[0, :, None]
                overlap = ((result.indices[0, :, :, None]
                            == historical.indices[0, :, None, :])
                           & valid[:, :, None] & historical_valid[:, None, :])
                overlap = overlap.any(dim=-1).sum(dim=-1)
                print(f"TRACE_HIST_L2 layer={layer_offset} step={step} "
                      f"apply={_hist_apply} overlap={overlap.cpu().tolist()} "
                      f"current_counts={result.counts[0].cpu().tolist()} "
                      f"historical_counts={historical.counts[0].cpu().tolist()}",
                      flush=True)
            reference_indices = reference_counts = None
            if step == 0 and not _torch_l2 and not _hist_apply:
                reference_indices = torch.empty_like(result.indices)
                reference_counts = torch.empty_like(result.counts)
                token_bits = 32 if capacity >= 65536 else 16
                selector._nfa_filtered_topk_kernel[(cand_idx2.shape[0] * hkv,)](
                    logits, cand_idx2, lower, upper, quota,
                    reference_indices, reference_counts, cand_idx2.shape[-1],
                    R_PAD=triton.next_power_of_2(cand_idx2.shape[-1]),
                    S=capacity, K=width, SINK=sink, META_GROUP=hkv,
                    TOKEN_BITS=token_bits, num_warps=16)
                reference_valid = columns[None, :] < reference_counts[0, :, None]
                mismatch = (result.counts[0] != reference_counts[0])
                mismatch |= ((result.indices[0] != reference_indices[0])
                             & (valid | reference_valid)).any(dim=-1)
                bad = torch.nonzero(mismatch).flatten().cpu().tolist()
                print(f"TRACE_L2_REF layer={layer_offset} step={step} "
                      f"capacity={capacity} token_bits={token_bits} "
                      f"bitexact={not bad} mismatch_heads={bad}", flush=True)
            for lo, hi in _spans:
                positions = torch.arange(lo, hi + 1, device=cand_idx2.device)
                cand_hits = ((cand_idx2[0, :, None, :] == positions[None, :, None])
                             & _current["candidate_valid"][0, :, None, :]).any(dim=-1)
                middle_hits = ((result.indices[0, :, None, :] == positions[None, :, None])
                               & valid[:, None, :]).any(dim=-1)
                historical_note = ""
                if historical is not None:
                    historical_hits = (
                        (historical.indices[0, :, None, :] == positions[None, :, None])
                        & historical_valid[:, None, :]).any(dim=-1)
                    historical_note = f" historical(any,all)={_head_lists(historical_hits)}"
                reference_note = ""
                if reference_indices is not None:
                    reference_hits = (
                        (reference_indices[0, :, None, :] == positions[None, :, None])
                        & reference_valid[:, None, :]).any(dim=-1)
                    reference_note = f" reference(any,all)={_head_lists(reference_hits)}"
                print(f"TRACE_FILTER layer={layer_offset} step={step} span={lo}-{hi} "
                      f"candidate(any,all)={_head_lists(cand_hits)} "
                      f"middle(any,all)={_head_lists(middle_hits)}"
                      f"{historical_note}{reference_note}", flush=True)
        if _hist_apply and historical is not None:
            return historical
        return result

    def _traced_select(*args, **kwargs):
        cache, layer_idx, req_ids, query_states = args[:4]
        attention = kwargs.get("attention", args[4] if len(args) > 4 else None)
        _current.update(cache=cache, query=query_states, attention=attention)
        workspace = getattr(cache, "_nfa_selector_workspace", None)
        plan_key = ("eager_selector_call_plan", layer_idx)
        if (_hist_apply or _torch_l2 or _hist_l1 or _stats_oracle
                or _l1_oracle or _fine_oracle or _l1_variants
                or _l1_apply_variant or _l2_variants
                or _l2_apply_variant) and workspace is not None:
            workspace.pop(plan_key, None)
        try:
            result = _original_select(*args, **kwargs)
        finally:
            if (_hist_apply or _torch_l2 or _hist_l1 or _stats_oracle
                    or _l1_oracle or _fine_oracle or _l1_variants
                    or _l1_apply_variant or _l2_variants
                or _l2_apply_variant) and workspace is not None:
                workspace.pop(plan_key, None)
        step = _select_steps.get(layer_idx, 0)
        _select_steps[layer_idx] = step + 1
        if step < 3:
            gather = cache.gather_indices_for_layer(layer_idx)[0]
            visible = cache.visible_lens_for_layer(layer_idx)[0]
            columns = torch.arange(gather.shape[-1], device=gather.device)
            valid = columns[None, :] < visible[:, None]
            for lo, hi in _spans:
                logical = torch.arange(lo, hi + 1, device=gather.device)
                physical = (cache.block_table_buf[
                    layer_idx, 0, (logical // cache.geometry.page).long()]
                    * cache.geometry.page + logical % cache.geometry.page)
                hits = ((gather[:, None, :] == physical[None, :, None])
                        & valid[:, None, :]).any(dim=-1)
                print(f"TRACE_UNION layer={layer_idx} step={step} span={lo}-{hi} "
                      f"final(any,all)={_head_lists(hits)}", flush=True)
        return result

    selector._nfa_filtered_topk = _traced_filtered
    selector._l1_scores = _l1_scores_diagnostic
    selector._page_stats_all_layers = _traced_page_stats
    selector._candidate_pack_batched = _traced_candidate
    selector._fused_fine_scores_paged = _traced_fine
    selector.select_topk_for_layer_batched = _traced_select
sys.argv = [
    "benchmark_ruler_quality.py",
    "--method", "nfa", "--execution", "eager",
    "--nfa-subspace", SUBSPACE, "--nfa-gate", "off",
    "--model-path", str(MODEL_PATH),
    "--model-name", "Qwen3-8B",
    "--task", TASK, "--context-length", str(CONTEXT_LENGTH),
    "--data-root", str(DATA_ROOT),
    "--output-dir", str(OUT), "--sample-indices", SAMPLE,
    "--device", "cuda:0", "--visible-len",
    str(max(8192, K2, 65536 if DENSE_GATHER else 8192)),
]
print(f"DIAG sample={SAMPLE} subspace={SUBSPACE} K1={K1} K2={K2} out={OUT}", flush=True)
quality.main()
