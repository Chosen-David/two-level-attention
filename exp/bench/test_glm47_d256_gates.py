#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GLM-4.7 D=256 TLI 合成正确性门禁（不加载真实模型）。

用法（仓根）：
  PYTHONPATH=$HOME/.local/pylibs_tf512:$PWD PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES=0 python3 exp/bench/test_glm47_d256_gates.py

仅测合成 KV 与小型真实 HF 5.12.1 Glm4MoeLiteAttention。D=128 图测试是
独立对照；本文件不调用旧 G7/G8/G9，任何普遍基线差异均如实报 FAIL。
"""
import argparse
import os
import sys
import traceback
from types import MethodType, SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import torch  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"[PASS] {name} {detail}", flush=True)
    else:
        FAIL.append(name)
        print(f"[FAIL] {name} {detail}", flush=True)


def run_group(name, fn):
    print(f"\n===== {name} =====", flush=True)
    try:
        fn()
    except Exception as exc:
        FAIL.append(name)
        print(f"[FAIL] {name} 未捕获异常: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()


def make_prod_args(head_dim=256):
    args = argparse.Namespace(
        method="tli", tia_block_size=64, tia_level1_topk=128,
        tia_level2_topk=1024, tia_level2_cmp_ratio=4,
        tia_enable_async_topk=False, tli_enable_subspace=True,
        tli_enable_kmeans=False, tli_far_select="4bit",
        tli_near_select="4bit", tli_far_clusters=256, tli_far_niter=10,
        tli_far_blocks=16, tli_far_tokens=512, tli_far_method="minmax",
        tli_near_method="avg", tli_sim=0.9, tli_sim_dims="subspace",
        tli_sigma_select="none", tli_moba=False, tli_sigma=8.0,
        tli_sink_tokens=4, tli_swa_size=128, tli_sinkswa_only=False,
        tli_dyn_dgate=False, tli_dyn_dgate_eps=0.05,
        tli_dyn_dgate_rows=256, tli_per_q_head=False,
        tli_enable_layer_skip=False, tli_layer_skip_path=None,
        tli_alpha=None, tli_beta=None, tli_gamma=None,
        tli_near_pages=32, tli_near_tokens="off", tli_near_len=4096,
        tli_boundary_bypass=True, tli_l2_quant=False, tli_proj_basis=None,
        tli_static_pair=False, tli_l2_subspace="same", tli_subspace="tail",
    )
    args.head_dim_for_test = head_dim
    return args


def make_indexer(head_dim=256, args=None):
    from sparse_attn.indexer.tli_indexer import TLIIndexer

    idx = TLIIndexer(make_prod_args(head_dim) if args is None else args)
    idx.layer_idx = 2
    idx.head_dim = head_dim
    idx.clear()
    return idx


def randn(shape, device, seed, dtype=torch.bfloat16):
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(shape, device=device, dtype=dtype, generator=gen)


def dense_reference(q, k, v, mask, scale):
    """独立 Torch dense 参考；支持共享 KV mask 与 per-q-head mask。"""
    h = q.shape[2]
    hkv = k.shape[2]
    group = h // hkv
    qg = (q[0, 0].float() * scale).view(hkv, group, -1)
    scores = torch.einsum("hgd,thd->hgt", qg, k[0].float())
    selected = mask[0, 0]
    if selected.shape[0] == h and h != hkv:
        selected = selected.view(hkv, group, -1)
    else:
        selected = selected[:, None, :]
    scores = scores.masked_fill(~selected, float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(q.dtype)
    out = torch.einsum("hgt,thd->hgd", probs, v[0]).reshape(h, v.shape[-1])
    return out.view(1, 1, h, v.shape[-1])


def row_variable_reference(q, k_full, v_full, length, scale):
    """DynamicCache 单序列准备口径：有效前缀、无 valid_length。"""
    from sparse_attn.ops.eager_decoding import eager_decoding_attn

    idx = make_indexer(q.shape[-1])
    k = k_full[:, :length]
    v = v_full[:, :length]
    cu = torch.tensor([0, length], device=q.device, dtype=torch.int32)
    pos = torch.tensor([length - 1], device=q.device, dtype=torch.long)
    mask, block_size = idx.prepare_mask(q, pos, k, cu, softmax_scale=scale)
    return eager_decoding_attn(q, k, v, mask, block_size, cu, softmax_scale=scale)


def batched_decode(q, k, v, lengths):
    from sparse_attn.baseline_runtime import BatchCtx
    from sparse_attn.patches.qwen3_attn_patch import _tl_decode_batched

    idx = make_indexer(q.shape[-1])
    ctx = BatchCtx(batch=len(lengths), seq_lens=list(lengths),
                   gen_lens=[0] * len(lengths), alive=[True] * len(lengths), pad_id=0)
    shim = SimpleNamespace(scaling=q.shape[-1] ** -0.5)
    return _tl_decode_batched(shim, idx, q.transpose(1, 2), k.transpose(1, 2),
                              v.transpose(1, 2), ctx)


def test_tail_indices():
    dev = torch.device("cuda:0")
    expected = {
        128: list(range(48, 64)) + list(range(112, 128)),
        256: list(range(208, 224)) + list(range(240, 256)),
    }
    for dim, want in expected.items():
        idx = make_indexer(dim)
        ref = torch.tensor(want, device=dev, dtype=torch.long)
        l1 = idx._subspace_indices(dev)
        l2 = idx._qat_indices(dev)
        check(f"tail_mapping_D{dim}", torch.equal(l1, ref) and torch.equal(l2, ref),
              f"L1={l1.tolist()} L2={l2.tolist()}")


def test_full_incremental_reset():
    dev = torch.device("cuda:0")
    lengths = [61, 62, 65, 129, 131, 196, 67, 73]
    reset_before = {3, 7}
    keys = ("k_min", "k_max", "k_avg", "k_qat",
            "cu_seqlens_k_coarse", "cu_seqlens_k_fine")
    for dim in (128, 256):
        k = randn((1, max(lengths), 5, dim), dev, 20261028 + dim)
        inc = make_indexer(dim)
        ok = True
        details = []
        for step, length in enumerate(lengths):
            if step in reset_before:
                inc.clear()
            cur_k = k[:, :length]
            cu = torch.tensor([0, length], device=dev, dtype=torch.int32)
            got = inc.prepare_index(cur_k, cu)
            ref_idx = make_indexer(dim)
            ref = ref_idx.prepare_index(cur_k, cu)
            bad = []
            for key in keys:
                a, b = got.get(key), ref.get(key)
                if (a is None) != (b is None):
                    bad.append(key + ":presence")
                elif a is not None and not torch.equal(a, b):
                    nd = int((a != b).sum().item())
                    bad.append(f"{key}:nd={nd}")
            if bad:
                ok = False
                details.append(f"step={step},S={length}:" + ",".join(bad))
        check(f"full_inc_reset_D{dim}", ok,
              "8 步（非对齐/跨块/多 token/clear/S 回退）逐位"
              + ("； " + " | ".join(details) if details else ""))


def test_eager_dense_and_full_budget():
    dev = torch.device("cuda:0")
    h, hkv, dim = 20, 5, 256
    scale = dim ** -0.5
    for length in (4099, 16384):
        q = randn((1, 1, h, dim), dev, 31000 + length)
        k = randn((1, length, hkv, dim), dev, 32000 + length)
        v = randn((1, length, hkv, dim), dev, 33000 + length)
        mask = torch.rand((1, 1, hkv, length), device=dev,
                          generator=torch.Generator(device=dev).manual_seed(34000 + length)) > 0.98
        mask[..., -128:] = True
        cu = torch.tensor([0, length], device=dev, dtype=torch.int32)
        from sparse_attn.ops.eager_decoding import eager_decoding_attn
        out = eager_decoding_attn(q, k, v, mask, 1, cu, softmax_scale=scale)
        ref = dense_reference(q, k, v, mask, scale)
        close = torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
        diff = (out.float() - ref.float()).abs().max().item()
        check(f"eager_dense_GQA_D256_S{length}", close,
              f"H/Hkv={h}/{hkv} max_abs={diff:.4g} atol=rtol=1e-2")

    length = 323
    q = randn((1, 1, h, dim), dev, 35001)
    k = randn((1, length, hkv, dim), dev, 35002)
    v = randn((1, length, hkv, dim), dev, 35003)
    args = make_prod_args(dim)
    args.tia_level1_topk = 999999
    args.tia_level2_topk = 999999
    # sink 同时参与 mid 竞争与强制区会重复计额；独立 dense 锚关闭 sink。
    args.tli_sink_tokens = 0
    idx = make_indexer(dim, args)
    cu = torch.tensor([0, length], device=dev, dtype=torch.int32)
    pos = torch.tensor([length - 1], device=dev, dtype=torch.long)
    mask, block_size = idx.prepare_mask(q, pos, k, cu, softmax_scale=scale)
    from sparse_attn.ops.eager_decoding import eager_decoding_attn
    out = eager_decoding_attn(q, k, v, mask, block_size, cu, softmax_scale=scale)
    ref = dense_reference(q, k, v, torch.ones_like(mask), scale)
    close = torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
    check("full_budget_select_all_D256", bool(mask.all()) and close,
          f"mask_all={bool(mask.all())} max_abs={(out.float() - ref.float()).abs().max().item():.4g}")


def test_batch_fixed_and_dynamic_equivalence():
    dev = torch.device("cuda:0")
    h, hkv, dim = 20, 5, 256
    scale = dim ** -0.5
    cap = 225
    lengths = [61, 130]
    k = randn((2, cap, hkv, dim), dev, 41001)
    v = randn((2, cap, hkv, dim), dev, 41002)
    for row, length in enumerate(lengths):
        k[row, length:] = 1000.0
        v[row, length:] = -1000.0
    work_k, work_v = k.clone(), v.clone()
    ok = True
    details = []
    for step in range(3):
        new_lengths = [length + step for length in lengths]
        if step:
            for row, pos in enumerate(new_lengths):
                work_k[row, pos - 1] = randn((hkv, dim), dev, 41100 + step * 10 + row)
                work_v[row, pos - 1] = randn((hkv, dim), dev, 41200 + step * 10 + row)
        q = randn((2, 1, h, dim), dev, 41300 + step)
        got = batched_decode(q, work_k, work_v, new_lengths)
        refs = [row_variable_reference(q[row:row + 1], work_k[row:row + 1],
                                       work_v[row:row + 1], length, scale)
                for row, length in enumerate(new_lengths)]
        ref = torch.cat(refs, dim=0)
        if not torch.equal(got, ref):
            ok = False
            details.append(f"step={step},nd={int((got != ref).sum().item())}")
    check("B2_fixed_mixed_3step_D256", ok,
          "逐行 DynamicCache 参考逐位；pad=±1000" + ("； " + " | ".join(details) if details else ""))

    length = 61
    q = randn((1, 1, h, dim), dev, 42001)
    got = batched_decode(q, k[:1], v[:1], [length])
    ref = row_variable_reference(q, k[:1], v[:1], length, scale)
    check("B1_fixed_vs_DynamicCache_D256", torch.equal(got, ref),
          f"nd={int((got != ref).sum().item())}")

    idx = make_indexer(dim)
    cu = torch.tensor([0, cap], device=dev, dtype=torch.int32)
    pos = torch.tensor([length - 1], device=dev, dtype=torch.long)
    mask, _ = idx.prepare_mask(q, pos, k[:1], cu, softmax_scale=scale,
                               valid_length=length)
    pad_clear = not bool(mask[..., length:].any())
    check("fixed_pad_all_false_D256", pad_clear,
          f"pad_selected={int(mask[..., length:].sum().item())}")


def mask_graph_for_dim(dim):
    dev = torch.device("cuda:0")
    h, hkv = 20, 5
    scale = dim ** -0.5
    cap = 8192
    k = randn((1, cap, hkv, dim), dev, 51001 + dim)
    q = randn((1, 1, h, dim), dev, 51002 + dim)
    cu = torch.tensor([0, cap], device=dev, dtype=torch.int32)
    for length in (64, 132, 3000, 3201, 4096, 4352, 8191, 8192):
        cur_k = k.clone()
        cur_k[:, length:] = 0
        pos = torch.tensor([length - 1], device=dev, dtype=torch.long)
        eager_idx = make_indexer(dim)
        mask_e, _ = eager_idx.prepare_mask(q, pos, cur_k, cu, softmax_scale=scale,
                                           valid_length=length)
        graph_idx = make_indexer(dim)
        index_dict = graph_idx.prepare_index(cur_k, cu)
        mask_g = graph_idx.compute_mask_graph(
            q, index_dict, scale, torch.tensor([length], device=dev, dtype=torch.long))
        same = mask_e.shape == mask_g.shape and torch.equal(mask_e, mask_g)
        diff = int((mask_e != mask_g).sum().item()) if mask_e.shape == mask_g.shape else -1
        check(f"mask_graph_equiv_D{dim}_S{length}", same,
              f"diff={diff} eager={int(mask_e.sum())} graph={int(mask_g.sum())}")


def test_mask_graph_d256():
    mask_graph_for_dim(256)
    mask_graph_for_dim(128)


def graph_capture_for_dim(dim):
    from sparse_attn.ops.cuda_graph_state import disable_graph, enable_graph, graph_state
    from sparse_attn.patches.qwen3_attn_patch import _tl_decode_batched_graph

    dev = torch.device("cuda:0")
    h, hkv = 20, 5
    cap = 16512
    scale = dim ** -0.5
    q = randn((1, 1, h, dim), dev, 61000 + dim)
    k = randn((1, cap, hkv, dim), dev, 62000 + dim)
    v = randn((1, cap, hkv, dim), dev, 63000 + dim)
    shim = SimpleNamespace(scaling=scale)
    idx = make_indexer(dim)
    enable_graph(1)
    try:
        gs = graph_state()
        gs.valid_len_buf = torch.tensor([cap], device=dev, dtype=torch.long)

        def run(indexer):
            return _tl_decode_batched_graph(shim, indexer, q, k, v, gs)

        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(2):
                run(idx)
        torch.cuda.current_stream().wait_stream(warm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = run(idx)
        ok = True
        details = []
        for length in (cap, 16000, 4000, cap):
            gs.valid_len_buf.fill_(length)
            graph.replay()
            got = graph_out.clone()
            eager_idx = make_indexer(dim)
            ref = run(eager_idx)
            same = torch.equal(got, ref)
            ok &= same
            details.append(f"valid={length},nd={int((got != ref).sum().item())}")
        check(f"graph_capture_replay_D{dim}", ok,
              "capture/replay == eager graph datapath 逐位；" + " | ".join(details))
    finally:
        disable_graph()


def test_graph_capture_replay():
    graph_capture_for_dim(256)
    graph_capture_for_dim(128)


def small_glm_attention(device, production=False, args=None):
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
        Glm4MoeLiteAttention, Glm4MoeLiteConfig)
    from sparse_attn.patches.glm4_moe_lite_attn_patch import glm4_moe_lite_attn_forward

    heads = 20 if production else 4
    config = Glm4MoeLiteConfig(
        hidden_size=16, num_hidden_layers=47 if production else 1,
        num_attention_heads=heads, num_key_value_heads=heads,
        q_lora_rank=None, kv_lora_rank=16, qk_nope_head_dim=192,
        qk_rope_head_dim=64, qk_head_dim=256, v_head_dim=256,
        rope_interleave=True, attention_dropout=0.0,
        _attn_implementation="eager",
    )
    attn = Glm4MoeLiteAttention(config, 0).to(device=device, dtype=torch.bfloat16).eval()
    attn.indexer = make_indexer(256, args)
    attn.forward = MethodType(glm4_moe_lite_attn_forward, attn)
    return attn


def expect_raises(name, fn, text=None):
    try:
        fn()
    except (RuntimeError, ValueError) as exc:
        matched = text is None or text in str(exc)
        check(name, matched, f"{type(exc).__name__}: {exc}")
        return
    except Exception as exc:
        check(name, False, f"异常类型不符 {type(exc).__name__}: {exc}")
        return
    check(name, False, "未拒绝非法输入")


def test_dynamic_far_skip():
    from transformers import DynamicCache
    from sparse_attn.baseline_runtime import BatchCtx
    from sparse_attn.patches.qwen3_attn_patch import _tl_decode_batched

    dev = torch.device("cuda:0")
    args = make_prod_args(256)
    args.tli_dyn_dgate = True
    args.tli_dyn_dgate_eps = 1.0
    args.tli_dyn_dgate_rows = 8
    attn = small_glm_attention(dev, args=args)
    length = 4352
    hidden = randn((1, length, attn.config.hidden_size), dev, 72501)
    angles = randn((1, length, 64), dev, 72502, dtype=torch.float32) * 0.01
    positions = angles.cos().to(torch.bfloat16), angles.sin().to(torch.bfloat16)
    cache = DynamicCache()
    with torch.no_grad():
        attn(hidden, positions, None, past_key_values=cache,
             cache_position=torch.arange(length, device=dev))
    idx = attn.indexer
    calibrated = idx._dyn_skip_pending and not idx._dyn_frozen and not idx.skip_far
    seen = []
    original_prepare = idx.prepare_mask

    def checked_prepare(*call_args, **call_kwargs):
        seen.append(bool(idx.skip_far and idx._dyn_frozen))
        return original_prepare(*call_args, **call_kwargs)

    idx.prepare_mask = checked_prepare
    next_hidden = randn((1, 1, attn.config.hidden_size), dev, 72503)
    next_angles = randn((1, 1, 64), dev, 72504, dtype=torch.float32) * 0.01
    with torch.no_grad():
        attn(next_hidden,
             (next_angles.cos().to(torch.bfloat16),
              next_angles.sin().to(torch.bfloat16)),
             None, past_key_values=cache,
             cache_position=torch.tensor([length], device=dev))
    check("GLM_dynamic_far_skip_calibrate_freeze",
          calibrated and seen == [True],
          f"pending={idx._dyn_skip_pending} frozen={idx._dyn_frozen} decode_seen={seen}")

    batch_idx = make_indexer(256, args)
    batch_idx._dyn_skip_pending = True
    q = randn((1, 4, 1, 256), dev, 72505)
    k = randn((1, 4, length + 1, 256), dev, 72506)
    v = randn((1, 4, length + 1, 256), dev, 72507)
    batch_seen = []
    original_batch_prepare = batch_idx.prepare_mask

    def checked_batch_prepare(*call_args, **call_kwargs):
        batch_seen.append(bool(batch_idx.skip_far and batch_idx._dyn_frozen))
        return original_batch_prepare(*call_args, **call_kwargs)

    batch_idx.prepare_mask = checked_batch_prepare
    ctx = BatchCtx(batch=1, seq_lens=[length + 1], gen_lens=[0],
                   alive=[True], pad_id=0)
    shim = SimpleNamespace(scaling=256 ** -0.5)
    _tl_decode_batched(shim, batch_idx, q, k, v, ctx)
    check("GLM_dynamic_far_skip_batch_clear_preserves_gate",
          batch_seen == [True], f"decode_seen={batch_seen}")


def test_glm_forward_fail_closed():
    from transformers import DynamicCache

    dev = torch.device("cuda:0")
    attn = small_glm_attention(dev)
    heads, dim = attn.num_heads, attn.qk_head_dim

    def hidden(batch, length, seed):
        return randn((batch, length, attn.config.hidden_size), dev, seed)

    def positions(batch, length, seed):
        angles = randn((batch, length, 64), dev, seed, dtype=torch.float32) * 0.01
        return angles.cos().to(torch.bfloat16), angles.sin().to(torch.bfloat16)

    def cache_with_history(batch, length=3):
        cache = DynamicCache()
        key = randn((batch, heads, length, dim), dev, 71001 + batch + length)
        value = randn((batch, heads, length, dim), dev, 72001 + batch + length)
        cache.update(key, value, 0)
        return cache

    def invalid_mask():
        cache = cache_with_history(1)
        mask = torch.ones((1, 4), device=dev, dtype=torch.bool)
        mask[0, -1] = False
        attn(hidden(1, 1, 73001), positions(1, 1, 73002), mask,
             past_key_values=cache, cache_position=torch.tensor([3], device=dev))

    def chunked_prefill():
        attn(hidden(1, 2, 73101), positions(1, 2, 73102), None,
             past_key_values=cache_with_history(1),
             cache_position=torch.tensor([3, 4], device=dev))

    def batch_without_ctx():
        attn(hidden(2, 1, 73201), positions(2, 1, 73202), None,
             past_key_values=cache_with_history(2),
             cache_position=torch.tensor([3, 3], device=dev))

    def no_cache_decode():
        attn(hidden(1, 1, 73211), positions(1, 1, 73212), None,
             past_key_values=None, cache_position=torch.tensor([3], device=dev))

    def empty_cache_decode():
        attn(hidden(1, 1, 73221), positions(1, 1, 73222), None,
             past_key_values=DynamicCache(),
             cache_position=torch.tensor([3], device=dev))

    expect_raises("GLM_invalid_mask_fail_closed", invalid_mask, "attention_mask")
    expect_raises("GLM_chunked_prefill_fail_closed", chunked_prefill, "chunked prefill")
    expect_raises("GLM_B2_without_BatchCtx_fail_closed", batch_without_ctx, "BatchCtx")
    expect_raises("GLM_no_cache_decode_fail_closed", no_cache_decode, "KV cache")
    expect_raises("GLM_empty_cache_nonzero_position_fail_closed",
                  empty_cache_decode, "空 cache")

    args = make_prod_args(256)
    args.tli_subspace = "full"
    bad_idx = make_indexer(256, args)
    bad_k = randn((1, 65, 5, 256), dev, 73301)
    bad_cu = torch.tensor([0, 65], device=dev, dtype=torch.int32)
    expect_raises("D256_nonproduction_indexer_fail_closed",
                  lambda: bad_idx.prepare_index(bad_k, bad_cu), "D=256")


def test_multigpu_decode_device_context():
    from transformers import DynamicCache

    if torch.cuda.device_count() < 2:
        check("GLM_multigpu_D256_S131073_device_context", False, "需要两张 CUDA GPU")
        return
    dev = torch.device("cuda:1")
    attn = small_glm_attention(dev)
    length = 131072
    cache = DynamicCache()
    key = randn((1, attn.num_heads, length, attn.qk_head_dim), dev, 74001)
    value = randn((1, attn.num_heads, length, attn.v_head_dim), dev, 74002)
    cache.update(key, value, 0)
    hidden = randn((1, 1, attn.config.hidden_size), dev, 74003)
    angles = randn((1, 1, 64), dev, 74004, dtype=torch.float32) * 0.01
    positions = angles.cos().to(torch.bfloat16), angles.sin().to(torch.bfloat16)
    torch.cuda.set_device(0)
    out, _ = attn(hidden, positions, None, past_key_values=cache,
                  cache_position=torch.tensor([length], device=dev))
    torch.cuda.synchronize(dev)
    ok = out.device == dev and torch.cuda.current_device() == 0
    check("GLM_multigpu_D256_S131073_device_context", ok,
          f"out_device={out.device} current_device={torch.cuda.current_device()}")


def registration_model(device_map=None):
    import torch.nn as nn
    attn = small_glm_attention(torch.device("cpu"), production=True)
    model = nn.Module()
    model.config = attn.config
    model.layers = nn.ModuleList([
        attn if layer == 0 else small_glm_attention(torch.device("cpu"), production=True)
        for layer in range(47)
    ])
    if device_map is not None:
        model.hf_device_map = device_map
    return model


def test_registration_fail_closed():
    from sparse_attn.patches.patch import register_patch

    args = make_prod_args(256)
    args.tli_subspace = "full"
    model = registration_model()
    expect_raises("GLM_nonproduction_registration_fail_closed",
                  lambda: register_patch(model, args), "D=256")
    del model

    model = registration_model()
    args = make_prod_args(256)
    args.tli_dyn_dgate = True
    n_patched = register_patch(model, args)
    check("GLM_dynamic_far_skip_registration_47_layers",
          n_patched == 47 and all(
              layer.indexer.dyn_dgate and layer.indexer.head_dim == 256
              for layer in model.layers), f"n_patched={n_patched}")
    del model

    old = os.environ.get("TLI_CUDA_GRAPH")
    os.environ["TLI_CUDA_GRAPH"] = "1"
    try:
        model = registration_model({"layers.0": 0, "layers.1": "cuda:1"})
        expect_raises("GLM_multigpu_graph_prescan_fail_closed",
                      lambda: register_patch(model, make_prod_args(256)), "单卡")
        del model
    finally:
        if old is None:
            os.environ.pop("TLI_CUDA_GRAPH", None)
        else:
            os.environ["TLI_CUDA_GRAPH"] = old


GROUPS = {
    "tail": test_tail_indices,
    "index": test_full_incremental_reset,
    "eager": test_eager_dense_and_full_budget,
    "batch": test_batch_fixed_and_dynamic_equivalence,
    "mask_graph": test_mask_graph_d256,
    "far_skip": test_dynamic_far_skip,
    "forward": test_glm_forward_fail_closed,
    "multigpu": test_multigpu_decode_device_context,
    "registration": test_registration_fail_closed,
    "capture": test_graph_capture_replay,
}


def main():
    assert torch.cuda.is_available(), "需要 H20/CUDA GPU"
    import transformers
    if transformers.__version__ != "5.12.1":
        raise RuntimeError(
            "真实 GLM 结构门需要隔离 Transformers 5.12.1；请设置 "
            "PYTHONPATH=$HOME/.local/pylibs_tf512:$PWD，当前="
            f"{transformers.__version__} ({transformers.__file__})")
    torch.cuda.set_device(0)
    selected = sys.argv[1:] or list(GROUPS)
    unknown = set(selected) - set(GROUPS)
    if unknown:
        raise ValueError(f"未知门禁组: {sorted(unknown)}；可选 {list(GROUPS)}")
    for name in selected:
        run_group(name, GROUPS[name])
    print(f"\n=== PASS {len(PASS)} / FAIL {len(FAIL)} ===", flush=True)
    for name in FAIL:
        print(f"  FAIL: {name}", flush=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
