#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 2c 门禁套件：TLI CUDA Graph decode 张量化前置（deepseek 轨）。

用法（仓根）：
  CUDA_VISIBLE_DEVICES=0 python3 exp/bench/test_tli_graph_gates.py

背景：阶段 2c 把 TLI batch decode 全链路张量化——decode 步间变化的数据
（KV 写入位置 / valid_length / mask 边界）全部由 tensor 内容承载，消除
capture 时烘焙的 Python int launch 参数（切片边界/topk k/arange 起点）在
replay 时的陈旧问题 → 整步 forward 可 capture 一次 + replay 循环 copy_ 注入。

本套件先固守最底层的 KV 写入张量化（TLIStaticCache.update 的
index_copy_ 单 kernel vs 旧逐行 copy_），后续 2c 子门禁（compute_score /
compute_mask 张量化对拍、graph vs eager 整步对拍、wall）追加于此。

门禁（单元级——不加载模型，合成 KV 直测）：
  G7a kv_update_tensor: TLIStaticCache.update 张量化（index_copy_）==
     参考逐行 copy_ 逐位（多步 decode 含 dead 行冻结语义）。
  G7b kv_update_graph: graph 模式（pos_buf 每步 copy_ 注入）== 非 graph
     模式（Python 构造 pos）逐位。
  G8 mask_graph_equiv: compute_mask_graph（容量宽全张量，graph 专用路径）
     == eager prepare_mask（变长参考臂）多 S_real 值 mask 逐位对拍——
     生产主配置（nb=32/nt=off/nl=4096/γ=None/boundary_bypass/sink4/swa128）。
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import torch  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"[PASS] {name} {detail}")
    else:
        FAIL.append(name)
        print(f"[FAIL] {name} {detail}")


def _ref_layer_update(layer, key_states, value_states, seq_lens, alive):
    """Independent scalar reference: finished rows preserve their KV bytes."""
    keys, values = layer.keys, layer.values
    for b in range(layer.max_batch_size):
        if not alive[b]:
            continue
        p = max(0, min(seq_lens[b] - 1, layer.max_cache_len - 1))
        keys[b, :, p] = key_states[b, :, 0]
        values[b, :, p] = value_states[b, :, 0]
    return keys, values


def _kv_step(g, B, Hkv, D, dev):
    """本步新 token 的 key/value [B, Hkv, 1, D]（固定 seed 流）。"""
    return (torch.randn(B, Hkv, 1, D, generator=g, dtype=torch.bfloat16,
                        device=dev),
            torch.randn(B, Hkv, 1, D, generator=g, dtype=torch.bfloat16,
                        device=dev))


def _step_seq_lens(seq_lens, alive):
    """forward 前 alive 行 seq_lens +1（dead 行冻结，与 _decode_batch_chunk
    逐行语义一致）。"""
    for b in range(len(seq_lens)):
        if alive[b]:
            seq_lens[b] += 1
    return list(seq_lens)


def test_kv_update_tensor():
    """G7a：update 张量化（index_copy_）== 参考逐行 copy_ 逐位（多步含 dead）。"""
    from types import SimpleNamespace
    from sparse_attn.ops.tli_static_cache import TLIStaticCache
    from sparse_attn.baseline_runtime import (BatchCtx, set_batch_decode_ctx,
                                               clear_batch_decode_ctx)

    dev = torch.device('cuda:0')
    Hkv, D = 8, 128
    B, S_cap = 4, 80
    cfg = SimpleNamespace(num_hidden_layers=2,
                          layer_types=["full_attention", "full_attention"])
    seed = 20261020
    g = torch.Generator(device=dev).manual_seed(seed)

    seq_lens = [16, 32, 48, 64]
    alive = [True, True, True, True]
    cache_a = TLIStaticCache(cfg, S_cap, B)     # 张量化臂（调 update）
    cache_b = TLIStaticCache(cfg, S_cap, B)     # 参考逐行臂（不调 update）
    # 参考臂只首次 lazy_initialization（生产 update 在 keys is None 时一次
    # 分配；该函数每次调用都重新 zero 分配，重复调用会清掉前步写入）。
    ref_inited = [False] * len(cache_b.layers)
    ok = True
    for step in range(4):
        seq_lens = _step_seq_lens(seq_lens, alive)
        key_states, value_states = _kv_step(g, B, Hkv, D, dev)
        key_states = key_states.contiguous()    # [B, Hkv, 1, D]
        value_states = value_states.contiguous()
        ctx = BatchCtx(batch=B, seq_lens=list(seq_lens),
                       gen_lens=[0] * B, alive=list(alive), pad_id=0)
        set_batch_decode_ctx(ctx)
        # 张量化臂：逐层调 TLIStaticLayer.update（Cache 基类分派的生产
        # 形态是 cache.update(k, v, layer_idx, cache_kwargs) → 本层 update）。
        for l in range(len(cache_a.layers)):
            cache_a.layers[l].update(key_states, value_states)
        clear_batch_decode_ctx()
        for l in range(len(cache_b.layers)):
            if not ref_inited[l]:
                cache_b.layers[l].lazy_initialization(key_states)
                ref_inited[l] = True
            _ref_layer_update(cache_b.layers[l], key_states, value_states,
                              seq_lens, alive)
        for l in range(len(cache_a.layers)):
            ka, va = cache_a.layers[l].keys, cache_a.layers[l].values
            kb, vb = cache_b.layers[l].keys, cache_b.layers[l].values
            if not (torch.equal(ka, kb) and torch.equal(va, vb)):
                ok = False
        if step == 0:
            alive[2] = False                    # row2 触发 EOS 停止
        elif step == 1:
            alive[0] = False                    # row0 触发 EOS 停止
    clear_batch_decode_ctx()
    check("G7a kv_update_tensor: index_copy_ == 逐行 copy_ 逐位(4步含dead)",
          ok)


def test_kv_update_graph():
    """G7b：graph 模式（pos_buf 每步 copy_ 注入）== 非 graph（Python 构造）。

    同一输入序列（固定 seed 重放）分别跑非 graph 与 graph 两臂，独立 cache，
    每步 KV buffer 逐位对比——证明 pos_buf 注入路径与 Python 构造路径语义
    逐位等价（CUDA Graph 前置：replay 时 pos_buf 内容变化 → 写新列）。
    """
    from types import SimpleNamespace
    from sparse_attn.ops.tli_static_cache import TLIStaticCache
    from sparse_attn.ops.cuda_graph_state import (enable_graph, disable_graph,
                                                   graph_state)
    from sparse_attn.baseline_runtime import (BatchCtx, set_batch_decode_ctx,
                                               clear_batch_decode_ctx)

    dev = torch.device('cuda:0')
    Hkv, D = 8, 128
    B, S_cap = 4, 80
    cfg = SimpleNamespace(num_hidden_layers=2,
                          layer_types=["full_attention", "full_attention"])

    def run(graph_on):
        cache = TLIStaticCache(cfg, S_cap, B)
        g = torch.Generator(device=dev).manual_seed(777)
        seq_lens = [16, 32, 48, 64]
        alive = [True, True, True, True]
        snapshots = []
        if graph_on:
            pos_buf = torch.zeros(B, dtype=torch.long, device=dev)
            enable_graph(B)
            graph_state().pos_buf = pos_buf
            graph_state().alive_buf = torch.ones(B, dtype=torch.bool, device=dev)
        for step in range(4):
            seq_lens = _step_seq_lens(seq_lens, alive)
            key_states, value_states = _kv_step(g, B, Hkv, D, dev)
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
            ctx = BatchCtx(batch=B, seq_lens=list(seq_lens),
                           gen_lens=[0] * B, alive=list(alive), pad_id=0)
            set_batch_decode_ctx(ctx)
            if graph_on:
                graph_state().alive_buf.copy_(torch.tensor(alive, device=dev))
                graph_state().pos_buf.copy_(torch.tensor(
                    [seq_lens[b] - 1 for b in range(B)],
                    dtype=torch.long, device=dev))
            for l in range(len(cache.layers)):
                cache.layers[l].update(key_states, value_states)
            clear_batch_decode_ctx()
            snapshots.append((cache.layers[0].keys.clone(),
                              cache.layers[0].values.clone()))
            if step == 0:
                alive[2] = False
            elif step == 1:
                alive[0] = False
        if graph_on:
            disable_graph()
        return snapshots

    ng = run(False)
    gr = run(True)
    ok = len(ng) == len(gr) and all(
        torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        for a, b in zip(ng, gr))
    check("G7b kv_update_graph: graph(pos_buf 注入)==非 graph(Python 构造)",
          ok)


def _make_prod_args(subspace='tail'):
    """生产主配置（pred_ruler.py 派单口径，deepseek 轨 2c 对拍基准）。

    nb=32/nt=off/nl=4096（固定配额几何，α/β/γ 禁传 → 归一 alpha=beta=1.0、
    gamma=None）+ far=minmax/near=avg + boundary_bypass + sink4/swa128/
    bs64/K1=128/K2=1024 + subspace=tail（生产；'full' 为 G8 的全维回退臂）
    + kmeans 关 + layer_skip 关。
    """
    import argparse
    args = argparse.Namespace(
        method='tli', tia_block_size=64, tia_level1_topk=128, tia_level2_topk=1024,
        tia_level2_cmp_ratio=4, tia_enable_async_topk=False,
        tli_enable_subspace=True, tli_enable_kmeans=False,
        tli_far_select='4bit', tli_near_select='4bit',
        tli_far_clusters=256, tli_far_niter=10, tli_far_blocks=16, tli_far_tokens=512,
        tli_far_method='minmax', tli_near_method='avg',
        tli_sim=0.9, tli_sim_dims='subspace',
        tli_sigma_select='none', tli_moba=False, tli_sigma=8.0,
        tli_sink_tokens=4, tli_swa_size=128,
        tli_sinkswa_only=False, tli_dyn_dgate=False, tli_dyn_dgate_eps=0.05,
        tli_dyn_dgate_rows=256, tli_per_q_head=False,
        tli_enable_layer_skip=False, tli_layer_skip_path=None,
        tli_alpha=None, tli_beta=None, tli_gamma=None,
        tli_near_pages=32, tli_near_tokens='off', tli_near_len=4096,
        tli_boundary_bypass=True, tli_l2_quant=False,
        tli_proj_basis=None, tli_static_pair=False, tli_l2_subspace='same',
    )
    args.tli_subspace = subspace
    return args


def test_mask_graph_equiv():
    """G8：compute_mask_graph（容量宽全张量）== eager prepare_mask（变长参考臂）。

    同一 k 前缀（固定 seed，[0,S_real) 有效、[S_real,S_cap) pad 0——TLIStaticCache
    left-pad 语义）分别走两路径：
      - eager：prepare_mask(..., valid_length=S_real)，k 内部切片 + 块统计
        pad 0 到块界 + softmax 归约有效区 + mask 末尾 pad False 回全宽；
      - graph：prepare_index 容量宽全量一次烘焙 + compute_mask_graph，动态几何
        全部从 valid_len tensor 派生。
    断言两 mask [1,1,Hkv,S_cap] bool 逐位 ==。

    测试点覆盖几何边界：
      - 非对齐（%64≠0）：3000（swa 跨界页 %64=56 触发）、3201（%64=1）；
      - 对齐：4096（%64=0，无跨界页）；
      - 短序列：64（swa_lo_tok=0，S_real<swa）、132（swa_lo_tok=4，恰 sink+swa）；
      - near 边界：4352（S_real−swa−nl=128 → near_blks=2）；
      - 满容量/近满：8192（S_real==S_cap 无 pad）、8191（满容量非对齐）。
    两套 subspace 同步覆盖：tail（生产主配置，Q/K 均 32 维子空间）与 full
    （idx_sub=None 全维粗筛 + L2 尾段 32 维，走 `_qat_indices` 回退臂）。
    """
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    dev = torch.device('cuda:0')
    H, Hkv, D = 32, 8, 128
    scale = D ** -0.5
    S_cap = 8192
    SEED = 20261021
    cases = [3000, 3201, 4096, 64, 132, 4352, 8192, 8191]
    all_ok = True
    detail = []
    for subspace in ('tail', 'full'):
        args = _make_prod_args(subspace)
        for S_real in cases:
            assert 0 < S_real <= S_cap
            g = torch.Generator(device=dev).manual_seed(SEED)
            k_full = torch.randn(1, S_cap, Hkv, D, device=dev, dtype=torch.bfloat16,
                                 generator=g)
            k_full[0, S_real:] = 0.0                  # left-pad 未写区
            q = torch.randn(1, 1, H, D, device=dev, dtype=torch.bfloat16,
                            generator=torch.Generator(device=dev).manual_seed(SEED + 1))
            cu_full = torch.tensor([0, S_cap], device=dev, dtype=torch.int32)
            pos = torch.tensor([S_real - 1], device=dev, dtype=torch.long)
            # eager 参考臂（变长切片 + pad False 回全宽）
            idx_e = TLIIndexer(args); idx_e.layer_idx = 2; idx_e.clear()
            mask_e, bs_e = idx_e.prepare_mask(
                q, pos, k_full, cu_full, softmax_scale=scale, valid_length=S_real)
            # graph 臂（容量宽 index_dict 一次烘焙 + compute_mask_graph）
            idx_g = TLIIndexer(args); idx_g.layer_idx = 2; idx_g.clear()
            index_dict_cap = idx_g.prepare_index(k_full, cu_full)
            sel_g = idx_g.compute_mask_graph(
                q, index_dict_cap, scale, torch.tensor([S_real], device=dev))
            eq = (mask_e.shape == sel_g.shape
                  and bool((mask_e == sel_g).all().item()))
            if not eq:
                n_diff = int((mask_e != sel_g).sum())
                n_e = int(mask_e.sum()); n_g = int(sel_g.sum())
                detail.append(f"[{subspace}] S={S_real} "
                              f"shape {tuple(mask_e.shape)} vs "
                              f"{tuple(sel_g.shape)} diff={n_diff} "
                              f"(e={n_e}, g={n_g})")
            all_ok &= eq
            print(f"  [{'PASS' if eq else 'FAIL'}] subspace={subspace} "
                  f"S_real={S_real} bs_e={bs_e} "
                  f"n_sel_e={int(mask_e.sum().item())} "
                  f"n_sel_g={int(sel_g.sum().item())}", flush=True)
    check("G8 mask_graph_equiv: compute_mask_graph==prepare_mask(变长) "
          "多 S_real×tail/full 逐位", all_ok, "; ".join(detail))


def test_graph_datapath_equiv():
    """G9：_tl_decode_batched_graph（2c-4 tensor 几何）== staged 逐行路径。

    同一容量宽 KV / 同 q，B 行不同有效长（left-pad 有效在前）：
      - staged：prepare_mask(valid_length=L_b Python int) 逐行；
      - graph：compute_mask_graph(valid_len_buf tensor) 逐行。
    断言 attention 输出 [B,1,H,D] 逐位 ==（mask 逐位等价由 G8 覆盖；
    T≥16K 走 Triton gather 批量，valid_length 不参与归约 → 输出可逐位）。
    """
    from types import SimpleNamespace
    import argparse
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    from sparse_attn.patches.qwen3_attn_patch import (
        _tl_decode_batched, _tl_decode_batched_graph)
    from sparse_attn.ops.cuda_graph_state import enable_graph, disable_graph, \
        graph_state
    from sparse_attn.baseline_runtime import BatchCtx

    dev = torch.device('cuda:0')
    H, Hkv, D = 32, 8, 128
    scale = D ** -0.5
    S_cap = 16512                     # ≥16384：Triton gather 批量路径
    lens = [16200, 16512, 4000]       # 行有效长（left-pad 有效在前）
    B = len(lens)
    SEED = 20261022
    g = torch.Generator(device=dev).manual_seed(SEED)
    k = torch.randn(B, S_cap, Hkv, D, device=dev, dtype=torch.bfloat16,
                    generator=g)
    v = torch.randn(B, S_cap, Hkv, D, device=dev, dtype=torch.bfloat16,
                    generator=g)
    for b, L in enumerate(lens):
        k[b, L:] = 0.0
        v[b, L:] = 0.0
    q = torch.randn(B, 1, H, D, device=dev, dtype=torch.bfloat16,
                    generator=torch.Generator(device=dev).manual_seed(SEED + 1))
    args = _make_prod_args('tail')
    shim = SimpleNamespace(scaling=scale)

    # staged 路径（B>1 批量：逐行 prepare_mask + 一次批量 eager）
    ctx = BatchCtx(batch=B, seq_lens=list(lens), gen_lens=[0] * B,
                   alive=[True] * B, pad_id=0)
    idx_s = TLIIndexer(args); idx_s.layer_idx = 2; idx_s.clear()
    o_staged = _tl_decode_batched(
        shim, idx_s, q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        ctx)
    # graph 数据通路（graph_state 使能但此处只做 eager 直跑，不 capture）
    idx_g = TLIIndexer(args); idx_g.layer_idx = 2; idx_g.clear()
    enable_graph(B)
    graph_state().valid_len_buf = torch.tensor(lens, dtype=torch.long,
                                               device=dev)
    o_graph = _tl_decode_batched_graph(
        shim, idx_g, q, k, v, graph_state())
    disable_graph()
    ok = (o_staged.shape == o_graph.shape
          and bool((o_staged == o_graph).all().item()))
    nd = int((o_staged != o_graph).sum())
    check("G9 graph_datapath_equiv: _tl_decode_batched_graph==staged 逐位(B=3)",
          ok, f"diff={nd}/{o_staged.numel()}")


def test_graph_capture_replay():
    """G10 capture/replay 硬门：整步 _tl_decode_batched_graph capture 一次，
    replay 时改 valid_len_buf 内容（模拟 decode 步间变化）→ 输出必须与本步
    参数下的 eager 直调逐位相同（证 capture 期没有任何 host 值被烘焙进图）。

    覆盖两组长度：同长（[16512,16512]，最长行无 pad）与变长
    （[16000,4000]，短行大量 pad 列）。逐位断言 [B,1,H,D]。
    """
    from types import SimpleNamespace
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    from sparse_attn.patches.qwen3_attn_patch import _tl_decode_batched_graph
    from sparse_attn.ops.cuda_graph_state import (enable_graph, disable_graph,
                                                   graph_state)

    dev = torch.device('cuda:0')
    H, Hkv, D = 32, 8, 128
    scale = D ** -0.5
    S_cap = 16512
    lens0 = [16200, 16512]
    B = len(lens0)
    SEED = 20261023
    g = torch.Generator(device=dev).manual_seed(SEED)
    k = torch.randn(B, S_cap, Hkv, D, device=dev, dtype=torch.bfloat16,
                    generator=g)
    v = torch.randn(B, S_cap, Hkv, D, device=dev, dtype=torch.bfloat16,
                    generator=g)
    for b, L in enumerate(lens0):
        k[b, L:] = 0.0
        v[b, L:] = 0.0
    q = torch.randn(B, 1, H, D, device=dev, dtype=torch.bfloat16,
                    generator=torch.Generator(device=dev).manual_seed(SEED + 1))
    args = _make_prod_args('tail')
    shim = SimpleNamespace(scaling=scale)

    vl_buf = torch.tensor(lens0, dtype=torch.long, device=dev)
    idx = TLIIndexer(args); idx.layer_idx = 2; idx.clear()
    enable_graph(B)
    gs = graph_state()
    gs.valid_len_buf = vl_buf

    def run():
        return _tl_decode_batched_graph(shim, idx, q, k, v, gs)

    # warmup（capture 外，side stream）+ capture
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(2):
            run()
    torch.cuda.current_stream().wait_stream(warm)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_g = run()

    ok, detail = True, []
    for newl in ([16512, 16512], [16000, 4000]):
        vl_buf.copy_(torch.tensor(newl, dtype=torch.long, device=dev))
        graph.replay()
        o_g = out_g.clone()
        # eager 参考：新 indexer，同一 vl 值（graph 使能但未 capture → 直跑）
        idx_e = TLIIndexer(args); idx_e.layer_idx = 2; idx_e.clear()
        o_e = _tl_decode_batched_graph(shim, idx_e, q, k, v, gs)
        nd = int((o_e != o_g).sum())
        eq = nd == 0
        ok &= eq
        detail.append(f"lens={newl} diff={nd}/{o_e.numel()}")
        print(f"  [{'PASS' if eq else 'FAIL'}] capture/replay lens={newl} "
              f"diff={nd}", flush=True)
    disable_graph()
    check("G10 graph_capture_replay: capture 一次 + vl 注入 replay == eager 逐位",
          ok, "; ".join(detail))


if __name__ == "__main__":
    assert torch.cuda.is_available(), "需要 GPU（KV update 索引写入）"
    dev = torch.device('cuda:0')
    torch.cuda.set_device(dev)
    test_kv_update_tensor()
    test_kv_update_graph()
    test_mask_graph_equiv()
    test_graph_datapath_equiv()
    test_graph_capture_replay()
    print(f"\n=== PASS {len(PASS)} / FAIL {len(FAIL)} ===")
    for f in FAIL:
        print(f"  FAIL: {f}")
    sys.exit(1 if FAIL else 0)
