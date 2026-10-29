# -*- coding: utf-8 -*-
"""【E139 纲领②③】稀疏 prefill 组选择批量化（用户 2026-10-14 四点纲领）。

纲领②：共享 K 索引 + 逐行批量选择——每层一次统计/量化（prepare_index 全前缀
冷建一次），组末行批量打分 + 批量两级 topk，替代逐组 ~496 次顺序三段式
（prepare_index→compute_score→compute_mask，每次 ~40 launch → ~20k
launch/层，是 grp64 981ms vs chunk-shared 168.6ms 的主因）。
纲领③：L1 之后才算候选内 L2——L2 细筛只在 L1 入围块（far 112 + near 16
= 128 块 × 64 token = 8192 候选/行）上算分数与 topk，不再全序列物化。

数学等价性（对拍硬门 e139_gang23_bitdiff.py 全门 PASS：mask 双档
L∈{8192,32768} 逐位一致率 1.000000 / diff_token=0 / 因果越界=0，输出级
allclose 且 maxdiff=0.000e+00，因果探针非跨界 chunk 零 diff + 同输入
逐位一致，回归 7/7+17/17+5/5）：
  ① 组末行 g1 恒为 bs 倍数（grp%bs==0 且 dense_below%bs==0 由调用方保证，
     未对齐组由调用方回退逐组路径）→ 行前缀统计 = 共享块统计（块内归约
     独立于前缀长度，逐位相同）；prepare_index(k_full) 与逐组链的 inc 缓存
     append-only 前进到 S=L 后的统计逐位一致，decode 兼容（inc 缓存留
     S=L，append-only 前进语义不变）。
  ② 打分数值树对齐（E139 核心发现）：逐组参考链走 Triton fast path
     （_fast_score_coarse/_fast_score_fine，fp32 累加顺序与 cublas 不同
     → bf16 舍入后 ULP 级分数差 → topk 边界翻转）——einsum/cublas 批量
     打分实测 ~0.45% 选择差、输出 maxdiff 1.8e-1（allclose 数学不可达）。
     唯一解法：逐行调与参考完全相同的 Triton kernel（kernel per-tile
     独立 grid=(cdiv(T,bt),Hkv) → 行级段切片/候选 gather 切片与参考
     全前缀调用的对应行逐位一致，切片不改变 tile 内数值树）。
  ③ L2 softmax/双池 topk 全宽逐位复刻：候选列 softmax 的归约树宽度差
     （g1 vs C=8192）与 topk 池序差（位置序 vs L1 topk rank 序）是仅剩
     的 ULP 翻转源（候选列版 mask 一致率 0.9990 过 mask 门但 0.1% 翻转
     使输出级 allclose(1e-2) 不可达）。终版：打分仍只在候选列（纲领③
     省算主体：k_qat 候选 gather + 同一 fine kernel），softmax/GQA mean/
     L2 双池 topk 按行以与参考完全相同的形状·数值·内存布局逐算子复刻
     （[1,1,H,g1] where(topk_mask,sf,-inf) 等价张量 → F.softmax →
     nan_to_num → view(Hkv,G,·).mean(-2) → far/near 位置切片 topk →
     scatter → sink/swa 强制）——同形同值确定性算子 → mask 逐位一致。
     softmax/mean/topk 是逐元素/归约廉价算子（O(H·g1) ≪ 打分
     O(H·C·δ)），复刻成本远小于打分省算收益。
  ④ 池边界/预算公式逐行向量化（纯整数算术零 host 同步），公式逐字段回
     tli_indexer.py 原码核对（L1880-1897 池边界 / L1945-1983 L1 双池 /
     L2126-2359 L2 双池预算）。
  ⑤ L1 段分型省算（主会话注入·acc23d564ad2eff91，已落实）：行 j 只在
     far 段 [far_lo, nb_j) 算 far_method 对应分（生产 mavg=minmax 上界
     分）、near 段 [nb_j, sl_j) 算 near_method 对应分（生产 mavg=avg
     精确分），swa/因果无效块不打分——行级段切片逐行调与参考链同一
     Triton fast kernel（_fast_score_coarse/_fast_score_fine），消费段
     分数与逐组版逐位一致（kernel per-tile 独立，切片不改 tile 内数值
     树），对拍口径不变。
  ⑥ L2 候选 scatter 的无效槽防护：L1 topk 的 -inf 补位块号是垃圾索引，
     不得散写 [0,g1)（可能命中有效候选位）——无效槽统一指向 token 0
     （sink 区 [0,sink) 恒非候选，-inf 幂等写入无覆盖）。历史教训：
     trash 列全宽 scatter 版的 False 直写会覆盖先前段已写的 True 选中
     （near 池补位块号落 far 段，组 0 far L2 选中被全部抹 False）。
  ⑦ 【#253 Step2】L2 双池 topk 候选列化（env TLI_PREFILL_L2_TOPK_CAND
     默认 0 fail-closed）：p 全宽计算（②③ 段）不变，④ 步 topk 从池位置
     切片 O(W) radix select 换成候选列 topk O(C)。p 值 gather 自同一全宽
     p 张量（零 ULP——区别于已证伪的候选列 softmax 归约树差 0.1% 翻转）。
     等价性实证 e139_step2_topk_cand_feasibility（0/320 全格）：
     (a) 候选按位置序（cand_blk sort）后 CUDA radix topk 相等值取低
     index ≡ 全宽位置序 tie 序；(b) k > cnt（池内候选数，仅大 K2/窄池
     档出现）时全宽零补位 = 池内非候选块 token 位置序前 (k-cnt) 个——
     _zero_pad_pool 补位仿真。cnt 只依赖池几何（sc -inf 模式头不变，
     与数据无关）→ nbad 是 host 常量分支，CUDA Graph capture 安全。
     已知残余边界：候选 p 值 bf16 下溢恰为 0.0 时与补位零的位置序混合
     （生产 score 范围内概率极低，bitdiff 硬门把关）。

fail-closed：任何配置不匹配（非 4bit 选择 / 非 e64 分区 / per_q_head /
σ / MoBA / sinkswa_only / async / 投影基 / l2=full / 非 tail 子空间 /
skip_far / D≠128 / 非 CUDA 等）返回 None，调用方回退逐组路径——宁可回退
不可错。

注意：本路径不调 metrics.add_select_result（逐组版每组一条 GPU 侧记录），
批量版跳过——metrics 仅诊断用途，不影响选择/输出正确性。
"""
import os

import torch
import torch.nn.functional as F

# Process-local cache; shared across equivalent layer instances on one CUDA stream.
# Callers must consume outputs on that stream before the next replay.
_GRAPH_CACHE = {}
_GRAPH_CACHE_MAX = 8
_GRAPH_STATS = dict(capture=0, replay=0, fallback=0)


def _input_schema(inputs):
    """Metadata only: never copy/convert or synchronize tensor values."""
    if any(type(name) is not str or
           (t is not None and (not isinstance(t, torch.Tensor)
                              or t.layout != torch.strided))
           for name, t in inputs.items()):
        raise ValueError("Graph inputs must be named strided tensors or None")
    return tuple((name, None if t is None else
                  (tuple(t.shape), tuple(t.stride()), t.dtype, t.device,
                   t.layout, t.requires_grad))
                 for name, t in sorted(inputs.items()))


def _backend_signature(indexer):
    # Unbound methods preserve cross-layer reuse without binding instance IDs.
    # Instance-specific callables stay distinct. Tile settings affect reduction
    # kernels; a disabled Triton backend must not replay an earlier graph.
    def implementation(name):
        fn = getattr(indexer, name)
        return getattr(fn, "__func__", fn)
    return (type(indexer), implementation("_fast_score_coarse"),
            implementation("_fast_score_fine"),
            tuple(sorted(indexer._TILE_COARSE.items())),
            tuple(sorted(indexer._TILE_FINE.items())), indexer._triton_ok)


def _graph_signature(indexer, inputs, *, geometry, modes, output_dtype,
                     stream):
    # Geometry is already rounded integer host data at the callsite. Reject
    # bool/float/NaN rather than allowing Python's True == 1 == 1.0 key alias.
    def integer_geometry(value):
        return (type(value) is int or
                (type(value) is tuple and all(integer_geometry(v) for v in value)))
    if not integer_geometry(geometry) or type(modes) is not tuple or any(
            type(mode) is not bool for mode in modes):
        raise ValueError("Graph signature requires integer geometry and bool modes")
    return ("batch-select-v2", geometry, modes, output_dtype,
            _backend_signature(indexer), _input_schema(inputs), stream)


def _graph_cached_run(key, body_fn, inputs):
    """capture-or-replay：首次 clone 静态缓冲 → side stream warmup 3 →
    capture（warmup/capture 期间 body 真实执行，首层输出即 capture 结果）；
    后续 copy_ 注入新层输入 → replay。返回静态输出缓冲（下一次调用 replay
    前被覆盖——调用方须在同 stream 顺序消费完毕，eager_prefill 现口径满足）。
    逐位一致性机制：replay 重放与 eager 完全相同的 kernel 序列 + 相同地址
    + 相同归约树（e139_l2_graph_poc 双重硬门 0/1008 diff 实证）。"""
    schema = _input_schema(inputs)
    key = (key, schema)  # qualify even direct callers; incompatible schema recaptures
    ent = _GRAPH_CACHE.get(key)
    if ent is None:
        if len(_GRAPH_CACHE) >= _GRAPH_CACHE_MAX:
            _GRAPH_CACHE.clear()      # LRU 简化版：超限全清（重新 capture）
        statics = {n: (t.clone() if t is not None else None)
                   for n, t in inputs.items()}
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                body_fn(**statics)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        from sparse_attn.ops.cuda_graph_state import capture_graph
        with capture_graph(g):
            out = body_fn(**statics)
        # capture 只 record 不执行——replay 一次产出首层真实结果
        g.replay()
        # ent 必须持有 body_fn 闭包：闭包捕获的几何张量（m_far/m_near/
        # near_blks/...）是 graph kernel 固化指针的数据源——失去引用会被
        # allocator 回收复用，replay 将读到被覆盖的垃圾（实测 810/1008 diff）
        ent = (g, statics, out, body_fn, schema, _input_schema(statics))
        _GRAPH_STATS["capture"] += 1
        _GRAPH_CACHE[key] = ent
        return out
    g, statics, out, _, expected, static_schema = ent
    try:
        # Check every field before any copy_: copy_ otherwise broadcasts and casts.
        # clone may normalize source strides, hence separate source/static schemas.
        if schema != expected or _input_schema(statics) != static_schema:
            raise ValueError("CUDA Graph input schema mismatch")
        for n, t in inputs.items():
            if t is not None:
                statics[n].copy_(t)
        g.replay()
        _GRAPH_STATS["replay"] += 1
    except Exception:
        _GRAPH_CACHE.pop(key, None)
        raise
    return out


def _scatter_candidate_scores(buf, scores, tokens, valid, shared_indices=False):
    """Scatter unchanged per-query-head scores into full-history storage.

    Internal contract: tokens/valid originate from disjoint near/far topk pools;
    valid candidates are unique and exclude token zero on the shared path.
    Invalid slots all write -inf to protected token zero, so collisions there
    are idempotent. No softmax, arithmetic reduction or topk order changes.
    """
    if (buf.ndim != 4 or tuple(buf.shape[:2]) != (1, 1)
            or tokens.ndim != 2 or valid.shape != tokens.shape
            or tokens.dtype != torch.long or valid.dtype != torch.bool
            or scores.ndim != 2 or scores.shape[0] != buf.shape[2]
            or scores.shape[1] != tokens.shape[1] or tokens.shape[0] == 0
            or scores.shape[0] % tokens.shape[0] != 0
            or scores.dtype != buf.dtype
            or any(t.device != buf.device for t in (scores, tokens, valid))):
        raise ValueError("candidate scatter schema mismatch")
    Hkv, C = tokens.shape
    H = scores.shape[0]
    G = H // Hkv
    if shared_indices:
        # Index/valid metadata are shared across GQA heads. expand is a view;
        # scatter reads its zero-stride G axis without materializing [H,C].
        idx_w = torch.where(valid, tokens, 0)              # [Hkv,C]
        src_w = torch.where(valid[:, None, :],
                            scores.reshape(Hkv, G, C),
                            scores.new_full((), float("-inf")))
        buf.view(Hkv, G, -1).scatter_(
            2, idx_w[:, None, :].expand(Hkv, G, C), src_w)
    else:
        # Original baseline; retained for opt-out and unsupported geometry.
        valid4 = valid.view(Hkv, 1, C).expand(Hkv, G, C).reshape(H, C)
        idx4 = tokens.view(Hkv, 1, C).expand(Hkv, G, C).reshape(H, C)
        idx_w = torch.where(valid4, idx4, torch.zeros_like(idx4))
        src_w = torch.where(valid4, scores,
                            scores.new_full((), float("-inf")))
        buf[0, 0].scatter_(1, idx_w, src_w)


def _zero_pad_pool(m, blk_idx, blk_valid, lo_blk, n_pool, nbad, dev):
    """【#253 Step2】k > cnt 零补位仿真（分支免化·graph capture 安全）。

    全宽 topk 在 k 超过池内候选数 cnt 时，零值（非候选）位置按位置序
    升序补位。候选是 64-token 整块 → 非候选池位置 = 非 L1 选中块的
    token，位置序 = 块序 × 块内偏移：非选中块排他 cumsum×64 + offset
    < nbad 即入选（固定形状，无数据相关分支）。nbad = k - cnt 是 host
    常量（cnt = min(k1 池预算, 池块数)×64 只依赖池几何）。无效槽
    （L1 topk -inf 垃圾块号，可能越界）统一散到 dummy 列 n_pool 防止
    scatter 越界/与真实选中块碰撞。
    """
    h = blk_idx.shape[0]
    selb = torch.zeros((h, n_pool + 1), dtype=torch.bool, device=dev)
    safe = torch.where(blk_valid, blk_idx - lo_blk,
                       torch.full_like(blk_idx, n_pool))
    selb.scatter_(1, safe, blk_valid)
    nonsel = ~selb[:, :n_pool]                              # [h, n_pool]
    csum = torch.cumsum(nonsel.to(torch.int32), dim=-1)
    base = (csum - nonsel.to(torch.int32)) * 64             # 排他块序 ×64
    off = torch.arange(64, device=dev)
    cond = nonsel[..., None] & (base[..., None] + off < nbad)
    m[:, lo_blk * 64:(lo_blk + n_pool) * 64] |= cond.reshape(h, -1)


def _gate_ok(indexer, q, k):
    """fail-closed 配置门：仅 mavg_tail 生产口径（含其参数化邻域）生效。
    宁可回退不可错——任何未验证组合一律走逐组参考路径。"""
    a = indexer.args
    bs = int(a.tia_block_size)
    try:
        if not (
            getattr(indexer, "_triton_ok", False)
            and indexer.far_select == "4bit" and indexer.near_select == "4bit"
            # e64 分区几何（α>0 且 β>0）——L1/L2 双池公式的前提
            and indexer.alpha > 0 and indexer.beta > 0
            and indexer.gamma is not None
            and not indexer.per_q_head
            and indexer.sigma_select == "none"
            and not indexer.moba_gate
            and not indexer.sinkswa_only
            # async prev_mask 会跨步复用上一 mask——批量版无对应语义
            and not getattr(indexer, "enable_async", False)
            # E151c overrides per-layer L2 budgets only in prepare_mask.
            # Batch selection bypasses that method and does not implement the
            # override. Check configuration, not _e151c_on: batch dispatch
            # occurs before the sequential prefill activation scope.
            and getattr(indexer, "_e151c_quota", None) is None
            # E155 absolute near quotas/window override alpha/beta/gamma in
            # compute_mask. This batch implementation still uses the legacy
            # ratio formulas; reject before preparing shared indices/graphs.
            # Zero is an explicit quota too, not an absent value.
            and getattr(indexer, "nt_near_pages", None) is None
            and getattr(indexer, "nt_near_tokens", None) is None
            and getattr(indexer, "nt_near_len", None) is None
            # 投影基路径 k_qat 为投影特征（散布语义不同），不支持
            and indexer._basis is None and indexer._basis_all is None
            # L2 细筛维度必须与 L1 子空间同源（l2=full 是 128 维全量化，
            # b_q_full 口径不同，不支持）
            and indexer.l2_subspace == "same"
            # 子空间仅支持 tail（生产口径；rope/nope/random/highfreq 未对拍）
            and indexer.enable_subspace and indexer.subspace == "tail"
            and not indexer.static_pair
            # Invalid L1 slots scatter -inf to token zero. It must lie in a
            # protected sink block, never in the candidate pool. Zero sink is
            # legal for the sequential selector, but unsafe on this fast path.
            and indexer.sink_tokens >= bs and indexer.sink_tokens % bs == 0
            and indexer.sliding_window_size % bs == 0
            and k.dim() == 4 and k.shape[0] == 1 and k.shape[-1] == 128
            and k.is_cuda and q.is_cuda and q.shape[0] == 1
            and q.shape[3] == 128
        ):
            return False
    except AttributeError:
        return False
    return True


def _unsupported_selection(require_graph):
    if require_graph:
        raise RuntimeError("required graph execution unavailable: unsupported selector input/configuration")
    return None


def batched_group_sels(indexer, q, k, softmax_scale, group_ends, dense_below,
                       *, require_graph=False):
    """整层一次批量组选择。

    参数：
      indexer: TLIIndexer（生产 mavg_tail 口径）
      q: [1, L, H, D]；k: [1, S, Hkv, D]（S == L）
      softmax_scale: float
      group_ends: 该层全部稀疏组的组末行位置 g1_j（严格递增 int 列表，
        均须为 bs 倍数——未对齐组由调用方先行剔除走逐组回退）
      dense_below: dense 前缀宽（须 bs 倍数；仅作门禁校验）
      require_graph: 验收/测速专用；必须 GRAPH=1 且成功执行 graph，
        不允许静默 eager/None/空组回退。默认 False 保留生产容错行为。
    返回：
      list[[Hkv, g1_j] bool]（顺序与 group_ends 一致；与逐组版
      _prepare_sel 的 m0[..., :c1] 同形）；配置不匹配返回 None（fail-closed）。
    """
    if require_graph and os.environ.get("TLI_PREFILL_SELECT_GRAPH", "0") != "1":
        raise ValueError("require_graph=True needs TLI_PREFILL_SELECT_GRAPH=1")

    bs = int(indexer.args.tia_block_size)
    if not _gate_ok(indexer, q, k):
        return _unsupported_selection(require_graph)
    if len(group_ends) == 0:
        if require_graph:
            return _unsupported_selection(require_graph)
        return []
    if any(int(g) % bs != 0 for g in group_ends):
        return _unsupported_selection(require_graph)  # 未对齐组与逐组 pad 统计语义不同
    if int(dense_below) % bs != 0:
        return _unsupported_selection(require_graph)
    L = int(k.shape[1])
    if L != int(q.shape[1]):
        return _unsupported_selection(require_graph)
    N = len(group_ends)
    H, Hkv = int(q.shape[2]), int(k.shape[2])
    G = H // Hkv
    if H % Hkv != 0 or G < 1:
        return _unsupported_selection(require_graph)
    if softmax_scale is None:
        softmax_scale = float(q.shape[3]) ** -0.5
    dev = q.device
    qdtype = q.dtype
    g1 = torch.tensor(group_ends, dtype=torch.long, device=dev)   # [N]

    # D' 静态层掩码解析（与逐组 prepare_index 头部 _resolve_skip 同构；
    # skip_far 层走省算分支，批量版未实现该分支 → fail-closed 回退）
    if getattr(indexer, "_skip_ids", None) is not None:
        indexer._resolve_skip()
        if indexer.skip_far:
            return _unsupported_selection(require_graph)

    # ---- ① 每层一次全前缀统计（纲领②：共享 K 索引） ----
    # 组末行 g1 均为 bs 倍数 → 行前缀 = 整块前缀（无部分块），全前缀统计的
    # 块 [0, g1/bs) 与逐组版 k[:, :g1] 的块统计逐位相同（块内归约独立）；
    # inc 缓存留 S=L 对 decode append-only 前进兼容
    cu_full = torch.tensor([0, L], dtype=torch.int32, device=dev)
    index_dict = indexer.prepare_index(k, cu_full)
    k_min = index_dict["k_min"][0]        # [Tb, Hkv, d']（q/k dtype）
    k_max = index_dict["k_max"][0]
    k_avg_d = index_dict.get("k_avg")
    k_avg = k_avg_d[0] if k_avg_d is not None else None   # [Tb, Hkv, d']
    k_qat = index_dict["k_qat"][0]        # [S, Hkv, delta]（l2=same → delta=d'）
    Tb = int(k_min.shape[0])
    indexer.group_size = G                # 与 compute_score 副作用对齐

    # ---- ② 池边界先行（段分型省算需先知道每行 far/near 段几何） ----
    # 公式逐字段回 tli_indexer.py compute_mask L1880-1897（零 host 同步）
    sink_tok = int(indexer.sink_tokens)
    swa_tok = int(indexer.sliding_window_size)
    sink_lo = int(indexer._sink_pool_lo_blk)
    kt = g1 // bs                                           # [N]
    blk = torch.arange(Tb, device=dev)
    # near_len_dyn = max(bs, int(alpha*mid_len))——float64 与 Python int() 同精度
    # 同截断（compute_mask L1883）
    mid_len = (g1 - (sink_tok + swa_tok)).clamp(min=0)
    nld = (mid_len.to(torch.float64) * float(indexer.alpha)).to(torch.int64)
    near_len_dyn = torch.maximum(torch.full_like(g1, bs), nld)
    near_base = g1 - swa_tok
    near_blks = torch.maximum(
        torch.full_like(g1, sink_lo), (near_base - near_len_dyn) // bs)
    swa_lo_blk = torch.maximum(near_blks, kt - max(1, swa_tok // bs))
    far_lo = sink_lo
    # L1 双池预算（compute_mask L1947-1950）
    k1 = int(indexer.args.tia_level1_topk)
    nb_near = max(1, int(round(k1 * float(indexer.beta))))
    nb_far = max(0, k1 - nb_near)
    kf = min(nb_far, Tb)
    kn = min(nb_near, Tb)
    if kf < 0 or kn < 0:
        return _unsupported_selection(require_graph)
    # 块级因果（硬门）：块 b 可见于行 j ⟺ b < g1_j//bs（整块 ⊆ 前缀）
    blk_causal = blk[None, None, :] < kt.view(N, 1, 1)      # [N,1,Tb]
    m_far = (blk_causal
             & (blk[None, None, :] >= far_lo)
             & (blk[None, None, :] < near_blks.view(N, 1, 1)))
    m_near = (blk_causal
              & (blk[None, None, :] >= near_blks.view(N, 1, 1))
              & (blk[None, None, :] < swa_lo_blk.view(N, 1, 1)))

    # ---- ③ 段分型省算批量 L1 打分（主会话注入·acc23d564ad2eff91，已落实） ----
    # 行 j 实际消费块 = far 段 [far_lo, nb_j)（far_method 分）∪ near 段
    # [nb_j, sl_j)（near_method 分）；swa/因果无效块不打分——行级段切片
    # （比 chunk 段并集更紧，省算彻底落实：far 段只算 far_method 对应的
    # 那种分、near 段只算 near_method 对应的那种分）。
    # 消费谓词对齐 compute_mask L1959-1970：far 池用 avg ⟺ far_method=="avg"
    # and k_avg 存在；near 池用 minmax ⟺ near_method=="minmax" or k_avg 缺失
    # （统一 use_mm 极性：True=minmax 上界分，False=avg 精确分）
    far_use_mm = not (indexer.far_method == "avg" and k_avg is not None)
    near_use_mm = (indexer.near_method == "minmax" or k_avg is None)
    # b_q 口径对齐 compute_score L1281-1282：(q*scale)（bf16 标量乘，与参考
    # 同为先 bf16 舍入再 fp32 cast）→ tail32 子空间 gather → [N,Hkv,G,d']
    idx_sub = indexer._subspace_indices(dev)
    q_rows = q[0, g1 - 1]                                  # [N, H, D] 组末行
    b_q = (q_rows * softmax_scale).to(torch.float32)[..., idx_sub]
    b_q = b_q.view(N, Hkv, G, -1)                          # q-head i = h*G+g
    # 【对拍门自洽关键】逐行调与逐组参考链完全相同的 Triton fast kernel
    # （compute_score L1334-1345 _fast_score_coarse / L1393-1404
    # _fast_score_fine）：einsum/cublas 与 Triton fp32 累加顺序不同 → bf16
    # 舍入后 ULP 级分数差 → L1 topk 边界翻转（块级 minmax 上界分平坦，边界
    # 密集）→ ~0.45% 选择差 → 输出级 allclose(1e-2) 数学不可达（einsum 版
    # 实测 maxdiff 1.8e-1）。对齐同一 kernel 消除 L1 侧翻转源；kernel
    # per-tile 独立（grid=(cdiv(T,bt), Hkv)）→ 行级段切片 [t0,t1) 与参考
    # 全前缀调用的对应列逐位一致（切片不改变 tile 内数值树）。GQA mean
    # 批量收口：kernel 输出 per-q-head [H,W]（kernel 内 GQA K 复用）→
    # 段写入 [N,H,Tb] buffer → 一次性 view(N,Hkv,G,Tb).mean(2)；cast 序
    # 与 compute_mask L1842-1845 同序（kernel 直出 q.dtype 再 mean over g）。
    # launch 面：每行 2 次小 kernel（far 段 + near 段）+ 段写入 ≈ 2N 次/层
    # （L=32K N=496 时 ~1k，vs 逐组链 ~20k——批量化收益主体在 prepare_index/
    # 量化/池几何/胶水的 N 倍削减，打分 kernel 本就是单行形状）。
    # 池边界 nb_j/sl_j 一次性 D2H（[N] 小张量 tolist 单次同步；行循环
    # kernel launch 在 host 侧排队，无逐行同步）。
    # fail-closed：kernel 异常（含 _triton_ok 置 False 的实例级回退）→
    # return None 走逐组参考路径。
    # ---- 【E139 #253】层间不变量外提（graph capture 硬约束：capture 区内
    # 禁 D2H 同步——tolist()/item() 全部外提；几何/预算 host 值只依赖
    # g1/池边界/indexer 常量，同一 prompt 内 36 层全同 → capture 一次固化） ----
    nb_l = near_blks.tolist()
    sl_l = swa_lo_blk.tolist()
    # ④ L2 预算向量化（compute_mask L2126-2359 逐字段；纯 GPU 张量算术，
    # 层间不变——闭包捕获地址供 eager，graph capture 固化 kernel 参数）
    K2arg = int(indexer.args.tia_level2_topk)
    K2 = torch.minimum(g1, torch.full_like(g1, K2arg))      # min(p 宽=g1_j, K2)
    far_tok_lo = far_lo * bs
    far_tok_hi = torch.minimum(near_blks * bs, g1)          # min(near_blks*bs, p 宽)
    # K2_mid = K2 − sink − swa（sink/swa 强制区计入 K2 总预算；mid 选择名额 = K2_mid）
    K2_mid = (K2 - sink_tok - torch.minimum(g1, torch.full_like(g1, swa_tok))
              ).clamp(min=0)
    # nt_near = min(int(nb_near·bs·γ), K2_mid)（e64 分支 L2300；int 为 Python
    # 标量算术与参考逐位同值，K2_mid 逐行 min）
    nt_near = torch.minimum(
        torch.full_like(K2_mid, int(nb_near * bs * float(indexer.gamma))), K2_mid)
    far_budget = (K2_mid - nt_near).clamp(min=0)
    far_width = (far_tok_hi - far_tok_lo).clamp(min=0)
    k2_far = torch.minimum(torch.minimum(far_budget, far_width), K2_mid)
    k2_near = (K2_mid - k2_far).clamp(min=0)
    # near 池内截断饥饿语义（L2349 min(k2_near, 池宽)）在 ⑤ 逐行调用处以
    # min(k2_near_l[j], near_p.shape[-1]) 落地（池宽 = swa_lo_tok−far_tok_hi）
    far_lo_tok = far_lo * bs
    h_idx_col = torch.arange(Hkv, device=dev).view(Hkv, 1)  # 广播 gather 列
    k2_far_l = k2_far.tolist()
    k2_near_l = k2_near.tolist()
    # 【#253 Step2】L2 双池 topk 候选列化（默认 0 fail-closed；graph key
    # 含该 env 指纹——env 翻转不混用缓存）。逐行小 kernel 版实测倒挂
    # （[Hkv,C] 微 grid × 2N 次 topk 的 launch/延迟 > topk 节省）→ v2
    # 批量化：池掩码/topk/rank 过滤全部跨行批量（层内 2 次大 topk），
    # 逐行仅剩 pc gather + mask scatter 两个不可批量 kernel。
    # Candidate zero-padding counts and offsets below are specialized for 64
    # tokens per block. Other block sizes retain full-width L2 topk; applying
    # the specialization would underfill or write outside the intended pool.
    use_cand_topk = (os.environ.get("TLI_PREFILL_L2_TOPK_CAND", "0") == "1"
                     and bs == 64)
    # Opt-in allocation reduction only. A protected dummy token and GQA
    # replication are required; unsupported geometry retains the baseline.
    use_shared_indices = (
        os.environ.get("TLI_PREFILL_SHARED_GQA_INDICES", "0") == "1"
        and sink_tok >= bs and far_lo >= 1 and Hkv > 1 and G > 1)
    if use_cand_topk:
        # 层不变 device/host 值（graph capture 外一次性外提——k_max 的
        # .max() 是 D2H 同步，禁入 capture 区）
        swa_lo_all = (g1 - swa_tok).clamp(min=0)              # [N]
        k_near_all = torch.minimum(
            k2_near, (swa_lo_all - far_tok_hi).clamp(min=0))  # [N]
        k_max_f = int(k2_far.max())
        k_max_n = int(k_near_all.max())
        keep_f = (torch.arange(k_max_f, device=dev)[None, :]
                  < k2_far[:, None]) if k_max_f > 0 else None
        keep_n = (torch.arange(k_max_n, device=dev)[None, :]
                  < k_near_all[:, None]) if k_max_n > 0 else None
        k_near_l = k_near_all.tolist()
        # nbad = k − cnt（cnt = min(池预算, 池块数)×64 只依赖池几何——
        # L1 sc 的 -inf 模式头不变，与数据无关）→ host 常量，capture 安全
        nbad_f_l = [int(k2_far_l[j]) - min(kf, int(nb_l[j]) - far_lo) * 64
                    for j in range(N)]
        nbad_n_l = [k_near_l[j] - min(kn, int(sl_l[j]) - int(nb_l[j])) * 64
                    for j in range(N)]

    # ---- ③④⑤ 选择体（输入 k_min/k_max/k_avg/k_qat/b_q 逐层变化；其余
    # 闭包捕获层间不变量）——eager 与 CUDA Graph capture 共用同一 body ----
    def _body(k_min, k_max, k_avg, k_qat, b_q):
        sc_far_q = torch.full((N, H, Tb), float("-inf"),
                              dtype=qdtype, device=dev)    # [N,H,Tb] per-q-head
        sc_near_q = torch.full((N, H, Tb), float("-inf"),
                               dtype=qdtype, device=dev)
        for j in range(N):
            bq = b_q[j].reshape(H, -1)                     # [H, d'] fp32 连续
            # far 段 [far_lo, nb_j)：far_method 对应分
            r1 = int(nb_l[j])
            if kf > 0 and r1 > far_lo:
                if far_use_mm:
                    out = indexer._fast_score_coarse(
                        k_min[far_lo:r1], k_max[far_lo:r1], bq, qdtype)
                else:
                    out = indexer._fast_score_fine(
                        k_avg[far_lo:r1], bq, qdtype)
                sc_far_q[j, :, far_lo:r1] = out            # [H, W]
            # near 段 [nb_j, sl_j)：near_method 对应分
            r0, r1 = int(nb_l[j]), int(sl_l[j])
            if kn > 0 and r1 > r0:
                if near_use_mm:
                    out = indexer._fast_score_coarse(
                        k_min[r0:r1], k_max[r0:r1], bq, qdtype)
                else:
                    out = indexer._fast_score_fine(
                        k_avg[r0:r1], bq, qdtype)
                sc_near_q[j, :, r0:r1] = out                # [H, W]
        # GQA 组内 mean over g（批量一次；cast 序同上——数值树与逐组版同序）
        sc_far = sc_far_q.view(N, Hkv, G, Tb).mean(dim=2)      # [N,Hkv,Tb]
        sc_near = sc_near_q.view(N, Hkv, G, Tb).mean(dim=2)

        # 段外补 -inf 后按池掩码收口（行级段切片 ⊆ 池段；池掩码兜底因果与
        # 段边界外的残留值——与全宽版 where 同一选择集）。
        # 注：逐组版 compute_mask L1937「当前块置 inf」不复刻——kt-1 恒不在
        # 任何池内（sl ≤ kt-2 → kt-1 ∉ [nb, sl)；nb ≤ kt-3 → kt-1 ∉ far 段），
        # 两版该块均被池掩码 -inf，选择集逐位相同。
        sc_far = sc_far.masked_fill(~m_far, float("-inf"))     # [N,Hkv,Tb]
        sc_near = sc_near.masked_fill(~m_near, float("-inf"))
        # 统一 k=nb_far/nb_near topk + -inf 无效过滤 ≡ 逐组版 min(k, 池宽)：
        # 池宽不足的行 topk 补位是 -inf → 过滤后恰为全部有效块（同一选择集）
        tv_f = torch.topk(sc_far, kf, dim=-1) if kf > 0 else None
        tv_n = torch.topk(sc_near, kn, dim=-1) if kn > 0 else None
        if tv_f is not None:
            i_f, v_f = tv_f.indices, tv_f.values                # [N,Hkv,nb_far]
            valid_f = v_f > float("-inf")
        else:
            i_f = torch.zeros(N, Hkv, 0, dtype=torch.long, device=dev)
            valid_f = i_f.bool()
        if tv_n is not None:
            i_n, v_n = tv_n.indices, tv_n.values                # [N,Hkv,nb_near]
            valid_n = v_n > float("-inf")
        else:
            i_n = torch.zeros(N, Hkv, 0, dtype=torch.long, device=dev)
            valid_n = i_n.bool()
        # 候选块布局：[far 槽(nf) | near 槽(nn)]——far/near 池块区间不相交，
        # topk 索引无重复 → token 无重复
        if use_cand_topk:
            # L1 垃圾槽（-inf 补位块号，可 ≥ g1_j 越界）清零到块 0：块 0
            # ∈ sink 区恒非候选 → cand_tok 全程 < g1_j（gather/scatter 免
            # 越界防护），行为中性（② 段 idx_w 本就重定向无效槽到 0）
            i_f = torch.where(valid_f, i_f, torch.zeros_like(i_f))
            i_n = torch.where(valid_n, i_n, torch.zeros_like(i_n))
        cand_blk = torch.cat([i_f, i_n], dim=-1)                # [N,Hkv,C_blk]
        cand_valid_blk = torch.cat([valid_f, valid_n], dim=-1)
        if use_cand_topk:
            # Step2 tie 序等价性前提：cat 后块内序是 L1 topk rank 序——
            # sort 成全局 token 位置升序（CUDA radix topk 相等值取低
            # index，候选须位置序才与全宽位置序 tie 序一致）。valid 随
            # perm 同步重排；①②③ 段 gather/scatter 与序无关 → 数值不变。
            cand_blk, _perm = torch.sort(cand_blk, dim=-1)
            cand_valid_blk = torch.gather(cand_valid_blk, -1, _perm)
        n_far = int(i_f.shape[-1])
        C_blk = int(cand_blk.shape[-1])
        C = C_blk * bs                                          # 8192（生产口径）
        cand_tok = (cand_blk * bs)[..., None] + torch.arange(bs, device=dev)
        cand_tok = cand_tok.reshape(N, Hkv, C)                  # [N,Hkv,C]
        cand_valid = cand_valid_blk[..., None].expand(N, Hkv, C_blk, bs) \
            .reshape(N, Hkv, C)

        # ---- ⑤ L2：候选内 gather 打分（纲领③）+ softmax/双池 topk 全宽逐位复刻 ----
        # 打分仍只在候选列（纲领③省算主体：k_qat 候选 gather + 逐行调与参考链
        # 同一 Triton fine kernel——compute_score L1459-1471 _fast_score_fine，
        # kernel per-tile 独立 → 候选列分数与参考全前缀输出的对应 token 逐位
        # 一致；einsum/cublas 版累加顺序差是 L2 topk 边界翻转源，实测 ~0.45%
        # 选择差，不用）。
        # softmax/GQA mean/L2 双池 topk 则按行以与逐组参考完全相同的形状·数值
        # ·内存布局逐算子复刻：[1,1,H,g1] 的 where(topk_mask, sf, -inf) 等价
        # 张量（全宽 -inf + 候选 scatter）→ F.softmax → nan_to_num →
        # view(1,1,Hkv,G,·).mean(-2) → far/near 位置切片 topk → scatter →
        # sink/swa 强制。理由：候选列 softmax 的归约树宽度差（g1 vs C）与
        # topk 池序差（位置序 vs L1 topk rank 序）是仅剩的 ULP 翻转源——
        # 候选列版 mask 一致率 0.9990（过 mask 门）但 0.1% 翻转使输出级
        # allclose(1e-2) 不可达（实测 maxdiff 1.8e-1）；同形同值复刻后全部
        # 算子确定性 → mask 逐位一致。softmax/mean/topk 是逐元素/归约廉价
        # 算子（O(H·g1) ≪ 打分 O(H·C·δ)），复刻成本远小于打分省算收益。
        if use_cand_topk:
            # 池掩码批量化 [N,Hkv,C]（层内 6 kernel 替代逐行 6N）：
            # far [far_lo_tok, far_tok_hi) / near [far_tok_hi, swa_lo_all)
            in_far_b = (cand_tok >= far_lo_tok) & (
                cand_tok < far_tok_hi.view(N, 1, 1))
            in_near_b = (cand_tok >= far_tok_hi.view(N, 1, 1)) & (
                cand_tok < swa_lo_all.view(N, 1, 1))
            pc_all = torch.empty((N, Hkv, C), dtype=qdtype, device=dev)
        out_masks = []
        for j in range(N):
            g1_j = int(group_ends[j])
            ct_j = cand_tok[j]                              # [Hkv, C]
            valid_j = cand_valid[j]                         # [Hkv, C]
            # ① 候选内打分：gather + 同一 fine kernel → [H, C] per-q-head
            kq_c = k_qat[ct_j, h_idx_col]                   # [Hkv, C, δ] 连续
            kq_c = kq_c.permute(1, 0, 2)                    # [C, Hkv, δ]
            sf_c = indexer._fast_score_fine(
                kq_c, b_q[j].reshape(H, -1), qdtype)
            # ② where(topk_mask, sf, -inf) 等价张量：无效槽指向 token 0
            # （sink 区 [0,sink) 恒非候选 → -inf 幂等写入无覆盖问题；垃圾
            # 块号不得散写 [0,g1)——可能命中有效候选位）
            buf = torch.full((1, 1, H, g1_j), float("-inf"),
                             dtype=qdtype, device=dev)
            _scatter_candidate_scores(buf, sf_c, ct_j, valid_j,
                                      shared_indices=use_shared_indices)
            # ③ softmax → nan_to_num → GQA mean（与参考 L2043-2046 同形同值）
            p = F.softmax(buf, dim=-1).nan_to_num(nan=0)    # [1,1,H,g1]
            p = p.view(1, 1, Hkv, G, g1_j).mean(dim=-2)     # [1,1,Hkv,g1]
            # 参考 L2051 的 swa 区 p=1.0 赋值不复刻：双池切片
            # [far_lo,far_hi)/[far_hi,swa_lo) 均不含 [swa_lo,g1)，值不受影响
            if use_cand_topk:
                # ④'-pre：候选列 p 值 gather 进批量缓冲（零 ULP——纯
                # gather 无算术；垃圾槽已清零 → 恒 < g1_j 无越界）
                torch.gather(p[0, 0], 1, ct_j, out=pc_all[j])
                continue
            # ④ L2 双池位置切片 topk（预算 host 标量逐行；参考 L2262-2354
            # 同款 guard/k/偏移——far 空 guard L2254-2255、near 倒挂 L2350）
            # ——默认路径零扰动原样保留
            far_hi_tok = min(int(nb_l[j]) * bs, g1_j)
            swa_lo_tok_j = max(0, g1_j - swa_tok)
            if far_hi_tok > far_lo_tok:
                i_f2 = torch.topk(
                    p[..., far_lo_tok:far_hi_tok], k2_far_l[j],
                    dim=-1).indices + far_lo_tok
            else:
                i_f2 = p.new_zeros((1, 1, Hkv, 0), dtype=torch.long)
            if swa_lo_tok_j > far_hi_tok:
                near_p = p[..., far_hi_tok:swa_lo_tok_j]
                i_n2 = torch.topk(
                    near_p, min(k2_near_l[j], int(near_p.shape[-1])),
                    dim=-1).indices + far_hi_tok
            else:
                i_n2 = i_f2[..., :0]
            # ⑤ mask：scatter + sink/swa 正交强制区（参考 L2352-2358 同款）
            m = torch.zeros((1, 1, Hkv, g1_j), dtype=torch.bool, device=dev)
            m.scatter_(-1, i_f2, True)
            m.scatter_(-1, i_n2, True)
            m[..., :sink_tok] = True
            m[..., swa_lo_tok_j:g1_j] = True
            out_masks.append(m[0, 0])
        if use_cand_topk:
            # ④' 【#253 Step2 v2】L2 双池 topk 候选列化（跨行批量）：O(C)
            # topk 两池各一次全批量 kernel（[N·Hkv, C] 满占用，替代 2N 次
            # [Hkv, W/C] 微 grid topk——v1 逐行版实测倒挂的根因）。等价性：
            #   ① p 值 gather 自同一全宽 p 张量（零 ULP）+ 候选位置序
            #     （cand_blk sort）→ tie 序 ≡ 全宽位置序（radix topk
            #     相等值取低 index）；
            #   ② k 逐行差异（k2_far_l[k]）用 rank < k_j 掩码分支免化
            #     （keep_f/keep_n 层不变批量掩码），k > cnt 行的 -inf 槽
            #     过滤 + _zero_pad_pool 池内非候选块位置序补位仿真。
            # scatter src 用 ok 掩码张量（True/False 槽）——False 槽位置
            # 与 True 槽/他池/强制区互斥，写 False 等价不写。
            # 【关键】ok=False 槽（rank 溢出 + -inf 补位）批量重定向到
            # token 0（sink 区恒 True，幂等）：-inf 槽指向池外候选位置，
            # near/far 双 scatter 有序执行时后者会把前者已写的 True 抹
            # False（K2=8192 补位档实测 890/1008 崩）——与 ② 段 idx_w 同
            # 一防护，批量 where 层内 1 kernel。
            k_f_t = min(k_max_f, C)
            if k_f_t > 0:
                pc_f = torch.where(in_far_b, pc_all, float("-inf"))
                tv_f = torch.topk(pc_f, k_f_t, dim=-1)     # [N,Hkv,k_f_t]
                sel_f = cand_tok.gather(-1, tv_f.indices)
                ok_f = (tv_f.values > float("-inf")) & keep_f[:, None, :k_f_t]
                sel_f = torch.where(ok_f, sel_f, 0)
            k_n_t = min(k_max_n, C)
            if k_n_t > 0:
                pc_n = torch.where(in_near_b, pc_all, float("-inf"))
                tv_n = torch.topk(pc_n, k_n_t, dim=-1)
                sel_n = cand_tok.gather(-1, tv_n.indices)
                ok_n = (tv_n.values > float("-inf")) & keep_n[:, None, :k_n_t]
                sel_n = torch.where(ok_n, sel_n, 0)
            # ⑤ 逐行 mask scatter（m 宽度逐行不同，不可批量）+ 补位 +
            # sink/swa 强制
            for j in range(N):
                g1_j = int(group_ends[j])
                swa_lo_tok_j = max(0, g1_j - swa_tok)
                m = torch.zeros((1, 1, Hkv, g1_j), dtype=torch.bool, device=dev)
                if k_f_t > 0:
                    m[0, 0].scatter_(-1, sel_f[j], True)
                if nbad_f_l[j] > 0:
                    _zero_pad_pool(m[0, 0], i_f[j], valid_f[j], far_lo,
                                   int(nb_l[j]) - far_lo, nbad_f_l[j], dev)
                if k_n_t > 0:
                    m[0, 0].scatter_(-1, sel_n[j], True)
                if nbad_n_l[j] > 0:
                    _zero_pad_pool(m[0, 0], i_n[j], valid_n[j], int(nb_l[j]),
                                   int(sl_l[j]) - int(nb_l[j]), nbad_n_l[j], dev)
                m[..., :sink_tok] = True
                m[..., swa_lo_tok_j:g1_j] = True
                out_masks.append(m[0, 0])
        return out_masks

    # ---- 双路径：CUDA Graph（env TLI_PREFILL_SELECT_GRAPH 默认 0 fail-closed）
    # capture/replay 任何异常 → 清缓存回退 eager；eager body 异常 → return None
    # 走逐组参考路径（与原版语义一致）。graph 输出为静态缓冲——下一次调用
    # replay 前被覆盖，调用方（eager_prefill）同 stream 顺序消费满足生命周期。
    if os.environ.get("TLI_PREFILL_SELECT_GRAPH", "0") == "1":
        # key = 几何 + 配置指纹（不含 indexer 实例 id——e2e 每层独立
        # indexer 实例，跨层共享同一 graph：statics 注入层间数据，闭包持
        # 首层几何（层间不变）与首层 indexer（仅 capture 期发起 kernel，
        # 参数固化为 statics 指针，replay 不再访问实例）
        key = None
        try:
            inputs = dict(k_min=k_min, k_max=k_max, k_avg=k_avg,
                          k_qat=k_qat, b_q=b_q)
            # Bind all captured geometry/host budgets, including rounded pool
            # edges. No new D2H synchronization: these lists already exist.
            key = _graph_signature(
                indexer, inputs,
                geometry=(L, N, H, Hkv, G, Tb, bs, sink_tok, swa_tok,
                          far_lo, kf, kn, K2arg, tuple(group_ends),
                          tuple(nb_l), tuple(sl_l), tuple(k2_far_l),
                          tuple(k2_near_l)),
                modes=(far_use_mm, near_use_mm, use_cand_topk,
                       use_shared_indices),
                output_dtype=qdtype,
                stream=(dev, torch.cuda.current_stream(dev).cuda_stream))
            with torch.cuda.device(dev):
                return _graph_cached_run(key, _body, inputs)
        except Exception:
            _GRAPH_STATS["fallback"] += 1
            # Schema-qualified replay eviction happens inside _graph_cached_run.
            if require_graph:
                raise  # Never certify an eager fallback as Graph execution.
    try:
        return _body(k_min, k_max, k_avg, k_qat, b_q)
    except Exception:
        return None
