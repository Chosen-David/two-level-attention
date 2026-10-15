# -*- coding: utf-8 -*-
"""E139 #253 PoC：CUDA Graph capture L2 循环（1008 行 × ~15 算子 ≈ 15k launch
→ 1 次 replay）。判决性问题：L2-B 216.5ms 是 dispatch-bound（32K→64K 仅 2.2x
线性=行数），graph 化后是否塌到 GPU 工作量级。
硬门：replay 输出 mask 与 eager 逐位一致（同 kernel 同地址同归约树）。
协议：warmup 3（side stream）→ capture → 7-iter 中位 replay 计时，L=65536。
"""
import os
import sys
import time

TREE = os.path.expanduser("~/tmp_e138_abl_tree_187")
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "exp", "bench"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from e131_arm_bench import make_args  # noqa: E402
from sparse_attn.indexer.tli_indexer import TLIIndexer  # noqa: E402
from sparse_attn.ops.batch_select import _gate_ok  # noqa: E402

DEV = "cuda:0"
L = int(os.environ.get("L", "65536"))
D = 128
SS = D ** -0.5


def make_qkv(seed=999):
    g = torch.Generator(device=DEV).manual_seed(seed)
    H, Hkv = 32, 8
    return (torch.randn(1, L, H, D, device=DEV, dtype=torch.bfloat16, generator=g),
            torch.randn(1, L, Hkv, D, device=DEV, dtype=torch.bfloat16, generator=g),
            torch.randn(1, L, Hkv, D, device=DEV, dtype=torch.bfloat16, generator=g))


def fresh_idx(K2=1024):
    ns = make_args("mavg_tail")
    ns.tia_level2_topk = K2
    idx = TLIIndexer(ns)
    idx.layer_idx = 2
    idx.clear()
    return idx


gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
print(f"=== E139 L2 CUDA-Graph PoC L={L} GPU{gpu} ===", flush=True)
os.environ["TLI_SPARSE_PREFILL_CHUNK"] = "2048"
os.environ["TLI_PREFILL_PREFETCH"] = "0"
os.environ.pop("TLI_PREFILL_DENSE_BELOW", None)
os.environ["TLI_PREFILL_SELECT_GROUP"] = "64"
os.environ["TLI_PREFILL_BATCH_SELECT"] = "1"
os.environ["TLI_PREFILL_FUSED_ATTN"] = "1"

q, k, v = make_qkv()
bs = 64
dense_below = 1024
grp = 64
chunk = 2048
bounds = [(0, dense_below)] + [
    (c0, min(c0 + chunk, L)) for c0 in range(dense_below, L, chunk)
]
pairs = []
for (bc0, bc1) in bounds:
    if bc0 >= dense_below:
        for bg0 in range(bc0, bc1, grp):
            pairs.append((bg0, min(bg0 + grp, bc1)))
group_ends = [g1 for (_, g1) in pairs if g1 % bs == 0]
N = len(group_ends)
print(f"N={N}", flush=True)

idx = fresh_idx()
assert _gate_ok(idx, q, k)
dev = q.device
qdtype = q.dtype
H, Hkv = q.shape[2], k.shape[2]
G = H // Hkv
g1_t = torch.tensor(group_ends, dtype=torch.long, device=dev)
cu_full = torch.tensor([0, L], dtype=torch.int32, device=dev)

index_dict = idx.prepare_index(k, cu_full)
k_qat = index_dict["k_qat"][0]
k_min = index_dict["k_min"][0]
k_max = index_dict["k_max"][0]
Tb = int(k_min.shape[0])
idx.group_size = G

sink_tok = int(idx.sink_tokens)
swa_tok = int(idx.sliding_window_size)
sink_lo = int(idx._sink_pool_lo_blk)
kt = g1_t // bs
mid_len = (g1_t - (sink_tok + swa_tok)).clamp(min=0)
nld = (mid_len.to(torch.float64) * float(idx.alpha)).to(torch.int64)
near_blks = torch.maximum(torch.full_like(g1_t, bs), nld)
near_base = g1_t - swa_tok
near_blks = torch.maximum(
    torch.full_like(g1_t, sink_lo), (near_base - nld) // bs)
swa_lo_blk = torch.maximum(near_blks, kt - max(1, swa_tok // bs))
far_lo = sink_lo
k1 = int(idx.args.tia_level1_topk)
nb_near = max(1, int(round(k1 * float(idx.beta))))
nb_far = max(0, k1 - nb_near)
kf = min(nb_far, Tb)
kn = min(nb_near, Tb)
idx_sub = idx._subspace_indices(dev)
q_rows = q[0, g1_t - 1]
b_q = (q_rows * SS).to(torch.float32)[..., idx_sub]
b_q = b_q.view(N, Hkv, G, -1)
nb_l = near_blks.tolist()
sl_l = swa_lo_blk.tolist()

# ---- L1 打分循环（原样复刻，产出 L2 候选） ----
def l1_loop():
    sc_far_q = torch.full((N, H, Tb), float("-inf"), dtype=qdtype, device=dev)
    sc_near_q = torch.full((N, H, Tb), float("-inf"), dtype=qdtype, device=dev)
    for j in range(N):
        bq = b_q[j].reshape(H, -1)
        r1 = int(nb_l[j])
        if kf > 0 and r1 > far_lo:
            out = idx._fast_score_coarse(
                k_min[far_lo:r1], k_max[far_lo:r1], bq, qdtype)
            sc_far_q[j, :, far_lo:r1] = out
        r0, r1 = int(nb_l[j]), int(sl_l[j])
        if kn > 0 and r1 > r0:
            out = idx._fast_score_coarse(
                k_min[r0:r1], k_max[r0:r1], bq, qdtype)
            sc_near_q[j, :, r0:r1] = out
    return sc_far_q, sc_near_q

sc_far_q, sc_near_q = l1_loop()
sc_far = sc_far_q.view(N, Hkv, G, Tb).mean(dim=2)
sc_near = sc_near_q.view(N, Hkv, G, Tb).mean(dim=2)
blk = torch.arange(Tb, device=dev)
blk_causal = blk[None, None, :] < kt.view(N, 1, 1)
m_far = (blk_causal & (blk[None, None, :] >= far_lo)
         & (blk[None, None, :] < near_blks.view(N, 1, 1)))
m_near = (blk_causal & (blk[None, None, :] >= near_blks.view(N, 1, 1))
          & (blk[None, None, :] < swa_lo_blk.view(N, 1, 1)))
sc_far = sc_far.masked_fill(~m_far, float("-inf"))
sc_near = sc_near.masked_fill(~m_near, float("-inf"))
tv_f = torch.topk(sc_far, kf, dim=-1)
tv_n = torch.topk(sc_near, kn, dim=-1)
i_f, v_f = tv_f.indices, tv_f.values
i_n, v_n = tv_n.indices, tv_n.values
cand_tok = ((torch.cat([i_f, i_n], dim=-1) * bs)[..., None]
            + torch.arange(bs, device=dev)).reshape(N, Hkv, -1)
cand_valid = torch.cat([v_f > float("-inf"), v_n > float("-inf")],
                       dim=-1)[..., None].expand(N, Hkv, -1, bs) \
    .reshape(N, Hkv, -1)
C = int(cand_tok.shape[-1])
h_idx_col = torch.arange(Hkv, device=dev).view(Hkv, 1)
K2arg = int(idx.args.tia_level2_topk)
K2 = torch.minimum(g1_t, torch.full_like(g1_t, K2arg))
far_tok_hi = torch.minimum(near_blks * bs, g1_t)
K2_mid = (K2 - sink_tok - torch.minimum(g1_t, torch.full_like(g1_t, swa_tok))
          ).clamp(min=0)
nt_near = torch.minimum(
    torch.full_like(K2_mid, int(nb_near * bs * float(idx.gamma))), K2_mid)
far_budget = (K2_mid - nt_near).clamp(min=0)
far_width = (far_tok_hi - far_lo * bs).clamp(min=0)
k2_far = torch.minimum(torch.minimum(far_budget, far_width), K2_mid)
k2_near = (K2_mid - k2_far).clamp(min=0)
k2_far_l = k2_far.tolist()
k2_near_l = k2_near.tolist()
far_lo_tok = far_lo * bs

# ---- L2 循环（参数化：输入为显式 tensor 参数，graph 静态地址） ----
def l2_loop_args(k_qat_i, cand_tok_i, cand_valid_i, b_q_i):
    out = []
    for j in range(N):
        g1_j = int(group_ends[j])
        ct_j = cand_tok_i[j]
        valid_j = cand_valid_i[j]
        kq_c = k_qat_i[ct_j, h_idx_col]
        kq_c = kq_c.permute(1, 0, 2)
        sf_c = idx._fast_score_fine(kq_c, b_q_i[j].reshape(H, -1), qdtype)
        valid4 = valid_j.view(Hkv, 1, C).expand(Hkv, G, C).reshape(H, C)
        idx4 = ct_j.view(Hkv, 1, C).expand(Hkv, G, C).reshape(H, C)
        idx_w = torch.where(valid4, idx4, torch.zeros_like(idx4))
        src_w = torch.where(valid4, sf_c, sf_c.new_full((), float("-inf")))
        buf = torch.full((1, 1, H, g1_j), float("-inf"), dtype=qdtype, device=dev)
        buf[0, 0].scatter_(1, idx_w, src_w)
        p = F.softmax(buf, dim=-1).nan_to_num(nan=0)
        p = p.view(1, 1, Hkv, G, g1_j).mean(dim=-2)
        far_hi_tok = min(int(nb_l[j]) * bs, g1_j)
        swa_lo_tok_j = max(0, g1_j - swa_tok)
        if far_hi_tok > far_lo_tok:
            i_f2 = torch.topk(p[..., far_lo_tok:far_hi_tok], k2_far_l[j],
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
        m = torch.zeros((1, 1, Hkv, g1_j), dtype=torch.bool, device=dev)
        m.scatter_(-1, i_f2, True)
        m.scatter_(-1, i_n2, True)
        m[..., :sink_tok] = True
        m[..., swa_lo_tok_j:g1_j] = True
        out.append(m[0, 0])
    return out


def bench(fn, iters=7):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


# eager 锚点
t_eager = bench(lambda: l2_loop_args(k_qat, cand_tok, cand_valid, b_q))
masks_eager = l2_loop_args(k_qat, cand_tok, cand_valid, b_q)
print(f"eager L2 整循环 = {t_eager*1e3:7.1f}ms", flush=True)

# ---- CUDA Graph capture ----
# 静态输入（地址稳定，replay 前 copy_ 注入新数据）
k_qat_s = k_qat.clone()
cand_tok_s = cand_tok.clone()
cand_valid_s = cand_valid.clone()
b_q_s = b_q.clone()

# warmup（side stream，官方协议）
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        l2_loop_args(k_qat_s, cand_tok_s, cand_valid_s, b_q_s)
torch.cuda.current_stream().wait_stream(s)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    masks_static = l2_loop_args(k_qat_s, cand_tok_s, cand_valid_s, b_q_s)
print("capture OK", flush=True)

# 硬门 1：同输入 replay vs eager 逐位一致
g.replay()
torch.cuda.synchronize()
diff = sum(1 for a, b in zip(masks_static, masks_eager)
           if not torch.equal(a, b))
print(f"bitexact same-input replay vs eager: diff_rows = {diff}/{N}", flush=True)

# 硬门 2：换输入（不同 seed 走完整管线的真实数据）后 replay vs eager 一致
g2 = torch.Generator(device=DEV).manual_seed(777)
Hh, Hkvh = q.shape[2], k.shape[2]
q2 = torch.randn(1, L, Hh, D, device=DEV, dtype=torch.bfloat16, generator=g2)
k2 = torch.randn(1, L, Hkvh, D, device=DEV, dtype=torch.bfloat16, generator=g2)
idx2 = fresh_idx()
index_dict2 = idx2.prepare_index(k2, cu_full)
k_qat2 = index_dict2["k_qat"][0]
b_q2 = ((q2[0, g1_t - 1] * SS).to(torch.float32)[..., idx_sub]).view(N, Hkv, G, -1)
# L1 用第二组 k 的统计重跑（远/近段分与 k 无关，仅统计不同）
sc_far2 = torch.full((N, H, Tb), float("-inf"), dtype=qdtype, device=dev)
sc_near2 = torch.full((N, H, Tb), float("-inf"), dtype=qdtype, device=dev)
for j in range(N):
    bq = b_q2[j].reshape(H, -1)
    r1 = int(nb_l[j])
    if kf > 0 and r1 > far_lo:
        out = idx2._fast_score_coarse(
            index_dict2["k_min"][0][far_lo:r1], index_dict2["k_max"][0][far_lo:r1],
            bq, qdtype)
        sc_far2[j, :, far_lo:r1] = out
    r0, r1 = int(nb_l[j]), int(sl_l[j])
    if kn > 0 and r1 > r0:
        out = idx2._fast_score_coarse(
            index_dict2["k_min"][0][r0:r1], index_dict2["k_max"][0][r0:r1],
            bq, qdtype)
        sc_near2[j, :, r0:r1] = out
sf2 = sc_far2.view(N, Hkv, G, Tb).mean(dim=2).masked_fill(~m_far, float("-inf"))
sn2 = sc_near2.view(N, Hkv, G, Tb).mean(dim=2).masked_fill(~m_near, float("-inf"))
tv_f2 = torch.topk(sf2, kf, dim=-1)
tv_n2 = torch.topk(sn2, kn, dim=-1)
cand_tok2 = ((torch.cat([tv_f2.indices, tv_n2.indices], dim=-1) * bs)[..., None]
             + torch.arange(bs, device=dev)).reshape(N, Hkv, -1)
cand_valid2 = torch.cat([tv_f2.values > float("-inf"),
                         tv_n2.values > float("-inf")],
                        dim=-1)[..., None].expand(N, Hkv, -1, bs) \
    .reshape(N, Hkv, -1)
assert int(cand_tok2.shape[-1]) == C
masks_eager2 = l2_loop_args(k_qat2, cand_tok2, cand_valid2, b_q2)
k_qat_s.copy_(k_qat2)
cand_tok_s.copy_(cand_tok2)
cand_valid_s.copy_(cand_valid2)
b_q_s.copy_(b_q2)
g.replay()
torch.cuda.synchronize()
diff2 = sum(1 for a, b in zip(masks_static, masks_eager2)
            if not torch.equal(a, b))
print(f"bitexact new-input replay vs eager: diff_rows = {diff2}/{N}", flush=True)

# 计时：replay（不含 copy_ 注入）
t_graph = bench(g.replay)
print(f"graph replay L2 整循环 = {t_graph*1e3:7.1f}ms "
      f"（speedup vs eager = {t_eager/t_graph:.2f}x）", flush=True)
# 计时：copy 注入 + replay（生产口径）
def inject_replay():
    k_qat_s.copy_(k_qat2)
    cand_tok_s.copy_(cand_tok2)
    cand_valid_s.copy_(cand_valid2)
    b_q_s.copy_(b_q2)
    g.replay()
t_ir = bench(inject_replay)
print(f"inject+replay = {t_ir*1e3:7.1f}ms（含 74MB copy_ 注入）", flush=True)
# ---- 分段 graph：L2-A 打分段 / L2-B 复刻选择段 各自 capture，定位 replay 内 GPU 时间构成 ----
def l2_phaseA_args(k_qat_i, cand_tok_i, b_q_i):
    sfs = []
    for j in range(N):
        ct_j = cand_tok_i[j]
        kq_c = k_qat_i[ct_j, h_idx_col]
        kq_c = kq_c.permute(1, 0, 2)
        sf_c = idx._fast_score_fine(kq_c, b_q_i[j].reshape(H, -1), qdtype)
        sfs.append(sf_c)
    return sfs


def l2_phaseB_args(sfs_i, cand_tok_i, cand_valid_i):
    out = []
    for j in range(N):
        g1_j = int(group_ends[j])
        ct_j = cand_tok_i[j]
        valid_j = cand_valid_i[j]
        sf_c = sfs_i[j]
        valid4 = valid_j.view(Hkv, 1, C).expand(Hkv, G, C).reshape(H, C)
        idx4 = ct_j.view(Hkv, 1, C).expand(Hkv, G, C).reshape(H, C)
        idx_w = torch.where(valid4, idx4, torch.zeros_like(idx4))
        src_w = torch.where(valid4, sf_c, sf_c.new_full((), float("-inf")))
        buf = torch.full((1, 1, H, g1_j), float("-inf"), dtype=qdtype, device=dev)
        buf[0, 0].scatter_(1, idx_w, src_w)
        p = F.softmax(buf, dim=-1).nan_to_num(nan=0)
        p = p.view(1, 1, Hkv, G, g1_j).mean(dim=-2)
        far_hi_tok = min(int(nb_l[j]) * bs, g1_j)
        swa_lo_tok_j = max(0, g1_j - swa_tok)
        if far_hi_tok > far_lo_tok:
            i_f2 = torch.topk(p[..., far_lo_tok:far_hi_tok], k2_far_l[j],
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
        m = torch.zeros((1, 1, Hkv, g1_j), dtype=torch.bool, device=dev)
        m.scatter_(-1, i_f2, True)
        m.scatter_(-1, i_n2, True)
        m[..., :sink_tok] = True
        m[..., swa_lo_tok_j:g1_j] = True
        out.append(m[0, 0])
    return out


# A 段 graph
sA = torch.cuda.Stream()
sA.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(sA):
    for _ in range(3):
        l2_phaseA_args(k_qat_s, cand_tok_s, b_q_s)
torch.cuda.current_stream().wait_stream(sA)
gA = torch.cuda.CUDAGraph()
with torch.cuda.graph(gA):
    sfs_static = l2_phaseA_args(k_qat_s, cand_tok_s, b_q_s)
t_gA = bench(gA.replay)
print(f"graph L2-A 打分段 replay = {t_gA*1e3:7.1f}ms（eager 对照 63.2ms）", flush=True)

# B 段 graph（输入 = A 段静态输出）
sB = torch.cuda.Stream()
sB.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(sB):
    for _ in range(3):
        l2_phaseB_args(sfs_static, cand_tok_s, cand_valid_s)
torch.cuda.current_stream().wait_stream(sB)
gB = torch.cuda.CUDAGraph()
with torch.cuda.graph(gB):
    masksB_static = l2_phaseB_args(sfs_static, cand_tok_s, cand_valid_s)
t_gB = bench(gB.replay)
print(f"graph L2-B 复刻选择段 replay = {t_gB*1e3:7.1f}ms", flush=True)
print(f"A+B graph 合计 = {(t_gA+t_gB)*1e3:7.1f}ms（vs 整链 graph {t_graph*1e3:.1f}ms）",
      flush=True)

# B 段逐位自检（graph B vs eager B 同输入）
masksB_eager = l2_phaseB_args(sfs_static, cand_tok_s, cand_valid_s)
gB.replay()
torch.cuda.synchronize()
diffB = sum(1 for a, b in zip(masksB_static, masksB_eager)
            if not torch.equal(a, b))
print(f"B graph bitexact: diff_rows = {diffB}/{N}", flush=True)

# B 段内部再拆：topk 双池 vs softmax/mean（消 topk 的对照——只跑到 p 为止）
def l2_phaseB_pre(sfs_i, cand_tok_i, cand_valid_i):
    """B 段前半：where/scatter/softmax/mean（无 topk/mask）"""
    outs = []
    for j in range(N):
        g1_j = int(group_ends[j])
        ct_j = cand_tok_i[j]
        valid_j = cand_valid_i[j]
        sf_c = sfs_i[j]
        valid4 = valid_j.view(Hkv, 1, C).expand(Hkv, G, C).reshape(H, C)
        idx4 = ct_j.view(Hkv, 1, C).expand(Hkv, G, C).reshape(H, C)
        idx_w = torch.where(valid4, idx4, torch.zeros_like(idx4))
        src_w = torch.where(valid4, sf_c, sf_c.new_full((), float("-inf")))
        buf = torch.full((1, 1, H, g1_j), float("-inf"), dtype=qdtype, device=dev)
        buf[0, 0].scatter_(1, idx_w, src_w)
        p = F.softmax(buf, dim=-1).nan_to_num(nan=0)
        p = p.view(1, 1, Hkv, G, g1_j).mean(dim=-2)
        outs.append(p)
    return outs


sB2 = torch.cuda.Stream()
sB2.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(sB2):
    for _ in range(3):
        l2_phaseB_pre(sfs_static, cand_tok_s, cand_valid_s)
torch.cuda.current_stream().wait_stream(sB2)
gB2 = torch.cuda.CUDAGraph()
with torch.cuda.graph(gB2):
    p_static = l2_phaseB_pre(sfs_static, cand_tok_s, cand_valid_s)
t_gB2 = bench(gB2.replay)
print(f"graph B 前半（softmax/mean，无 topk）= {t_gB2*1e3:7.1f}ms", flush=True)
print(f"graph B 后半（topk+mask）估计 = {(t_gB-t_gB2)*1e3:7.1f}ms", flush=True)
# ---- L1 循环 graph 化（凑齐整链 GPU 地板账） ----
k_min_s = k_min.clone()
k_max_s = k_max.clone()
bq_s = b_q.clone()


def l1_loop_args(k_min_i, k_max_i, bq_i):
    sc_far_q = torch.full((N, H, Tb), float("-inf"), dtype=qdtype, device=dev)
    sc_near_q = torch.full((N, H, Tb), float("-inf"), dtype=qdtype, device=dev)
    for j in range(N):
        bq = bq_i[j].reshape(H, -1)
        r1 = int(nb_l[j])
        if kf > 0 and r1 > far_lo:
            out = idx._fast_score_coarse(
                k_min_i[far_lo:r1], k_max_i[far_lo:r1], bq, qdtype)
            sc_far_q[j, :, far_lo:r1] = out
        r0, r1 = int(nb_l[j]), int(sl_l[j])
        if kn > 0 and r1 > r0:
            out = idx._fast_score_coarse(
                k_min_i[r0:r1], k_max_i[r0:r1], bq, qdtype)
            sc_near_q[j, :, r0:r1] = out
    return sc_far_q, sc_near_q


s1 = torch.cuda.Stream()
s1.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s1):
    for _ in range(3):
        l1_loop_args(k_min_s, k_max_s, bq_s)
torch.cuda.current_stream().wait_stream(s1)
g1_ = torch.cuda.CUDAGraph()
with torch.cuda.graph(g1_):
    sc_f_s, sc_n_s = l1_loop_args(k_min_s, k_max_s, bq_s)
t_g1 = bench(g1_.replay)
print(f"graph L1 循环 replay = {t_g1*1e3:7.1f}ms（eager 对照 108.8ms）", flush=True)
# L1 bitexact 自检
sf_e, sn_e = l1_loop_args(k_min, k_max, b_q)
k_min_s.copy_(k_min)
k_max_s.copy_(k_max)
bq_s.copy_(b_q)
g1_.replay()
torch.cuda.synchronize()
ok1 = torch.equal(sc_f_s, sf_e) and torch.equal(sc_n_s, sn_e)
print(f"L1 graph bitexact: {ok1}", flush=True)
print(f"整链 GPU 地板估计：prep 2.0 + L1 {t_g1*1e3:.1f} + L2 {t_graph*1e3:.1f} "
      f"= {(2.0 + t_g1*1e3 + t_graph*1e3):.1f}ms（eager 423.3ms）", flush=True)
print("=== DONE ===", flush=True)
