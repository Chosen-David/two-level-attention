# -*- coding: utf-8 -*-
"""E139 #253 L2 循环子段分项计时：312.5ms 拆为 打分段（gather+fine kernel）
vs 复刻选择段（where/scatter/softmax/mean/topk/mask），L∈{32K,64K} 双档
判定增长律（线性 vs O(L²)）——决定批量化主攻方向。
协议：7-iter 中位、warmup 2、fresh indexer、单流，克隆 e139_selchain_inner_seg。
拆分语义自检：A+B 两段产物 mask 与原 l2_loop 逐位一致（拆分不改变数值树）。
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


gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
print(f"=== E139 L2 子段分项 L={L} GPU{gpu} ===", flush=True)
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
k_min = index_dict["k_min"][0]
k_max = index_dict["k_max"][0]
k_avg = index_dict["k_avg"][0]
k_qat = index_dict["k_qat"][0]
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
far_use_mm = not (idx.far_method == "avg" and k_avg is not None)
near_use_mm = (idx.near_method == "minmax" or k_avg is None)
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
            if far_use_mm:
                out = idx._fast_score_coarse(
                    k_min[far_lo:r1], k_max[far_lo:r1], bq, qdtype)
            else:
                out = idx._fast_score_fine(k_avg[far_lo:r1], bq, qdtype)
            sc_far_q[j, :, far_lo:r1] = out
        r0, r1 = int(nb_l[j]), int(sl_l[j])
        if kn > 0 and r1 > r0:
            if near_use_mm:
                out = idx._fast_score_coarse(
                    k_min[r0:r1], k_max[r0:r1], bq, qdtype)
            else:
                out = idx._fast_score_fine(k_avg[r0:r1], bq, qdtype)
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
cand_blk = torch.cat([i_f, i_n], dim=-1)
cand_valid_blk = torch.cat([v_f > float("-inf"), v_n > float("-inf")], dim=-1)
C_blk = int(cand_blk.shape[-1])
C = C_blk * bs
cand_tok = (cand_blk * bs)[..., None] + torch.arange(bs, device=dev)
cand_tok = cand_tok.reshape(N, Hkv, C)
cand_valid = cand_valid_blk[..., None].expand(N, Hkv, C_blk, bs) \
    .reshape(N, Hkv, C)
h_idx_col = torch.arange(Hkv, device=dev).view(Hkv, 1)
K2arg = int(idx.args.tia_level2_topk)
K2 = torch.minimum(g1_t, torch.full_like(g1_t, K2arg))
far_tok_lo = far_lo * bs
far_tok_hi = torch.minimum(near_blks * bs, g1_t)
K2_mid = (K2 - sink_tok - torch.minimum(g1_t, torch.full_like(g1_t, swa_tok))
          ).clamp(min=0)
nt_near = torch.minimum(
    torch.full_like(K2_mid, int(nb_near * bs * float(idx.gamma))), K2_mid)
far_budget = (K2_mid - nt_near).clamp(min=0)
far_width = (far_tok_hi - far_tok_lo).clamp(min=0)
k2_far = torch.minimum(torch.minimum(far_budget, far_width), K2_mid)
k2_near = (K2_mid - k2_far).clamp(min=0)
k2_far_l = k2_far.tolist()
k2_near_l = k2_near.tolist()
far_lo_tok = far_lo * bs

# ---- L2 原版整循环（锚点 + 拆分自检参照） ----
def l2_loop():
    out = []
    for j in range(N):
        g1_j = int(group_ends[j])
        ct_j = cand_tok[j]
        valid_j = cand_valid[j]
        kq_c = k_qat[ct_j, h_idx_col]
        kq_c = kq_c.permute(1, 0, 2)
        sf_c = idx._fast_score_fine(kq_c, b_q[j].reshape(H, -1), qdtype)
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

# ---- L2 拆段 A：打分段（gather + 同一 fine kernel）----
def l2_phaseA():
    sfs = []
    for j in range(N):
        ct_j = cand_tok[j]
        kq_c = k_qat[ct_j, h_idx_col]
        kq_c = kq_c.permute(1, 0, 2)
        sf_c = idx._fast_score_fine(kq_c, b_q[j].reshape(H, -1), qdtype)
        sfs.append(sf_c)
    return sfs

# ---- L2 拆段 B：复刻选择段（where/buf/scatter/softmax/mean/topk/mask）----
def l2_phaseB(sfs):
    out = []
    for j in range(N):
        g1_j = int(group_ends[j])
        ct_j = cand_tok[j]
        valid_j = cand_valid[j]
        sf_c = sfs[j]
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

# ---- 计时 ----
t_l2 = bench(l2_loop)
sfs_ref = l2_phaseA()
masks_ref = l2_phaseB(sfs_ref)
masks_orig = l2_loop()
# 拆分自检：A+B 与原版逐位一致（拆分不改变数值树）
assert len(masks_ref) == len(masks_orig)
diff = sum(1 for a, b in zip(masks_ref, masks_orig)
           if not torch.equal(a, b))
print(f"split_selfcheck diff_rows = {diff}/{N}", flush=True)

t_a = bench(l2_phaseA)
sfs = l2_phaseA()
t_b = bench(lambda: l2_phaseB(sfs))

print(f"L2 整循环     = {t_l2*1e3:7.1f}ms", flush=True)
print(f"L2-A 打分段   = {t_a*1e3:7.1f}ms（gather+fine kernel ×{N} 行）", flush=True)
print(f"L2-B 复刻选择 = {t_b*1e3:7.1f}ms（where/scatter/softmax/mean/topk/mask ×{N} 行）", flush=True)
print(f"A+B 合计      = {(t_a+t_b)*1e3:7.1f}ms", flush=True)
print("=== DONE ===", flush=True)
