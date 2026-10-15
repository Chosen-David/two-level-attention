# -*- coding: utf-8 -*-
"""E139 纲领④ 对拍硬门：融合紧凑 attention Triton kernel（fused_attn_triton）。

硬门（不过不许交付）：
  G1 输出级：L∈{8192, 32768}、grp=64、K2=1024、mavg_tail、bf16，fused vs
     原版 allclose(rtol=1e-2, atol=1e-3) + maxdiff 报告（E138b 口径）+
     NaN 检查；fp16 抽查档。
  G2 行级有效列集合恒等（语义硬门）：合成 chunk（含块对齐 sel / 逐 token
     非对齐 sel / 空 sel 组三形态），原实现 row_ok（复制 eager_prefill
     L156-255 组版代码重建 dense mask）vs kernel 判定谓词 dense mask，
     须逐位相等；另以谓词 oracle 朴素 attention 对拍 fused 输出（消除
     「谓词公式与 kernel 实现不一致」的循环论证盲区）。
  G3 因果探针：篡改后半 K/V（组边界对齐 16384 + 组内错位 16416 双档），
     篡改边界前行输出须逐位一致（因果零泄漏；选择侧 far 池块 < near_blks
     均在篡改前，swa 强制区 ⊆ 强制选择——选择集不变）。
  G4 回归零扰动：env 默认关时 test_sparse_prefill_impl 7/7 +
     /tmp/e138b_prefill_bitdiff.py 17/17 + exp/bench/e138_grp_bitdiff.py 5/5。
  G5 wall bench：L=32K grp64 组合态（批量选择×融合 attention vs 各关闭态）
     7-iter 中位锁卡 + dense SDPA 参照。
"""
import os
import subprocess
import sys
import time

TREE = "/tmp/wt_e138_ablation"
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "exp", "bench"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from einops import rearrange, einsum  # noqa: E402
from e131_arm_bench import make_args  # noqa: E402
from sparse_attn.indexer.tli_indexer import TLIIndexer  # noqa: E402
from sparse_attn.ops.eager_prefill import sparse_prefill_attn  # noqa: E402

DEV = "cuda:0"
FAILS = []


def report(name, ok, detail=""):
    FAILS.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""),
          flush=True)


def make_qkv(L, seed=999, dtype=torch.bfloat16):
    g = torch.Generator(device=DEV).manual_seed(seed)
    H, Hkv, D = 32, 8, 128
    return (torch.randn(1, L, H, D, device=DEV, dtype=dtype, generator=g),
            torch.randn(1, L, Hkv, D, device=DEV, dtype=dtype, generator=g),
            torch.randn(1, L, Hkv, D, device=DEV, dtype=dtype, generator=g))


def fresh():
    idx = TLIIndexer(make_args("mavg_tail"))
    idx.layer_idx = 2
    idx.clear()
    return idx


def run_prefill(idx, q, k, v, grp=64, fused=True, batch=False):
    os.environ["TLI_SPARSE_PREFILL_CHUNK"] = "2048"
    os.environ["TLI_PREFILL_SELECT_GROUP"] = str(grp)
    os.environ["TLI_PREFILL_BATCH_SELECT"] = "1" if batch else "0"
    os.environ["TLI_PREFILL_FUSED_ATTN"] = "1" if fused else "0"
    os.environ["TLI_PREFILL_PREFETCH"] = "0"
    return sparse_prefill_attn(idx, q, k, v)


# ================================================================ G1 输出级
print("=== G1 输出级对拍（L∈{8192,32768} grp=64 K2=1024 mavg_tail）===", flush=True)
for L, dtype, tag in [(8192, torch.bfloat16, "bf16"), (32768, torch.bfloat16, "bf16"),
                      (8192, torch.float16, "fp16抽查")]:
    q, k, v = make_qkv(L, dtype=dtype)
    o_ref = run_prefill(fresh(), q, k, v, fused=False)
    o_new = run_prefill(fresh(), q, k, v, fused=True)
    ok = torch.allclose(o_ref.float(), o_new.float(), rtol=1e-2, atol=1e-3)
    md = (o_ref.float() - o_new.float()).abs().max().item()
    nan = int(torch.isnan(o_new.float()).sum().item())
    report(f"G1 L={L} {tag}", ok and nan == 0, f"maxdiff={md:.3e} nan={nan}")
    del q, k, v, o_ref, o_new
    torch.cuda.empty_cache()

# ================================================================ G2 行级有效列集合恒等
print("=== G2 行级有效列集合恒等（语义硬门）===", flush=True)


def pred_dense_mask(sel_list, pos_lo, len_r, grp, T, Hkv, sink_tok, swa_tok):
    """kernel 判定谓词 dense mask [len_r, Hkv, T]：
    j <= p 且 (sel[j(r)//grp][h, j] | j < sink | j >= p-swa+1)；空行 → {p}。"""
    selp = []
    for s in sel_list:
        if s.shape[-1] < T:
            pad = torch.zeros((*s.shape[:-1], T - s.shape[-1]),
                              dtype=torch.bool, device=s.device)
            s = torch.cat([s, pad], dim=-1)
        selp.append(s)
    m = torch.zeros(len_r, Hkv, T, dtype=torch.bool, device=DEV)
    kcol = torch.arange(T, device=DEV)
    for r in range(len_r):
        p = pos_lo + r
        extra = (kcol < sink_tok) | (kcol >= p - swa_tok + 1)
        row = (selp[r // grp] | extra[None, :]) & (kcol <= p)[None, :]
        if not bool(row.any(dim=-1).all()):
            empty = ~row.any(dim=-1)
            row[empty, p] = True
        m[r] = row
    return m


def orig_row_ok_dense(q_c, k_c, sel_list, pos_lo, G, sink_tok, swa_tok, grp,
                      w_sel_cap=0):
    """原实现 row_ok 重建（复制 eager_prefill 组版代码），scatter 回 dense。"""
    len_r, H, D = q_c.shape[1], q_c.shape[2], q_c.shape[3]
    T, Hkv = k_c.shape[1], k_c.shape[2]
    device = q_c.device
    rhg = Hkv
    sels = []
    for s in sel_list:
        s = s[..., : T]
        if s.shape[-1] < T:
            pad = torch.zeros((*s.shape[:-1], T - s.shape[-1]),
                              dtype=torch.bool, device=s.device)
            s = torch.cat([s, pad], dim=-1)
        sels.append(s)
    sel_stack = torch.stack(sels)
    sel_e = sel_stack.any(dim=0)
    from sparse_attn.ops.eager_prefill import _row_block_size
    if w_sel_cap > 0:
        w_sel_max = int(w_sel_cap)
    else:
        w_sel_max = int(sel_e[:, : T].sum(dim=-1).max().item())
    R = _row_block_size(rhg, w_sel_max, swa_tok, sink_tok)
    R = max(grp, (R // grp) * grp)
    dense = torch.zeros(len_r, Hkv, T, dtype=torch.bool, device=device)
    hh = torch.arange(Hkv, device=device)
    for r0 in range(0, len_r, R):
        r1 = min(r0 + R, len_r)
        p_first = pos_lo + r0
        p_max = pos_lo + r1 - 1
        vis = sel_e[:, : p_max + 1].clone()
        vis[:, : min(sink_tok, p_max + 1)] = True
        prot_lo = max(0, p_first - swa_tok + 1)
        if swa_tok > 0:
            vis[:, prot_lo:] = True
        else:
            vis[:, p_first : p_max + 1] = True
        order = torch.argsort(~vis, dim=-1, stable=True)
        counts = vis.sum(dim=-1)
        W = min(int(counts.max().item()), p_max + 1)
        idx = order[:, :W]
        pos_w = idx
        n_g_blk = (r1 - r0 + grp - 1) // grp
        j0 = r0 // grp
        in_sel = torch.empty((r1 - r0, rhg, W), dtype=torch.bool, device=device)
        for j in range(n_g_blk):
            rs = j * grp
            re = min(rs + grp, r1 - r0)
            sel_j = sel_stack[j0 + j][:, : p_max + 1]
            in_sel[rs:re] = torch.gather(sel_j, 1, idx)
        valid = torch.arange(W, device=device)[None, :] < counts[:, None]
        p_r = torch.arange(pos_lo + r0, pos_lo + r1, device=device)
        pos_ok = pos_w[None, :, :] <= p_r[:, None, None]
        row_ok = (
            pos_ok
            & (in_sel
               | (pos_w[None, :, :] < sink_tok)
               | (pos_w[None, :, :] >= (p_r[:, None, None] - swa_tok + 1)))
            & valid[None, :, :]
        )
        diag = (pos_w[None, :, :] == p_r[:, None, None]) & valid[None, :, :]
        empty = ~row_ok.any(dim=-1)
        row_ok = row_ok | (diag & empty[..., None])
        rrb = torch.arange(r0, r1, device=device)
        dense[rrb[:, None, None], hh[None, :, None],
              idx[None, :, :].expand(r1 - r0, rhg, W)] = row_ok
    return dense


def naive_group_attn(q_c, k_c, v_c, mask, pos_lo, G, scale):
    """谓词 oracle 朴素 attention（fp32 全量，mask [len_r, Hkv, T]）。"""
    len_r, H, D = q_c.shape[1], q_c.shape[2], q_c.shape[3]
    T, Hkv = k_c.shape[1], k_c.shape[2]
    qf = rearrange(q_c[0] * scale, "r (h g) d -> r h g d", g=G).to(torch.float32)
    kf = k_c[0].to(torch.float32)
    vf = v_c[0].to(torch.float32)
    m = mask[:, :, None, :].expand(len_r, Hkv, G, T).reshape(len_r, H, T)
    o = torch.empty_like(q_c)
    for r0 in range(0, len_r, 512):
        r1 = min(r0 + 512, len_r)
        b_s = einsum(qf[r0:r1], kf, "r h g d, t h d -> r h g t")
        b_s = b_s.masked_fill(~m[r0:r1].view(r1 - r0, Hkv, G, T), float("-inf"))
        b_p = F.softmax(b_s, dim=-1)
        b_o = einsum(b_p, vf, "r h g t, t h d -> r h g d")
        o[0, r0:r1] = rearrange(b_o, "r h g d -> r (h g) d").to(q_c.dtype)
    return o


from sparse_attn.ops.fused_attn_triton import fused_group_attn  # noqa: E402


def make_group_sel(g1, Hkv, style, seed):
    """组末行选择 [Hkv, g1]：block 对齐（生产口径）/ 逐 token 随机 / 全空。"""
    g = torch.Generator(device=DEV).manual_seed(seed)
    if style == "empty":
        return torch.zeros(Hkv, g1, dtype=torch.bool, device=DEV)
    if style == "aligned":
        bs = 64
        sel = torch.zeros(Hkv, g1, dtype=torch.bool, device=DEV)
        nb = (g1 + bs - 1) // bs       # 尾组 g1 可非 64 倍数（尾块混合）
        sink_blk = 128 // bs
        swa_blk = 128 // bs
        gc = torch.Generator().manual_seed(seed)  # randperm 不支持 cuda gen
        for h in range(Hkv):
            blk = torch.randperm(nb - sink_blk - swa_blk, generator=gc)[:16] \
                + sink_blk
            selb = torch.zeros(nb, dtype=torch.bool)
            selb[:sink_blk] = True
            selb[blk] = True
            selb[nb - swa_blk:] = True
            sel[h] = selb.repeat_interleave(bs)[:g1].to(DEV)
        return sel
    sel = torch.rand(Hkv, g1, generator=g, device=DEV) < 0.04
    sel[:, :128] = True
    sel[:, g1 - 128:] = True
    return sel


H, Hkv, D, G, GRP = 32, 8, 128, 4, 64
for style in ("aligned", "random", "empty"):
    pos_lo, len_r = 2048, 1000       # 尾组不满（1000 = 15.625 组）
    T = pos_lo + len_r               # 3048
    sink_tok, swa_tok = (128, 128) if style != "empty" else (0, 0)
    q_c = torch.randn(1, len_r, H, D, device=DEV, dtype=torch.bfloat16)
    k_c = torch.randn(1, T, Hkv, D, device=DEV, dtype=torch.bfloat16)
    v_c = torch.randn(1, T, Hkv, D, device=DEV, dtype=torch.bfloat16)
    sel_list = [make_group_sel(pos_lo + min((j + 1) * GRP, len_r), Hkv,
                               style, 100 + j) for j in range((len_r + GRP - 1) // GRP)]
    scale = D ** -0.5
    # ① dense mask 恒等：原 row_ok vs kernel 谓词
    d_ref = orig_row_ok_dense(q_c, k_c, sel_list, pos_lo, G, sink_tok, swa_tok, GRP)
    d_pred = pred_dense_mask(sel_list, pos_lo, len_r, GRP, T, Hkv, sink_tok, swa_tok)
    same = torch.equal(d_ref, d_pred)
    report(f"G2 集合恒等 [{style}]", same,
           f"differ={int((d_ref ^ d_pred).sum())} cells")
    # ② 谓词 oracle 输出对拍 fused（消除谓词↔kernel 实现不一致盲区）
    os.environ["TLI_PREFILL_FUSED_ATTN"] = "1"
    o_f = fused_group_attn(q_c, k_c, v_c, sel_list, pos_lo, G, scale,
                           sink_tok=sink_tok, swa_tok=swa_tok, grp=GRP)
    o_or = naive_group_attn(q_c, k_c, v_c, d_pred, pos_lo, G, scale)
    ok = torch.allclose(o_or.float(), o_f.float(), rtol=1e-2, atol=1e-3)
    md = (o_or.float() - o_f.float()).abs().max().item()
    nan = int(torch.isnan(o_f.float()).sum().item())
    report(f"G2 oracle 对拍 [{style}]", ok and nan == 0,
           f"maxdiff={md:.3e} nan={nan}")
    del q_c, k_c, v_c, sel_list, d_ref, d_pred, o_f, o_or
    torch.cuda.empty_cache()

# ================================================================ G3 因果探针
print("=== G3 因果探针（篡改后半 K/V）===", flush=True)
L = 32768
q, k, v = make_qkv(L, seed=777)
for t0, tag in [(16384, "组边界对齐"), (16416, "组内错位")]:
    g2 = torch.Generator(device=DEV).manual_seed(31337)
    k_t = k.clone()
    v_t = v.clone()
    k_t[0, t0:] = torch.randn(L - t0, Hkv, D, device=DEV, dtype=k.dtype, generator=g2)
    v_t[0, t0:] = torch.randn(L - t0, Hkv, D, device=DEV, dtype=v.dtype, generator=g2)
    o_ref = run_prefill(fresh(), q, k, v, fused=True)
    o_tam = run_prefill(fresh(), q, k_t, v_t, fused=True)
    d = (o_ref[0, :t0].float() - o_tam[0, :t0].float()).abs()
    md = d.max().item()
    nrow = int((d.amax(dim=-1) > 0).any(dim=-1).sum().item())
    # 跨界组（组末 > t0 的最早组）内的行：选择读 tampered 前缀属组共享
    # 语义固有（far 池块 < near_blks ≤ t0、swa 强制区块对齐强制选中 →
    # 选择集不变；若选择翻转会在本门暴露）
    report(f"G3 因果探针 [篡改@{t0} {tag}]", md == 0.0,
           f"前段 maxdiff={md:.3e} diff 行数={nrow}")
    del k_t, v_t, o_ref, o_tam, d
    torch.cuda.empty_cache()
del q, k, v
torch.cuda.empty_cache()

# ================================================================ G4 回归零扰动
print("=== G4 回归零扰动（env 默认关）===", flush=True)
env = dict(os.environ)
env.pop("TLI_PREFILL_FUSED_ATTN", None)
env.pop("TLI_PREFILL_BATCH_SELECT", None)
env.pop("TLI_PREFILL_SELECT_GROUP", None)
env.pop("TLI_SPARSE_PREFILL_CHUNK", None)
env.pop("TLI_PREFILL_PREFETCH", None)
for name, cmd, want in [
    ("test_sparse_prefill_impl 7/7", [sys.executable, "test_sparse_prefill_impl.py"], "7/7 PASS"),
    ("e138b_prefill_bitdiff 17/17", [sys.executable, "/tmp/e138b_prefill_bitdiff.py"], "ALL 17 PASS"),
    ("e138_grp_bitdiff 5/5", [sys.executable, "exp/bench/e138_grp_bitdiff.py"], "=== 5/5 ==="),
]:
    p = subprocess.run(cmd, cwd=TREE, env=env, capture_output=True, text=True,
                       timeout=1200)
    ok = (p.returncode == 0) and (want in p.stdout)
    report(f"G4 {name}", ok, f"rc={p.returncode}")
    if not ok:
        print(p.stdout[-2000:], p.stderr[-2000:])

# ================================================================ G5 wall bench
print("=== G5 wall bench（L=32K grp64，7-iter 中位，锁卡）===", flush=True)


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


L = 32768
q, k, v = make_qkv(L)


def combo(batch, fused):
    return bench(lambda: run_prefill(fresh(), q, k, v, fused=fused, batch=batch))


t_orig = combo(False, False)
t_bs = combo(True, False)
t_fu = combo(True, True)
qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
td = bench(lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=True,
                                                   enable_gqa=True))
print(f"逐组+原attention={t_orig*1e3:.1f}ms 批量+原attention={t_bs*1e3:.1f}ms "
      f"批量+融合={t_fu*1e3:.1f}ms dense={td*1e3:.1f}ms", flush=True)
print(f"加速：vs 逐组 {t_orig/t_fu:.2f}x | vs 批量 {t_bs/t_fu:.2f}x | "
      f"vs dense {td/t_fu:.2f}x", flush=True)
report("G5 wall 目标 ≤300ms", t_fu * 1e3 <= 300, f"组合态 {t_fu*1e3:.1f}ms")

print(f"\n===== 总门：{sum(1 for _, ok in FAILS if ok)}/{len(FAILS)} PASS =====")
sys.exit(0 if all(ok for _, ok in FAILS) else 1)
