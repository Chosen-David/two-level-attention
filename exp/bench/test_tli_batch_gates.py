#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 1 门禁套件：TLI 主方法 batch>1 decode（deepseek 轨）。

用法（仓根）：
  CUDA_VISIBLE_DEVICES=0 python3 exp/bench/test_tli_batch_gates.py

门禁（对应计划「阶段 1 验收」4 项，单元级——不加载模型，合成 KV 直测
indexer + eager 消费端）：
  G1 gate_bs1_equiv：batch=1 的 _tl_decode_batched == 逐行单序列核心
     （prepare_mask + eager_decoding_attn）逐位（==）——batch 分支 B=1
     退化与旧单序列入口逐位一致。
  G3 gate_multiseq_consistency：B=4 混合长度多步 decode，batch 分支逐行
     输出 == 同样本单序列核心逐位（==）（fixed seed）。
  G2 gate_anchor_batch：预算拉满==dense 锚点（batch 逐行 mask 全连）+
     候选化开关（TLI_FINE_GATHER 0/1）mask 逐位一致（阶段 0 口径复用）。
  G4 gate_full_vs_inc：全量重建 == 单序列增量路径（E133c 逐位依据）——
     batch 全量重建与 pred_ruler.py 单序列增量语义一致的根基。

right-pad 协议（pred_ruler_batch left-pad）：KV cache 行 b 有效列占尾部
seq_lens[b]、pad 前缀在前。逐行切 [b:b+1, -L_b:] 即完整单序列 KV。
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


def _make_cfg(alpha, beta, gamma, far, near):
    from exp.bench.e127_stage_bench import make_args
    a = make_args(alpha, beta, gamma)
    a.tli_far_method, a.tli_near_method = far, near
    return a


CONFIGS = [
    ("champion_e64", _make_cfg(0.25, 0.125, 0.625, "minmax", "avg")),
    ("singlepool_00", _make_cfg(0.0, 0.0, 0.625, "minmax", "minmax")),
    ("aavg_e64", _make_cfg(0.125, 0.125, 0.375, "avg", "avg")),
    ("aavg_00", _make_cfg(0.0, 0.0, 0.375, "avg", "avg")),
]


class _StubAttn:
    """_tl_decode_batched 需要的属性（self.scaling + self.indexer）。"""
    def __init__(self, indexer, scale):
        self.indexer = indexer
        self.scaling = scale


def _right_pad_kv(S_total, seq_lens, device, seed):
    """构造 [B, S_total, Hkv, D] seq-first right-pad KV。

    行 b 第 j 个真实 token（j=0..L-1）放列 S_total-L+j；pad 前缀列填零
    （被有效列切片排除，值无关）。每行独立 seed 流（行内 token 序列
    与同 seed 单序列构造一致）。
    """
    from sparse_attn.baseline_runtime import BatchCtx  # noqa: F401（docstring 契约）
    B = len(seq_lens)
    Hkv, D = 8, 128
    g = torch.Generator().manual_seed(seed)
    k = torch.zeros(B, S_total, Hkv, D, dtype=torch.bfloat16, device=device)
    v = torch.zeros_like(k)
    for b, L in enumerate(seq_lens):
        base = S_total - L
        k[b, base:] = torch.randn(1, L, Hkv, D, generator=g,
                                  dtype=torch.bfloat16).to(device)[0]
        v[b, base:] = torch.randn(1, L, Hkv, D, generator=g,
                                  dtype=torch.bfloat16).to(device)[0]
    return k, v


def _run_row_single(idx, q_b, k_b, v_b, scale):
    """逐行单序列核心调用序列（与 pred_ruler.py decode 分支一致）。"""
    cu = torch.tensor([0, k_b.shape[1]], device=k_b.device, dtype=torch.int32)
    pos = torch.tensor([k_b.shape[1] - 1], device=k_b.device, dtype=torch.long)
    idx.clear()
    mask, bs = idx.prepare_mask(q_b, pos, k_b, cu, softmax_scale=scale)
    from sparse_attn.ops.eager_decoding import eager_decoding_attn
    return eager_decoding_attn(q_b, k_b, v_b, mask, bs, cu,
                               softmax_scale=scale)


def _run_batch(idx, q_hf, k_hf, v_hf, ctx, scale):
    from sparse_attn.patches.qwen3_attn_patch import _tl_decode_batched
    from sparse_attn.baseline_runtime import BatchCtx  # noqa: F401
    return _tl_decode_batched(_StubAttn(idx, scale), idx, q_hf, k_hf, v_hf,
                              ctx)


# ------------------------------------------------------------- G1 + G3
def test_batch_rows_equiv():
    """G1+G3：batch 分支 vs 逐行单序列核心，B=1 与 B=4 多步逐位一致。

    覆盖 4 配置 × normal/skip 层 × 3 步 decode（每步全 batch append 1 列、
    seq_lens+1）。batch 分支输入形状流与 qwen3_attn_forward decode 入口
    一致：q_hf [B,H,1,D]、k_hf/v_hf [B,Hkv,S_total,D]（cache.update 后）。
    """
    import torch as T
    from exp.bench.e127_stage_bench import make_args  # noqa: F401
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    from sparse_attn.baseline_runtime import BatchCtx

    dev = T.device('cuda:0')
    H, Hkv, D = 32, 8, 128
    scale = D ** -0.5
    SEED = 20261020

    for cname, args in CONFIGS:
        for layer_idx, ltag in [(2, "normal"), (0, "skip")]:
            # ---- B=1（G1）----
            seq_lens1 = [48]
            S1 = 48
            k1, v1 = _right_pad_kv(S1, seq_lens1, dev, SEED)
            q1 = T.randn(1, 1, H, D, device=dev, dtype=T.bfloat16)
            idx = TLIIndexer(args); idx.layer_idx = layer_idx; idx.clear()
            q_hf1 = q1.transpose(1, 2)                     # [1,H,1,D]
            k_hf1 = k1.transpose(1, 2).contiguous()        # [1,Hkv,S,D]
            v_hf1 = v1.transpose(1, 2).contiguous()
            ctx1 = BatchCtx(batch=1, seq_lens=seq_lens1,
                            gen_lens=[0], alive=[True], pad_id=0)
            o_b1 = _run_batch(idx, q_hf1, k_hf1, v_hf1, ctx1, scale)
            o_s1 = _run_row_single(idx, q1, k1[0:1], v1[0:1], scale)
            eq1 = T.equal(o_b1, o_s1)
            check(f"G1 batch_rows_equiv {cname} {ltag} B=1",
                  eq1, f"nd={int((o_b1 != o_s1).sum().item()) if not eq1 else 0}")

            # ---- B=4 多步（G3）----
            seq_lens = [16, 32, 48, 64]
            S0 = max(seq_lens)
            k, v = _right_pad_kv(S0, seq_lens, dev, SEED)
            kk, vv = k.clone(), v.clone()
            all_ok = True
            for step in range(3):
                # 全 batch append 1 列（dead 行喂 filler，此处全 alive）
                new_k = T.randn(4, 1, Hkv, D, device=dev, dtype=T.bfloat16)
                new_v = T.randn(4, 1, Hkv, D, device=dev, dtype=T.bfloat16)
                kk = T.cat([kk, new_k], dim=1)
                vv = T.cat([vv, new_v], dim=1)
                seq_lens = [L + 1 for L in seq_lens]       # forward 前 +1
                S_now = kk.shape[1]
                # 每行当前 query（行内最后真实 token 的 q）
                qs = [T.randn(1, 1, H, D, device=dev, dtype=T.bfloat16)
                      for _ in seq_lens]
                idx.clear()
                q_hf = T.stack([q.transpose(1, 2)[0] for q in qs], dim=0)
                k_hf = kk.transpose(1, 2).contiguous()
                v_hf = vv.transpose(1, 2).contiguous()
                ctx = BatchCtx(batch=4, seq_lens=seq_lens,
                               gen_lens=[0]*4, alive=[True]*4, pad_id=0)
                o_b = _run_batch(idx, q_hf, k_hf, v_hf, ctx, scale)
                o_s = []
                for b, L in enumerate(seq_lens):
                    o_s.append(_run_row_single(
                        idx, qs[b], kk[b:b+1, -L:], vv[b:b+1, -L:], scale))
                o_s = T.cat(o_s, dim=0)
                if not T.equal(o_b, o_s):
                    all_ok = False
                    break
            check(f"G3 multiseq_consistency {cname} {ltag} B=4×3step",
                  all_ok, "逐行 vs 单序列核心逐位")


# ------------------------------------------------------------- G2 锚点
def test_anchor_batch():
    """G2：预算拉满==dense 锚点（batch 逐行 mask 全连）+ 候选化开关
    TLI_FINE_GATHER 0/1 逐位一致（阶段 0 口径在 batch 路径复用）。
    """
    import torch as T
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    from sparse_attn.baseline_runtime import BatchCtx

    dev = T.device('cuda:0')
    H, Hkv, D = 32, 8, 128
    scale = D ** -0.5
    SEED = 20261020
    seq_lens = [16, 32, 48, 64]
    S_total = max(seq_lens)

    # q 用固定 seed 生成（两臂须逐位同输入；全局 RNG 在两次 run 间会漂移）
    g = T.Generator().manual_seed(SEED)
    qs = [T.randn(1, 1, H, D, generator=g, dtype=T.bfloat16).to(dev)
          for _ in seq_lens]
    q_hf = T.stack([q.transpose(1, 2)[0] for q in qs], dim=0)
    k, v = _right_pad_kv(S_total, seq_lens, dev, SEED)
    k_hf = k.transpose(1, 2).contiguous()
    v_hf = v.transpose(1, 2).contiguous()

    def run(gather_on):
        os.environ["TLI_FINE_GATHER"] = "1" if gather_on else "0"
        args = _make_cfg(0.25, 0.125, 0.625, "minmax", "avg")
        args.tia_level1_topk = 999999
        args.tia_level2_topk = 999999
        idx = TLIIndexer(args); idx.layer_idx = 2; idx.clear()
        ctx = BatchCtx(batch=4, seq_lens=seq_lens,
                       gen_lens=[0]*4, alive=[True]*4, pad_id=0)
        return _run_batch(idx, q_hf, k_hf, v_hf, ctx, scale)

    o_c = run(True)
    o_f = run(False)
    eq = T.equal(o_c, o_f)
    check("G2 anchor_batch: TLI_FINE_GATHER 0/1 逐位一致", eq,
          f"nd={int((o_c != o_f).sum().item()) if not eq else 0}")

    # mask 全连（dense 锚点语义：预算拉满下每行全覆盖；mask 为 token 级
    # [1,1,Hkv,T]，全连元素数 = Hkv×T = mask.numel()）
    args = _make_cfg(0.25, 0.125, 0.625, "minmax", "avg")
    args.tia_level1_topk = 999999
    args.tia_level2_topk = 999999
    idx = TLIIndexer(args); idx.layer_idx = 2; idx.clear()
    k, v = _right_pad_kv(S_total, seq_lens, dev, SEED)
    cu = T.tensor([0, S_total], device=dev, dtype=T.int32)
    pos = T.tensor([S_total - 1], device=dev, dtype=T.long)
    mask, _bs = idx.prepare_mask(qs[0], pos, k[0:1], cu, softmax_scale=scale)
    n_all = mask.numel()
    n_true = int(mask.sum().item())
    check("G2 anchor_batch: 预算拉满==dense 锚点（最短行 mask 全连）",
          n_true == n_all, f"true={n_true}/{n_all} shape={tuple(mask.shape)}")


# ------------------------------------------------------------- G4 全量 vs 增量
def test_full_vs_inc():
    """G4：全量重建 == 单序列增量路径（E133c 逐位依据）。

    batch 分支每行走 prepare_mask → prepare_index 全量重建；pred_ruler.py
    单序列 decode 走 _prepare_index_inc 增量。两者 k_min/k_max/k_qat 逐位
    一致 → compute_score/compute_mask 逐位一致 → gate_bs1_equiv 成立。
    """
    import torch as T
    from sparse_attn.indexer.tli_indexer import TLIIndexer

    dev = T.device('cuda:0')
    H, Hkv, D = 32, 8, 128
    scale = D ** -0.5
    SEED = 20261020

    for cname, args in CONFIGS:
        for layer_idx, ltag in [(2, "normal"), (0, "skip")]:
            # 单序列 decode 模拟：S0 冷启动建 _inc，3 步 append 走增量
            g = T.Generator().manual_seed(SEED)
            k_full = T.randn(1, 96, Hkv, D, generator=g,
                             dtype=T.bfloat16).to(dev)
            idx = TLIIndexer(args); idx.layer_idx = layer_idx; idx.clear()
            S0 = 64
            k0 = k_full[:, :S0]
            cu0 = T.tensor([0, S0], device=dev, dtype=T.int32)
            d_inc0 = idx.prepare_index(k0, cu0)          # 冷启动全量建 _inc
            # 增量推进到 S1=80（3 次 append × ~5 列）
            k_inc = k0
            for _ in range(3):
                k_inc = T.cat([k_inc, k_full[:, S0:S0+5]], dim=1)
                S0 += 5
                cu = T.tensor([0, k_inc.shape[1]], device=dev, dtype=T.int32)
                d_inc = idx.prepare_index(k_inc, cu)     # 走 _prepare_index_inc
            # 全量重建同一 KV
            idx.clear()
            cu = T.tensor([0, k_inc.shape[1]], device=dev, dtype=T.int32)
            d_full = idx.prepare_index(k_inc, cu)
            # k_min/k_max/k_qat 逐位
            ok = all(
                d_inc[kk].shape == d_full[kk].shape
                and T.equal(d_inc[kk], d_full[kk])
                for kk in ("k_min", "k_max", "k_qat"))
            # 全量 vs 增量 mask 逐位（消费端确定性）
            q = T.randn(1, 1, H, D, device=dev, dtype=T.bfloat16)
            pos = T.tensor([k_inc.shape[1] - 1], device=dev, dtype=T.long)
            sc_inc = idx.compute_score(q, pos, d_inc, scale)
            m_inc = idx.compute_mask(pos, sc_inc)
            idx.clear()
            sc_full = idx.compute_score(q, pos, d_full, scale)
            m_full = idx.compute_mask(pos, sc_full)
            ok_mask = T.equal(m_inc, m_full)
            check(f"G4 full_vs_inc {cname} {ltag}: 增量==全量",
                  ok and ok_mask,
                  f"dict_ok={ok} mask_ok={ok_mask}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "需要 GPU（Triton/eager 消费端）"
    dev = torch.device('cuda:0')
    torch.cuda.set_device(dev)
    test_batch_rows_equiv()
    test_anchor_batch()
    test_full_vs_inc()
    print(f"\n=== PASS {len(PASS)} / FAIL {len(FAIL)} ===")
    for f in FAIL:
        print(f"  FAIL: {f}")
    sys.exit(1 if FAIL else 0)
