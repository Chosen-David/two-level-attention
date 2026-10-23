#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 2a 门禁套件：TLI 固定宽度 batch>1 decode（deepseek 轨）。

用法（仓根）：
  CUDA_VISIBLE_DEVICES=0 python3 exp/bench/test_tli_batch_gates.py

背景：阶段 2a 把 batch decode 从「DynamicCache 变长 KV（有效占尾）」切到
「TLIStaticCache 固定宽度 KV（有效在前、S_cap 恒量）」——CUDA Graph 前置。
`_tl_decode_batched` 逐行消费全宽 k_b [1, S_cap, Hkv, D] +
prepare_mask(..., valid_length=L_b)：有效区 [0, L_b) 与变长路径逐位一致、
pad 区 [L_b, S_cap) 全 False（E155e fixed_width 探针已证）。

门禁（单元级——不加载模型，合成 KV 直测 indexer + eager 消费端）：
  G1 gate_bs1_equiv_fixed：batch=1 固定宽分支 == 变长单序列核心逐位（==）
     ——batch B=1 退化与 pred_ruler.py 单序列入口逐位一致。
  G3 gate_multiseq_consistency：B=4 混合长度多步 decode（固定宽 buffer
     每步 alive 行写新列），batch 分支逐行 == 变长参考臂逐位（==）。
  G2 gate_anchor_batch：预算拉满==dense 锚点（有效区 mask 全连）+ 候选化
     开关（TLI_FINE_GATHER 0/1）逐位一致 + pad 区防泄漏。
  G5 gate_fixed_pad_clear：默认冠军几何下，全宽 mask 有效区 == 变长 mask
     逐位 && pad 区全 False（E155e 探针断言固化）。
  G4 gate_full_vs_inc：全量重建 == 单序列增量路径（E133c 逐位依据）——
     batch 全量重建与 pred_ruler.py 单序列增量语义一致的根基。

left-pad 协议（TLIStaticCache）：行 b 有效 KV 在 [0, L_b)、pad 在尾
[L_b, S_cap)（垃圾值，验证 mask 防泄漏不选中）。
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


def _left_pad_kv(S_cap, seq_lens, device, seed):
    """构造 [B, S_cap, Hkv, D] seq-first left-pad KV（固定宽度）。

    行 b 有效 KV 在列 [0, L_b)（行内独立 seed 流，与同 seed 单序列一致）；
    pad 区 [L_b, S_cap) 填主 generator 垃圾（非零）——若 mask/eager 泄漏到
    pad 列，输出必然偏离参考臂（防泄漏验证的硬条件）。
    """
    B = len(seq_lens)
    Hkv, D = 8, 128
    g = torch.Generator(device=device).manual_seed(seed)
    k = torch.randn(B, S_cap, Hkv, D, dtype=torch.bfloat16, device=device,
                    generator=g)
    v = torch.randn_like(k)
    for b, L in enumerate(seq_lens):
        gb = torch.Generator(device=device).manual_seed(seed + b)
        k[b, :L] = torch.randn(1, L, Hkv, D, generator=gb, device=device,
                               dtype=torch.bfloat16)[0]
        v[b, :L] = torch.randn(1, L, Hkv, D, generator=gb, device=device,
                               dtype=torch.bfloat16)[0]
    return k, v


def _run_row_var(idx, q, k_full_row, v_full_row, L_b, scale):
    """变长路径参考臂（pred_ruler.py 单序列 decode 分支口径）。

    取有效区 k_var = k_full_row[:, :L_b]（变长 [1, L_b, Hkv, D]），
    prepare_mask 无 valid_length——即全量重建单序列语义。与固定宽分支
    的逐位一致性 = E155e fixed_width 探针核心断言。
    """
    k_var = k_full_row[:, :L_b]      # [1, L_b, Hkv, D]
    v_var = v_full_row[:, :L_b]
    cu = torch.tensor([0, L_b], device=k_var.device, dtype=torch.int32)
    pos = torch.tensor([L_b - 1], device=k_var.device, dtype=torch.long)
    idx.clear()
    mask, bs = idx.prepare_mask(q, pos, k_var, cu, softmax_scale=scale)
    from sparse_attn.ops.eager_decoding import eager_decoding_attn
    return eager_decoding_attn(q, k_var, v_var, mask, bs, cu,
                               softmax_scale=scale)


def _run_batch(idx, q_hf, k_hf, v_hf, ctx, scale):
    from sparse_attn.patches.qwen3_attn_patch import _tl_decode_batched
    from sparse_attn.baseline_runtime import BatchCtx  # noqa: F401
    return _tl_decode_batched(_StubAttn(idx, scale), idx, q_hf, k_hf, v_hf,
                              ctx)


# ------------------------------------------------------------- G1 + G3
def test_batch_rows_equiv():
    """G1+G3：固定宽 batch 分支 vs 变长单序列核心，B=1 与 B=4 多步逐位一致。

    覆盖 4 配置 × normal/skip 层 × 3 步 decode（固定宽 buffer，每步 alive
    行写新 KV 到列 seq_lens[b]，即当前有效区末尾）。batch 分支输入形状流
    与 qwen3_attn_forward decode 入口一致：q_hf [B,H,1,D]、k_hf/v_hf
    [B,Hkv,S_cap,D]（TLIStaticCache.update 返回全宽 buffer）。
    """
    import torch as T
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    from sparse_attn.baseline_runtime import BatchCtx

    dev = T.device('cuda:0')
    H, Hkv, D = 32, 8, 128
    scale = D ** -0.5
    SEED = 20261020

    for cname, args in CONFIGS:
        for layer_idx, ltag in [(2, "normal"), (0, "skip")]:
            # ---- B=1（G1）：固定宽 [1, S_cap] vs 变长 [1, L] ----
            L1 = 48
            S_cap1 = L1 + 32          # pad 区垃圾
            k1, v1 = _left_pad_kv(S_cap1, [L1], dev, SEED)
            q1 = T.randn(1, 1, H, D, device=dev, dtype=T.bfloat16)
            idx = TLIIndexer(args); idx.layer_idx = layer_idx; idx.clear()
            q_hf1 = q1.transpose(1, 2)                     # [1,H,1,D]
            k_hf1 = k1.transpose(1, 2).contiguous()        # [1,Hkv,S_cap,D]
            v_hf1 = v1.transpose(1, 2).contiguous()
            ctx1 = BatchCtx(batch=1, seq_lens=[L1],
                            gen_lens=[0], alive=[True], pad_id=0)
            o_b1 = _run_batch(idx, q_hf1, k_hf1, v_hf1, ctx1, scale)
            o_s1 = _run_row_var(idx, q1, k1[0:1], v1[0:1], L1, scale)
            eq1 = T.equal(o_b1, o_s1)
            check(f"G1 bs1_equiv_fixed {cname} {ltag} B=1",
                  eq1, f"nd={int((o_b1 != o_s1).sum().item()) if not eq1 else 0}")

            # ---- B=4 多步（G3）：固定宽 buffer 逐列写 ----
            seq_lens = [16, 32, 48, 64]
            S_cap = max(seq_lens) + 8
            k, v = _left_pad_kv(S_cap, seq_lens, dev, SEED)
            kk, vv = k.clone(), v.clone()
            all_ok = True
            for step in range(3):
                new_k = T.randn(4, Hkv, D, device=dev, dtype=T.bfloat16)
                new_v = T.randn(4, Hkv, D, device=dev, dtype=T.bfloat16)
                for b, L in enumerate(seq_lens):
                    kk[b, L] = new_k[b]     # 写当前有效长位置 L（末尾）
                    vv[b, L] = new_v[b]
                seq_lens = [L + 1 for L in seq_lens]       # forward 前 +1
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
                    o_s.append(_run_row_var(
                        idx, qs[b], kk[b:b+1], vv[b:b+1], L, scale))
                o_s = T.cat(o_s, dim=0)
                if not T.equal(o_b, o_s):
                    all_ok = False
                    break
            check(f"G3 multiseq_consistency {cname} {ltag} B=4×3step",
                  all_ok, "固定宽 vs 变长参考臂逐位")


# ------------------------------------------------------------- G2 锚点 + pad 防泄漏
def test_anchor_batch():
    """G2：预算拉满==dense 锚点（有效区 mask 全连）+ 候选化开关
    TLI_FINE_GATHER 0/1 逐位一致 + pad 区防泄漏（阶段 0/2a 口径复用）。
    """
    import torch as T
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    from sparse_attn.baseline_runtime import BatchCtx

    dev = T.device('cuda:0')
    H, Hkv, D = 32, 8, 128
    scale = D ** -0.5
    SEED = 20261020
    seq_lens = [16, 32, 48, 64]
    S_cap = max(seq_lens) + 8
    k, v = _left_pad_kv(S_cap, seq_lens, dev, SEED)

    # q 用固定 seed 生成（两臂须逐位同输入；全局 RNG 在两次 run 间会漂移）
    g = T.Generator().manual_seed(SEED)
    qs = [T.randn(1, 1, H, D, generator=g, dtype=T.bfloat16).to(dev)
          for _ in seq_lens]
    q_hf = T.stack([q.transpose(1, 2)[0] for q in qs], dim=0)
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

    # mask 全连（dense 锚点语义：预算拉满下每行有效区全覆盖）+ pad 防泄漏。
    # 全宽 mask [1,1,Hkv,S_cap]，有效区 [0,L_b) 应全真、pad [L_b,S_cap) 全假。
    args = _make_cfg(0.25, 0.125, 0.625, "minmax", "avg")
    args.tia_level1_topk = 999999
    args.tia_level2_topk = 999999
    idx = TLIIndexer(args); idx.layer_idx = 2; idx.clear()
    L_b = 48                                     # 取 seq_lens[2]
    cu = T.tensor([0, S_cap], device=dev, dtype=T.int32)
    pos = T.tensor([L_b - 1], device=dev, dtype=T.long)
    mask, _bs = idx.prepare_mask(qs[2], pos, k[2:3], cu, softmax_scale=scale,
                                 valid_length=L_b)
    eff_all = bool(mask[0, 0, :, :L_b].all())
    pad_clear = bool((~mask[0, 0, :, L_b:]).all())
    check("G2 anchor_batch: 预算拉满有效区全连 + pad 防泄漏",
          eff_all and pad_clear,
          f"eff={eff_all} pad_clear={pad_clear} shape={tuple(mask.shape)}")


# ------------------------------------------------------------- G5 固定宽 pad_clear
def test_fixed_pad_clear():
    """G5：默认冠军几何下全宽 mask 有效区 == 变长 mask 逐位 && pad 全 False。

    E155e fixed_width 探针断言固化：非预算拉满（自由竞争几何），固定宽
    prepare_mask(valid_length=L) 的有效区与变长路径逐位一致（同 token 集、
    同 block 决策），pad 区不被任何 head 选中。
    """
    import torch as T
    from sparse_attn.indexer.tli_indexer import TLIIndexer

    dev = T.device('cuda:0')
    H, Hkv, D = 32, 8, 128
    scale = D ** -0.5
    SEED = 20261020
    L_b = 641                      # 非块对齐：测 kt_valid<kt 且 near/swa 边界
    S_cap = L_b + 32
    k, v = _left_pad_kv(S_cap, [L_b], dev, SEED)

    for cname, args in CONFIGS:
        for layer_idx, ltag in [(2, "normal"), (0, "skip")]:
            for fg in (0, 1):
                os.environ["TLI_FINE_GATHER"] = str(fg)
                idx = TLIIndexer(args); idx.layer_idx = layer_idx; idx.clear()
                q = T.randn(1, 1, H, D, device=dev, dtype=T.bfloat16)
                # 变长参考 mask（无 valid_length）
                cu_var = T.tensor([0, L_b], device=dev, dtype=T.int32)
                m_var, _ = idx.prepare_mask(q, T.tensor([L_b - 1],
                    device=dev, dtype=T.long), k[0:1, :L_b], cu_var,
                    softmax_scale=scale)
                # 全宽 mask（valid_length=L_b）
                idx.clear()
                cu_fix = T.tensor([0, S_cap], device=dev, dtype=T.int32)
                m_fix, _ = idx.prepare_mask(q, T.tensor([L_b - 1],
                    device=dev, dtype=T.long), k[0:1], cu_fix,
                    softmax_scale=scale, valid_length=L_b)
                eff = m_fix[0, 0, :, :L_b]
                pad = m_fix[0, 0, :, L_b:]
                same = T.equal(eff, m_var[0, 0])
                pad_clear = bool((~pad).all())
                check(f"G5 fixed_pad_clear {cname} {ltag} fg={fg}",
                      same and pad_clear,
                      f"eff==var={same} pad_clear={pad_clear}")


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


# ------------------------------------------------------------- G6 批量 eager（Triton 批量）逐位一致
def test_batch_eager_equiv():
    """G6：批量 eager_decoding_attn（Triton 批量 kernel A/B）vs 逐行单 query
    eager_decoding_attn（同 kernel 逻辑）逐位一致。

    S_cap=16384 触发 Triton 批量路由（T≥16384、block_size==1）。批量臂
    逐行 prepare_mask 后 cat [B,1,Hkv,S_cap] + cu_b=[B+1] varlen；逐行臂
    同 mask 逐行单 query（valid_length=L）。批量 kernel 逐 (b,r) program
    独立归约（仅 batch 维寻址）→ 输出 == 逐行逐位（nd=0 硬门）。另验
    _EAGER_TRITON_OK 保持 True——证明批量 Triton 真被执行（而非 torch
    回退兜底）。
    """
    import torch as T
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    import sparse_attn.ops.eager_decoding as ed

    dev = T.device('cuda:0')
    H, Hkv, D = 32, 8, 128
    scale = D ** -0.5
    SEED = 20261020
    S_cap = 16384
    os.environ["TLI_FINE_GATHER"] = "1"

    def _row_mask(args, layer_idx, qb, kb, L_b):
        idx = TLIIndexer(args); idx.layer_idx = layer_idx; idx.clear()
        pos = T.tensor([L_b - 1], device=dev, dtype=T.long)
        cu = T.tensor([0, S_cap], device=dev, dtype=T.int32)
        return idx.prepare_mask(qb, pos, kb, cu, softmax_scale=scale,
                                valid_length=L_b)

    for cname, args in CONFIGS:
        for layer_idx, ltag in [(2, "normal"), (0, "skip")]:
            for tag, seq_lens in [("B2", [16000, 16384]),
                                  ("B4", [8000, 16000, 16384, 16032])]:
                k, v = _left_pad_kv(S_cap, seq_lens, dev, SEED)
                kk, vv = k.contiguous(), v.contiguous()
                q = T.randn(len(seq_lens), 1, H, D, device=dev,
                            dtype=T.bfloat16)
                # 逐行 mask（参考臂与批量臂共用同一批 mask，隔离 kernel 差异）
                masks = [_row_mask(args, layer_idx, q[b:b + 1], kk[b:b + 1], L)
                         for b, L in enumerate(seq_lens)]
                # 逐行参考臂（单 query Triton，valid_length=L）
                o_rows = T.cat([
                    ed.eager_decoding_attn(
                        q[b:b + 1], kk[b:b + 1], vv[b:b + 1], masks[b][0],
                        masks[b][1],
                        T.tensor([0, S_cap], device=dev, dtype=T.int32),
                        softmax_scale=scale, valid_length=L)
                    for b, L in enumerate(seq_lens)], dim=0)
                # 批量臂（同 mask cat → Triton 批量，无 valid_length——
                # mask 已由 prepare_mask 统一清 pad 区）
                m_b = T.cat([m for m, _ in masks], dim=0)   # [B,1,Hkv,S_cap]
                cu_b = T.arange(len(seq_lens) + 1, device=dev,
                                dtype=T.int32) * S_cap
                o_batch = ed.eager_decoding_attn(
                    q, kk, vv, m_b, masks[0][1], cu_b, softmax_scale=scale)
                nd = int((o_batch != o_rows).sum().item())
                check(f"G6 batch_eager_equiv {cname} {ltag} {tag}",
                      nd == 0, f"nd={nd}")
    # Triton 批量真被执行（而非 torch 回退）：_EAGER_TRITON_OK 须保持 True
    check("G6 triton_batch_active: 批量 Triton 路径未回退",
          ed._EAGER_TRITON_OK["ok"], f"ok={ed._EAGER_TRITON_OK['ok']}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "需要 GPU（Triton/eager 消费端）"
    dev = torch.device('cuda:0')
    torch.cuda.set_device(dev)
    test_batch_rows_equiv()
    test_anchor_batch()
    test_fixed_pad_clear()
    test_full_vs_inc()
    test_batch_eager_equiv()
    print(f"\n=== PASS {len(PASS)} / FAIL {len(FAIL)} ===")
    for f in FAIL:
        print(f"  FAIL: {f}")
    sys.exit(1 if FAIL else 0)
