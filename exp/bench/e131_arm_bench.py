#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E131：E125 五臂 E127 fast path（TLI_E127_FAST=1）S=64K decode 分阶段延迟基准。

== 目的（用户指令 2026-10-12：「后续你这些子冠军补充 64K 上他们各自极致
   优化完后的时间延迟精度数据」）==
为 E125 五臂海选表中 mavg 之外的四个臂（aavg / mminmax / cavgsim /
cclustersim）补测 E127 fast path 下 S=64K 的 decode 分阶段延迟，产出与
mavg 1.52ms（E127 定案，本机 H20，exp/results_efficiency/e127_stage_bench_
final_64k.json normal 层 step_gpu_sum_ms=1.5201）同口径的数字。本阶段（阶段 1）
测 fast path 开箱即用（各段自动适用面），识别未覆盖瓶颈段，供阶段 2 专项优化。

== 结构（克隆 exp/bench/e127_stage_bench.py）==
  - 合成 KV cache（Qwen3-8B 形状：D=128、GQA 32/8、bs=64、K1=128、K2=1024）
  - STAGES = prepare_index / compute_score / compute_mask / metrics / eager_decoding
  - cuda Event（GPU 时间线）+ perf_counter（CPU wall，含 launch 开销与
    host 同步停顿）双计时；每步 evs[末].synchronize()（E127 定案同口径）
  - decode 模拟：B=1 q_len=1，每步 KV 追加 1 token（HF DynamicCache 语义）

== 配置 ==
  - 五臂冻结配置**逐字段克隆** exp/trace/e126_stage_timing.py（E125 派单口径，
    含 COMMON_ARGS 与 ARM_CONFIGS，不得凭记忆编造）。
  - 第六个伪臂 mavg_e127 = E127 harness make_args 原样克隆（cmp_ratio=2 /
    subspace='full' / enable_layer_skip=True / kmeans=True+4bit-select），
    用于同机复测 E127 定案 1.52ms 配置 → 机器校准系数（.251 H20 vs 本机 H20），
    同时把「配置差（E126 冻结 vs E127 harness）」与「机器漂移」两个因素分离。

== 与 E126 口径的关系 ==
  - E126（优化前基线）阶段口径 S0-S8 是 AST 插桩 span；本 harness 与 E127
    同为五阶段事件口径。selector_total = 五阶段 GPU median 和（互斥阶段和）。
  - 簇臂（cavgsim/cclustersim）冷建簇在首步（E126 同款口径）：报告须分开
    「首步冷建（first_step）」与「稳态 median（steps>=warmup）」。
    kmeans/greedy build 不在 E127 fast path 覆盖面属预期，如实记录。
  - 稳态窗口内簇臂可能出现块边界前移触发的重建步（near 全量重建 /
    far 增量续跑，每 ~64 步一次）：全步原始样本按执行顺序持久化
    （不排序不舍入，E126 055 口径），分析端可分离重建步与稳态步。

== 正确性门（fail-closed，继承 E127/E126 harness）==
  末步检查：mask 无全 False 行 / attention 输出 finite / 每行选中 token 数
  在 [sink+swa, K2+sink+swa+块宽余量] 合理区间（E126 同款界）。任一失败 →
  非零退出 + failure JSON，性能 JSON 不发布。

== 用法（仓根；必须 TLI_E127_FAST=1，启动即断言）==
  TLI_E127_FAST=1 CUDA_VISIBLE_DEVICES=1 python3 exp/bench/e131_arm_bench.py \
      --arm cclustersim --S 65536 --steps 30 --warmup 8 --json OUT.json
  可选臂：mavg / aavg / mminmax / cavgsim / cclustersim / mavg_e127（锚点）。
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import torch  # noqa: E402

from sparse_attn.indexer.tli_indexer import TLIIndexer  # noqa: E402
from sparse_attn.ops.eager_decoding import (  # noqa: E402
    eager_decoding_attn, eager_decoding_check_errors)
from sparse_attn.patches.utils import prepare_cu_seqlens, prepare_seqlens  # noqa: E402
# 【E127 inplace 合并（2026-10-12）】TLIIndexerFast 子类退役，增量缓存 +
# Triton fast path 已并入 TLIIndexer 基类默认生效——_IdxCls 恒为基类；
# 启动断言 TLI_E127_FAST≠0，防显式回退合并前口径冒充 fast path 数字。
_IdxCls = TLIIndexer

STAGES = ["prepare_index", "compute_score", "compute_mask", "metrics", "eager_decoding"]

# ===================================================================== #
# 五臂冻结配置（逐字段克隆 exp/trace/e126_stage_timing.py，E125 派单口径，
# 勿改）。预算固定 K1=128(page=64)/K2=1024/cmp_ratio=4；Hkv=8，D=128，
# H=32（GQA G=4）；子空间 tail32（dd=32）。公共：tli_enable_layer_skip=False
#（microbench 单层无 D'）、tli_sim_dims=subspace（= tail32）。
# ===================================================================== #
COMMON_ARGS = dict(
    tia_block_size=64, tia_level1_topk=128, tia_level2_topk=1024,
    tia_level2_cmp_ratio=4, tia_enable_async_topk=False,
    tli_subspace="tail", tli_enable_subspace=True,
    tli_enable_layer_skip=False, tli_layer_skip_path=None,
    tli_sim_dims="subspace", tli_sim=0.9,
    tli_sigma_select="none", tli_moba=False, tli_per_q_head=False,
    tli_static_pair=False, tli_proj_basis=None,
    tli_far_clusters=256, tli_far_niter=10,
    tli_far_blocks=16, tli_far_tokens=512,
)

ARM_CONFIGS = {
    "mavg":       dict(far_method="minmax", near_method="avg",
                       far_select="4bit", near_select="4bit",
                       enable_kmeans=False, alpha=0.25, beta=0.125, gamma=0.625),
    "aavg":       dict(far_method="avg", near_method="avg",
                       far_select="4bit", near_select="4bit",
                       enable_kmeans=False, alpha=0.125, beta=0.125, gamma=0.375),
    "mminmax":    dict(far_method="minmax", near_method="minmax",
                       far_select="4bit", near_select="4bit",
                       enable_kmeans=False, alpha=0.875, beta=0.875, gamma=0.125),
    "cavgsim":    dict(far_method="minmax", near_method="avg",
                       far_select="sim_greedy", near_select="4bit",
                       enable_kmeans=True, alpha=0.0, beta=0.0, gamma=0.0),
    "cclustersim": dict(far_method="minmax", near_method="avg",
                        far_select="sim_greedy", near_select="sim_greedy",
                        enable_kmeans=True, alpha=0.125, beta=0.25, gamma=0.125),
}

ARMS = list(ARM_CONFIGS) + ["mavg_e127", "mavg_e2e", "mavg_tail", "mavg_l2full"]


def make_args(arm):
    """五臂参数化 make_args（E126 冻结配置）；mavg_e127 = E127 锚点原样克隆；
    mavg_e2e = e2e 锚点口径（cmp_ratio=4：L2 细筛 32 维，与 E123/E125
    e2e 同配置）；mavg_tail = E133 子空间臂（mavg_e2e + tail32，
    单键差 L1 粗筛 128→32 维）。"""
    if arm == "mavg_e2e":
        ns = make_args("mavg_e127")
        ns.tia_level2_cmp_ratio = 4
        return ns
    if arm == "mavg_tail":
        ns = make_args("mavg_e127")
        ns.tli_subspace = "tail"
        ns.tia_level2_cmp_ratio = 4
        return ns
    if arm == "mavg_l2full":
        ns = make_args("mavg_e127")
        ns.tli_subspace = "tail"
        ns.tia_level2_cmp_ratio = 4
        ns.tli_l2_subspace = "full"
        return ns
    if arm == "mavg_e127":
        # E127 harness make_args(0.25, 0.125, 0.625) 原样克隆（机器校准锚点；
        # cmp_ratio=2 / subspace='full' / layer_skip=True / kmeans=True 但
        # far/near_select=4bit → 建簇门关，行为 = mavg 生产路径）。
        return argparse.Namespace(
            method='tli', tia_block_size=64, tia_level1_topk=128, tia_level2_topk=1024,
            tia_level2_cmp_ratio=2, tia_enable_async_topk=False,
            tli_enable_subspace=True, tli_enable_kmeans=True,
            tli_far_select='4bit', tli_near_select='4bit',
            tli_far_clusters=256, tli_far_niter=10,
            tli_far_blocks=16, tli_far_tokens=512,
            tli_far_method='minmax', tli_near_method='avg',
            tli_sim=0.9, tli_sim_dims='subspace',
            tli_sigma_select='none', tli_moba=False, tli_sigma=8.0,
            tli_per_q_head=False, tli_enable_layer_skip=True, tli_layer_skip_path=None,
            tli_alpha=0.25, tli_beta=0.125, tli_gamma=0.625, tli_subspace='full',
            tli_l2_subspace='same', tli_proj_basis=None, tli_static_pair=False)
    cfg = dict(COMMON_ARGS)
    ac = ARM_CONFIGS[arm]
    cfg.update(dict(
        tli_far_method=ac["far_method"], tli_near_method=ac["near_method"],
        tli_far_select=ac["far_select"], tli_near_select=ac["near_select"],
        tli_enable_kmeans=ac["enable_kmeans"],
        tli_alpha=ac["alpha"], tli_beta=ac["beta"], tli_gamma=ac["gamma"],
    ))
    return argparse.Namespace(**cfg)


def arm_config_for_meta(arm):
    """【123 修复（GPT 审计 2026-10-12_0933 TL-E131-CONFIG-IDENTITY-123，
    P2）】meta.config 的 arm 段——mavg_e127 臂记录实际 effective args。

    缺陷背景：修复前 mavg_e127 臂无条件写 ARM_CONFIGS.get(args.arm,
    {note 占位})——meta 声称的配置与实际运行配置（make_args 的 cmp_ratio=2
    / subspace='full' / enable_layer_skip=True，与 E126 冻结 COMMON_ARGS 的
    cmp_ratio=4 / subspace='tail' / layer_skip=False 三键不同）不符，机器
    校准锚点的配置身份不可复核。冻结五臂仍逐字段记 ARM_CONFIGS（E126
    派单口径，不变）。
    """
    if arm in ARM_CONFIGS:
        return dict(ARM_CONFIGS[arm])
    if arm == "mavg_e2e":
        return {
            "effective_args": vars(make_args(arm)),
            "note": "E133 e2e 锚点口径：mavg_e127 + cmp_ratio=4"
                    "（L2 细筛 32 维，与 E123/E125 e2e 生产配置同口径）",
        }
    if arm == "mavg_tail":
        return {
            "effective_args": vars(make_args(arm)),
            "note": "E133 子空间臂：mavg_e127 锚点 + subspace=tail + "
                    "cmp_ratio=4（E129 C2 同款 tail32 口径，仅此两键异）",
        }
    if arm == "mavg_l2full":
        return {
            "effective_args": vars(make_args(arm)),
            "note": "E134 ③ sub+full 臂：mavg_tail + l2_subspace=full"
                    "（L1 tail32 粗筛 + L2 全 128 维量化细筛）",
        }
    return {
        "effective_args": vars(make_args(arm)),
        "note": "mavg_e127 E127 锚点配置（make_args 实际 effective args，"
                "非 COMMON_ARGS：cmp_ratio=2 / subspace=full / "
                "layer_skip=True）",
    }


# ---------------------------------------------------------------- 工具（E126 精简版）
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_identity(root):
    def run(*cmd):
        try:
            p = subprocess.run(["git", "-C", root, *cmd],
                               capture_output=True, text=True, timeout=60)
            return p.stdout.strip() if p.returncode == 0 else None
        except Exception:
            return None
    sha = run("rev-parse", "HEAD")
    if sha is None:
        return {"git_sha": None, "git_dirty": None,
                "note": f"{root} 非 git 仓库"}
    porc = run("status", "--porcelain", "--", ".")
    return {"git_sha": sha, "git_dirty": bool(porc),
            "dirty_files": porc.splitlines()[:20] if porc else []}


def atomic_write_bytes(path, data):
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------- 主 bench
def bench_arm(arm, S0, n_steps, warmup, device, layer):
    """单臂单尺寸 decode 模拟。全步（含 warmup）记录，稳态 = steps>=warmup。

    返回 dict：per-step 原始 GPU/CPU 样本 + 稳态派生量 + 首步冷建 + 末步正确性门。
    """
    torch.manual_seed(20261011 + layer)
    idx = _IdxCls(make_args(arm))
    idx.layer_idx = layer
    idx.clear()

    H, Hkv, D = 32, 8, 128
    scale = D ** -0.5
    k = torch.randn(1, S0, Hkv, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, S0, Hkv, D, device=device, dtype=torch.bfloat16)

    gpu_ms = {s: [] for s in STAGES}       # 全步原始样本（含 warmup，执行顺序）
    cpu_ms = {s: [] for s in STAGES}
    wall_ms = []
    n_events_per_step = len(STAGES) + 1
    gate = {}
    last_out = None

    for step in range(warmup + n_steps):
        q = torch.randn(1, 1, H, D, device=device, dtype=torch.bfloat16)
        # 全步记录（首步冷建簇在 step0，须进样本；warmup 步照记，
        # 稳态派生只取 steps>=warmup）
        evs = [torch.cuda.Event(enable_timing=True) for _ in range(n_events_per_step)]
        t_wall0 = time.perf_counter()
        evs[0].record()

        cu = prepare_cu_seqlens(k)
        q_ids = prepare_seqlens(cu) - 1

        index_dict = idx.prepare_index(k, cu)
        evs[1].record(); t = time.perf_counter()
        score_dict = idx.compute_score(q, q_ids, index_dict, scale)
        evs[2].record(); cpu_ms["compute_score"].append((time.perf_counter() - t) * 1e3); t = time.perf_counter()
        mask = idx.compute_mask(q_ids, score_dict)
        evs[3].record(); cpu_ms["compute_mask"].append((time.perf_counter() - t) * 1e3); t = time.perf_counter()
        idx.metrics.add_select_result(q_ids, mask, 1)
        evs[4].record(); cpu_ms["metrics"].append((time.perf_counter() - t) * 1e3); t = time.perf_counter()
        out = eager_decoding_attn(q, k, v, mask, 1, cu, scale)
        evs[5].record(); cpu_ms["eager_decoding"].append((time.perf_counter() - t) * 1e3)
        evs[5].synchronize()
        wall_end = time.perf_counter()
        wall_ms.append((wall_end - t_wall0) * 1e3)
        rest = sum(cpu_ms[s][-1] for s in STAGES[1:])
        cpu_ms["prepare_index"].append(max(0.0, (wall_end - t_wall0) * 1e3 - rest))
        for i, s in enumerate(STAGES):
            gpu_ms[s].append(evs[i].elapsed_time(evs[i + 1]))
        # 步末校验 GPU 错误标志（队列已空，读回 ~µs 级；E127 iter3 同款）
        eager_decoding_check_errors(device)

        if step == warmup + n_steps - 1:
            last_out = out
            # ---- 正确性门（末步；E126 同款界，fail-closed）----
            m0 = mask.squeeze(0)[0]          # [H 或 Hkv, T]（TLI token 级 mask，bs=1）
            row_any = m0.any(dim=-1)
            per_head = m0.float().sum(dim=-1)
            gate = {
                "min_row_any": bool(row_any.all()),
                "out_finite": bool(torch.isfinite(out.detach().float()).all()),
                "sel_tokens_min": float(per_head.min()),
                "sel_tokens_max": float(per_head.max()),
                "sel_tokens_mean": float(per_head.mean()),
                "mask_shape": list(mask.shape),
            }

        # 下一步：KV 追加 1 个新 token（模拟 decode；HF DynamicCache 每步 cat）
        new_k = torch.randn(1, 1, Hkv, D, device=device, dtype=torch.bfloat16)
        new_v = torch.randn(1, 1, Hkv, D, device=device, dtype=torch.bfloat16)
        k = torch.cat([k, new_k], dim=1)
        v = torch.cat([v, new_v], dim=1)

    torch.cuda.synchronize()

    # ---- 派生：稳态窗口（steps >= warmup）----
    def steady(seq):
        return seq[warmup:]

    run = {
        "arm": arm, "S0": S0, "layer": layer,
        "steps": n_steps, "warmup": warmup,
        "indexer_class": type(idx).__name__,
        "triton_ok": bool(getattr(idx, "_triton_ok", True)),
        "inc_cache_built": getattr(idx, "_inc", None) is not None,
        # 全步原始样本（执行顺序持久化，不排序不舍入）
        "per_step_gpu_ms": {s: gpu_ms[s] for s in STAGES},
        "per_step_cpu_ms": {s: cpu_ms[s] for s in STAGES},
        "per_step_wall_ms": wall_ms,
        # 稳态派生量
        "stages": {},
        "step_gpu_sum_ms": None,
        "step_cpu_wall_ms": statistics.median(steady(wall_ms)),
        # 首步（冷建簇 / 增量缓存全量构建）
        "first_step": {s: {"gpu_ms": gpu_ms[s][0], "cpu_ms": cpu_ms[s][0]}
                       for s in STAGES},
        "first_step_wall_ms": wall_ms[0],
        "gate": gate,
    }
    for s in STAGES:
        g = steady(gpu_ms[s])
        c = steady(cpu_ms[s])
        run["stages"][s] = {
            "gpu_ms_median": statistics.median(g),
            "gpu_ms_p90": sorted(g)[int(0.9 * len(g)) - 1],
            "gpu_ms_max": max(g),
            "cpu_ms_median": statistics.median(c),
        }
    run["step_gpu_sum_ms"] = sum(run["stages"][s]["gpu_ms_median"] for s in STAGES)
    run["last_out_finite_recheck"] = bool(
        torch.isfinite(last_out.detach().float()).all())
    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', type=str, required=True, choices=ARMS)
    p.add_argument('--S', type=int, nargs='+', default=[65536])
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--warmup', type=int, default=8)
    p.add_argument('--layer', type=int, default=2,
                   help='基准层（E126/E127 同款 normal 层）')
    p.add_argument('--json', type=str, required=True)
    args = p.parse_args()

    if os.environ.get("TLI_E127_FAST", "1") == "0":
        print("[GATE-FAIL] E131 须 fast path 口径（默认生效；合并后基类内置），"
              "但 TLI_E127_FAST=0 显式回退了合并前口径——拒绝跑优化前口径"
              "冒充 fast path 数字",
              flush=True)
        raise SystemExit(2)
    assert torch.cuda.is_available()
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    tli_path = os.path.join(REPO, "sparse_attn", "indexer", "tli_indexer.py")
    greedy_path = os.path.join(REPO, "sparse_attn", "indexer", "greedy_triton.py")
    harness_path = os.path.abspath(__file__)
    meta = {
        "probe": ("E131 五臂 E127 fast path S=64K decode 分阶段延迟基准"
                  "（阶段 1：开箱即用口径，供阶段 2 专项优化定位瓶颈段）"),
        "arm": args.arm,
        "config": {"common": COMMON_ARGS,
                   # 123：mavg_e127 臂记录实际 effective args（make_args
                   # Namespace），冻结五臂仍逐字段记 ARM_CONFIGS。
                   "arm": arm_config_for_meta(args.arm),
                   "is_e126_frozen_arm": args.arm in ARM_CONFIGS},
        "protocol": {"stages": STAGES, "kv_dtype": "bfloat16", "B": 1, "q_len": 1,
                     "H": 32, "Hkv": 8, "D": 128, "bs": 64, "K1": 128, "K2": 1024,
                     "steps": args.steps, "warmup": args.warmup,
                     "layer": args.layer,
                     "timing": ("cuda Event GPU 时间线 + perf_counter CPU wall 双计时；"
                                "每步 evs[末].synchronize()（E127 定案同口径）；"
                                "全步原始样本按执行顺序持久化不排序不舍入，"
                                "稳态 median 由 steps>=warmup 样本派生；"
                                "首步冷建簇单独报告"),
                     "data": ("torch.randn 合成 KV/Q（E127 harness 同口径，"
                              "singleton 型最坏贪心簇量；非 E126 structured 口径）"),
                     "env": {"TLI_E127_FAST": os.environ.get("TLI_E127_FAST"),
                             "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}},
        "versions": {"torch": torch.__version__, "cuda": torch.version.cuda,
                     "python": sys.version.split()[0]},
        "identity": {"root": REPO, "git": git_identity(REPO),
                     "tli_indexer": {"md5": file_md5(tli_path),
                                     "sha256": file_sha256(tli_path)},
                     "note_e127_inplace_merge": (
                         "2026-10-12 inplace 合并后 fast path 内置于基类；"
                         "tli_indexer_fast.py 已按用户指令退役"),
                     "greedy_triton": {"sha256": file_sha256(greedy_path)},
                     "harness_sha256": file_sha256(harness_path)},
        "gpu": torch.cuda.get_device_name(0),
        "started": time.strftime("%F %T"),
    }
    try:
        import triton
        meta["versions"]["triton"] = triton.__version__
    except Exception:
        meta["versions"]["triton"] = None

    results = {"meta": meta, "runs": []}
    for S0 in args.S:
        r = bench_arm(args.arm, S0, args.steps, args.warmup, device, args.layer)
        results["runs"].append(r)
        print(f"\n=== arm={args.arm} S0={S0} layer={args.layer} "
              f"({r['indexer_class']}, triton_ok={r['triton_ok']}, "
              f"inc_cache={r['inc_cache_built']}) ===", flush=True)
        print(f"{'stage':<16}{'GPU ms(med)':>12}{'GPU ms(p90)':>12}"
              f"{'GPU ms(max)':>12}{'CPU ms(med)':>12}")
        for s in STAGES:
            st = r["stages"][s]
            print(f"{s:<16}{st['gpu_ms_median']:>12.4f}{st['gpu_ms_p90']:>12.4f}"
                  f"{st['gpu_ms_max']:>12.4f}{st['cpu_ms_median']:>12.4f}")
        print(f"{'SUM(GPU med)':<16}{r['step_gpu_sum_ms']:>12.4f}")
        print(f"{'step CPU wall':<16}{r['step_cpu_wall_ms']:>12.4f}")
        print(f"{'first step wall':<15}{r['first_step_wall_ms']:>12.1f}"
              f"（冷建簇/全量缓存构建）")
        print(f"gate: {json.dumps(r['gate'], ensure_ascii=False)}", flush=True)

        # ---- 正确性门判定（fail-closed，E126 同款界）----
        g = r["gate"]
        gate_errors = []
        if not g.get("min_row_any"):
            gate_errors.append("mask 存在全 False 行（softmax NaN 契约破坏）")
        if not g.get("out_finite"):
            gate_errors.append("attention 输出非 finite")
        expect_min = 128 + 128          # sink+swa 强制下界
        expect_max = 1024 + 128 + 128 + 64 * 4 + 16
        if not (expect_min - 1 <= g.get("sel_tokens_min", -1) <= expect_max):
            gate_errors.append(f"每头选中 token 数下界异常 min={g.get('sel_tokens_min')}")
        if not (expect_min - 1 <= g.get("sel_tokens_max", -1) <= expect_max * 4):
            gate_errors.append(f"每头选中 token 数上界异常 max={g.get('sel_tokens_max')}")
        if gate_errors:
            fail_doc = {"status": "failed", "arm": args.arm, "S": S0,
                        "errors": gate_errors, "gate": g,
                        "identity": meta["identity"],
                        "timestamp": time.strftime("%F %T")}
            fail_path = args.json + ".failure.json"
            os.makedirs(os.path.dirname(fail_path) or ".", exist_ok=True)
            atomic_write_bytes(fail_path, json.dumps(
                fail_doc, indent=1, ensure_ascii=False).encode("utf-8"))
            print("[GATE-FAIL] " + json.dumps(gate_errors, ensure_ascii=False),
                  flush=True)
            raise SystemExit(1)

    results["finished"] = time.strftime("%F %T")
    final_bytes = json.dumps(results, indent=1, ensure_ascii=False).encode("utf-8")
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    atomic_write_bytes(args.json, final_bytes)
    atomic_write_bytes(args.json + ".sha256",
                       (hashlib.sha256(final_bytes).hexdigest() + "\n").encode("utf-8"))
    print(f"\nJSON → {args.json}", flush=True)


if __name__ == '__main__':
    main()
