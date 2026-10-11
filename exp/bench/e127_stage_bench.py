#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E127：TLI HF 路径 decode 分阶段基准（baseline 数字表 = E126 五臂计时复用源）。

对每个 decode step、每个 attention 层，TLI 稀疏路径依次执行：
  prepare_index → compute_score → compute_mask → metrics.add_select_result
  → eager_decoding_attn（最终稀疏 attention 消费端）

本脚本用合成 KV cache（Qwen3-8B 形状：D=128、GQA 32/8、bs=64、K1=128、K2=1024）
逐 token 模拟 decode，对每个阶段同时记录：
  - GPU 时间（torch.cuda.Event，步末统一 sync，median）
  - CPU wall 时间（perf_counter，含 launch 开销与 host 同步停顿——sync 阶段
    的真实代价在这里）

用法（仓根）：
  CUDA_VISIBLE_DEVICES=0 python3 exp/bench/e127_stage_bench.py \
      [--S 8192 16384 32768 65536] [--steps 30] [--warmup 8] \
      [--alpha 0.25 --beta 0.125 --gamma 0.625] [--layer 2] [--skip-layer 0] \
      [--json OUT.json] [--profile]

注意：
  - 与真实 decode 一致：prefill 走 dense（不进本路径），首步 prepare_index
    在 S0 全量冷算（增量缓存的冷启动点也在这一步）。
  - metrics 阶段含 2 次 .item() host 同步（q_ids/S 均超 threshold=2048，
    与生产 decode 相同）。
"""
import argparse
import json
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import torch  # noqa: E402

from sparse_attn.indexer.tli_indexer import TLIIndexer  # noqa: E402
from sparse_attn.ops.eager_decoding import eager_decoding_attn  # noqa: E402
from sparse_attn.patches.utils import prepare_cu_seqlens, prepare_seqlens  # noqa: E402
# 【E127 iter4】env=1 用增量缓存子类（prepare_index 逐位一致加速，
# 锚点文件零改动）；默认 0 用原类（与 baseline 完全同路径）。
if os.environ.get("TLI_E127_FAST", "0") == "1":
    from sparse_attn.indexer.tli_indexer_fast import TLIIndexerFast as _IdxCls
else:
    _IdxCls = TLIIndexer


def make_args(alpha, beta, gamma):
    """冠军臂配置（mavg：far=minmax / near=avg + E64 分区）。"""
    return argparse.Namespace(
        method='tli', tia_block_size=64, tia_level1_topk=128, tia_level2_topk=1024,
        tia_level2_cmp_ratio=2, tia_enable_async_topk=False,
        tli_enable_subspace=True, tli_enable_kmeans=True, tli_far_select='4bit',
        tli_near_select='4bit', tli_far_clusters=256, tli_far_niter=10,
        tli_far_blocks=16, tli_far_tokens=512, tli_far_method='minmax',
        tli_near_method='avg', tli_sim=0.9, tli_sim_dims='subspace',
        tli_sigma_select='none', tli_moba=False, tli_sigma=8.0,
        tli_per_q_head=False, tli_enable_layer_skip=True, tli_layer_skip_path=None,
        tli_alpha=alpha, tli_beta=beta, tli_gamma=gamma, tli_subspace='full',
        tli_proj_basis=None, tli_static_pair=False)


STAGES = ["prepare_index", "compute_score", "compute_mask", "metrics", "eager_decoding"]


def bench_layer(layer_idx, S0, n_steps, warmup, device, alpha, beta, gamma):
    """单层 decode 模拟。返回 {stage: {gpu_ms_median, cpu_ms_median}} + 全步统计。"""
    torch.manual_seed(20261011 + layer_idx)
    idx = _IdxCls(make_args(alpha, beta, gamma))
    idx.layer_idx = layer_idx
    idx.clear()

    H, Hkv, D = 32, 8, 128
    scale = D ** -0.5
    k = torch.randn(1, S0, Hkv, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, S0, Hkv, D, device=device, dtype=torch.bfloat16)

    gpu_ms = {s: [] for s in STAGES}
    cpu_ms = {s: [] for s in STAGES}
    total_wall = []
    n_events_per_step = len(STAGES) + 1

    for step in range(warmup + n_steps):
        q = torch.randn(1, 1, H, D, device=device, dtype=torch.bfloat16)
        record = step >= warmup
        if record:
            evs = [torch.cuda.Event(enable_timing=True) for _ in range(n_events_per_step)]
            t_wall0 = time.perf_counter()
            evs[0].record()

        cu = prepare_cu_seqlens(k)
        q_ids = prepare_seqlens(cu) - 1

        index_dict = idx.prepare_index(k, cu)
        if record:
            evs[1].record(); t = time.perf_counter()
        score_dict = idx.compute_score(q, q_ids, index_dict, scale)
        if record:
            evs[2].record(); cpu_ms["compute_score"].append((time.perf_counter() - t) * 1e3); t = time.perf_counter()
        mask = idx.compute_mask(q_ids, score_dict)
        if record:
            evs[3].record(); cpu_ms["compute_mask"].append((time.perf_counter() - t) * 1e3); t = time.perf_counter()
        idx.metrics.add_select_result(q_ids, mask, 1)
        if record:
            evs[4].record(); cpu_ms["metrics"].append((time.perf_counter() - t) * 1e3); t = time.perf_counter()
        _ = eager_decoding_attn(q, k, v, mask, 1, cu, scale)
        if record:
            evs[5].record(); cpu_ms["eager_decoding"].append((time.perf_counter() - t) * 1e3)
            evs[5].synchronize()
            total_wall.append((time.perf_counter() - t_wall0) * 1e3)
            # prepare_index 的 cpu 时间 = 全步 wall 减去其余四段
            rest = sum(cpu_ms[s][-1] for s in STAGES[1:])
            cpu_ms["prepare_index"].append(max(0.0, (time.perf_counter() - t_wall0) * 1e3 - rest))
            for i, s in enumerate(STAGES):
                gpu_ms[s].append(evs[i].elapsed_time(evs[i + 1]))
            # 【E127 iter3】零同步路径配套：步末校验 GPU 错误标志
            # （此时队列已空，读回是纯 D2H ~µs 级，不增加 drain 停顿）
            from sparse_attn.ops.eager_decoding import eager_decoding_check_errors
            eager_decoding_check_errors(device)

        # 下一步：KV 追加 1 个新 token（模拟 decode；HF DynamicCache 每步 cat）
        new_k = torch.randn(1, 1, Hkv, D, device=device, dtype=torch.bfloat16)
        new_v = torch.randn(1, 1, Hkv, D, device=device, dtype=torch.bfloat16)
        k = torch.cat([k, new_k], dim=1)
        v = torch.cat([v, new_v], dim=1)

    torch.cuda.synchronize()
    out = {"layer_idx": layer_idx, "S0": S0, "steps": n_steps}
    for s in STAGES:
        out[s] = {
            "gpu_ms_median": statistics.median(gpu_ms[s]),
            "gpu_ms_p90": sorted(gpu_ms[s])[int(0.9 * len(gpu_ms[s])) - 1],
            "cpu_ms_median": statistics.median(cpu_ms[s]),
        }
    out["step_gpu_sum_ms"] = sum(out[s]["gpu_ms_median"] for s in STAGES)
    out["step_cpu_wall_ms"] = statistics.median(total_wall)
    return out


def profile_launches(layer_idx, S0, device, alpha, beta, gamma, n_steps=5):
    """torch profiler 模式：数每阶段 kernel launch 数（单独 pass，不计入数字表）。"""
    from torch.profiler import profile, ProfilerActivity
    idx = _IdxCls(make_args(alpha, beta, gamma))
    idx.layer_idx = layer_idx
    idx.clear()
    H, Hkv, D = 32, 8, 128
    scale = D ** -0.5
    k = torch.randn(1, S0, Hkv, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, S0, Hkv, D, device=device, dtype=torch.bfloat16)

    def one_step():
        q = torch.randn(1, 1, H, D, device=device, dtype=torch.bfloat16)
        cu = prepare_cu_seqlens(k)
        q_ids = prepare_seqlens(cu) - 1
        index_dict = idx.prepare_index(k, cu)
        score_dict = idx.compute_score(q, q_ids, index_dict, scale)
        mask = idx.compute_mask(q_ids, score_dict)
        idx.metrics.add_select_result(q_ids, mask, 1)
        _ = eager_decoding_attn(q, k, v, mask, 1, cu, scale)
        return index_dict, score_dict, mask

    for _ in range(3):
        one_step()  # warmup（profiler 外）
    torch.cuda.synchronize()

    launches = {}
    for stage_name, fn in [
        ("prepare_index", lambda: idx.prepare_index(k, prepare_cu_seqlens(k))),
        ("compute_score", None), ("compute_mask", None), ("eager_decoding", None),
    ]:
        if fn is None:
            continue
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        launches[stage_name] = sum(
            1 for e in prof.key_averages() if e.device_type == torch.profiler.DeviceType.CUDA
            and e.self_device_time_total > 0)
    # 整步 launch 数（更稳的口径）
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        one_step()
        torch.cuda.synchronize()
    total = sum(e.count for e in prof.key_averages()
                if e.device_type == torch.profiler.DeviceType.CUDA
                and e.self_device_time_total > 0)
    return {"one_step_cuda_kernels": total, "per_stage_kernel_counts": launches}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--S', type=int, nargs='+', default=[8192, 16384, 32768, 65536])
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--warmup', type=int, default=8)
    p.add_argument('--alpha', type=float, default=0.25)
    p.add_argument('--beta', type=float, default=0.125)
    p.add_argument('--gamma', type=float, default=0.625)
    p.add_argument('--layer', type=int, default=2, help='非 skip 层（D′ 生效层）')
    p.add_argument('--skip-layer', type=int, default=0, help='skip 层（走 far -inf 分支）')
    p.add_argument('--json', type=str, default=None)
    p.add_argument('--profile', action='store_true')
    args = p.parse_args()

    assert torch.cuda.is_available()
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    results = {"config": {"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma,
                         "K1": 128, "K2": 1024, "bs": 64, "D": 128, "H": 32, "Hkv": 8},
               "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "runs": []}
    for S0 in args.S:
        for li, tag in [(args.layer, "normal"), (args.skip_layer, "skip")]:
            r = bench_layer(li, S0, args.steps, args.warmup, device,
                            args.alpha, args.beta, args.gamma)
            r["layer_kind"] = tag
            results["runs"].append(r)
            print(f"\n=== S0={S0} layer={li}({tag}) ===")
            print(f"{'stage':<16}{'GPU ms(med)':>12}{'GPU ms(p90)':>12}{'CPU ms(med)':>12}")
            for s in STAGES:
                print(f"{s:<16}{r[s]['gpu_ms_median']:>12.4f}"
                      f"{r[s]['gpu_ms_p90']:>12.4f}{r[s]['cpu_ms_median']:>12.4f}")
            print(f"{'SUM(GPU med)':<16}{r['step_gpu_sum_ms']:>12.4f}")
            print(f"{'step CPU wall':<16}{r['step_cpu_wall_ms']:>12.4f}")
    if args.profile:
        pr = profile_launches(args.layer, 16384, device, args.alpha, args.beta, args.gamma)
        results["profile"] = pr
        print(f"\n[profile] one-step CUDA kernel launches: {pr['one_step_cuda_kernels']}")
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nJSON → {args.json}")


if __name__ == '__main__':
    main()
