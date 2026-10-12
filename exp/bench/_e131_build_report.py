#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E131 阶段 1 报告构建脚本（一次性工具）：汇总六臂 JSON → e131_phase1_report.json。

【E132b TL-E131-AGG-FAILOPEN-119 修复】聚合前 fail-closed 校验：
唯一 run / arm 匹配预期 / S0·steps·warmup·layer 协议一致 / gate 全过 /
数值 finite / sidecar sha256 匹配 / triton_ok·inc_cache_built 全真。
任一失败 → stderr 报原因 + 退出非零，拒绝发布（此前错 arm/S0=8192/
NaN/翻 gate/缺 sidecar 均可被静默汇总成「64K 五臂报告」）。"""
import hashlib
import json
import math
import os
import sys
import time

STAGES = ["prepare_index", "compute_score", "compute_mask", "metrics", "eager_decoding"]

# 【119】E131 阶段 1 正式协议（.251 GPU1 独占采集）：S0=64K、steps=30
# （cavgsim 80 步补充文件例外）、warmup=8、layer=2——聚合产物自称
# 「64K 五臂报告」，输入任何一项偏离协议即身份不明，拒绝汇总。
_EXPECT = {"S0": 65536, "warmup": 8, "layer": 2}
_EXPECT_STEPS = {"mavg": 30, "aavg": 30, "mminmax": 30, "cavgsim": 30,
                 "cclustersim": 30, "mavg_e127": 30, "cavgsim_steps80": 80}


def _fail(msg):
    print(f"[E131-agg FAILOPEN-119] 拒绝发布: {msg}", file=sys.stderr)
    sys.exit(1)


def _finit(v, where):
    if isinstance(v, bool) or not isinstance(v, (int, float)) \
            or not math.isfinite(v):
        _fail(f"{where} = {v!r} 非有限数值")


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def load(arm):
    """【119】校验后加载：sidecar sha256 → 唯一 run → arm/协议/gate/finite。
    arm 亦可为 'cavgsim_steps80'（补充文件，steps 预期 80，arm 名 cavgsim）。"""
    expect_arm = arm[:-len("_steps80")] if arm.endswith("_steps80") else arm
    path = f'e131_arm_bench_{arm}.json'
    if not os.path.exists(path):
        _fail(f"输入缺失: {path}")
    # sidecar sha256 匹配（首 token，兼容 bare-hash 与 sha256sum 双格式）
    sc_path = path + '.sha256'
    if not os.path.exists(sc_path):
        _fail(f"sidecar 缺失: {sc_path}")
    sc_want = open(sc_path).read().split()[0].strip()
    sc_got = sha(path)
    if sc_want != sc_got:
        _fail(f"sidecar sha256 不匹配: {path}（want={sc_want} got={sc_got}）")
    d = json.load(open(path))
    runs = d.get('runs') or []
    if len(runs) != 1:
        _fail(f"{path}: runs={len(runs)} ≠ 1（唯一 run 校验）")
    r = runs[0]
    if d.get('meta', {}).get('arm') != expect_arm or r.get('arm') != expect_arm:
        _fail(f"{path}: arm 身份不匹配（meta={d.get('meta', {}).get('arm')!r} "
              f"run={r.get('arm')!r} 预期={expect_arm!r}）")
    for k in ("S0", "warmup", "layer"):
        if r.get(k) != _EXPECT[k]:
            _fail(f"{path}: run.{k}={r.get(k)!r} ≠ 协议预期 {_EXPECT[k]}")
    if r.get('steps') != _EXPECT_STEPS[arm]:
        _fail(f"{path}: run.steps={r.get('steps')!r} ≠ 预期 {_EXPECT_STEPS[arm]}")
    proto = d.get('meta', {}).get('protocol') or {}
    if proto.get('warmup') is not None and proto['warmup'] != r['warmup']:
        _fail(f"{path}: meta.protocol.warmup={proto['warmup']} 与 run.warmup={r['warmup']} 自相矛盾")
    if proto.get('steps') is not None and proto['steps'] != r['steps']:
        _fail(f"{path}: meta.protocol.steps={proto['steps']} 与 run.steps={r['steps']} 自相矛盾")
    # gate 全过（末步弱门禁是既有口径——此处只拒绝「明确翻红」的输入）
    g = r.get('gate') or {}
    if g.get('min_row_any') is not True or g.get('out_finite') is not True:
        _fail(f"{path}: gate 未全过（min_row_any={g.get('min_row_any')} "
              f"out_finite={g.get('out_finite')}）")
    if not (g.get('sel_tokens_min', 0) > 0):
        _fail(f"{path}: gate.sel_tokens_min={g.get('sel_tokens_min')} 非正")
    if r.get('last_out_finite_recheck') is not True:
        _fail(f"{path}: last_out_finite_recheck={r.get('last_out_finite_recheck')}")
    if r.get('triton_ok') is not True or r.get('inc_cache_built') is not True:
        _fail(f"{path}: triton_ok={r.get('triton_ok')} "
              f"inc_cache_built={r.get('inc_cache_built')}（fast path 覆盖断言）")
    # 数值 finite：稳态聚合字段 + 全步原始样本
    _finit(r['step_gpu_sum_ms'], f"{path}:run.step_gpu_sum_ms")
    _finit(r['step_cpu_wall_ms'], f"{path}:run.step_cpu_wall_ms")
    _finit(r['first_step']['prepare_index']['gpu_ms'],
           f"{path}:run.first_step.prepare_index.gpu_ms")
    for s in STAGES:
        _finit(r['stages'][s]['gpu_ms_median'], f"{path}:stages.{s}.gpu_ms_median")
        _finit(r['stages'][s]['gpu_ms_p90'], f"{path}:stages.{s}.gpu_ms_p90")
    for st_name, arr in (r.get('per_step_gpu_ms') or {}).items():
        for i, v in enumerate(arr):
            _finit(v, f"{path}:per_step_gpu_ms.{st_name}[{i}]")
    return d


arms = ['mavg', 'aavg', 'mminmax', 'cavgsim', 'cclustersim']
steady = {}
for a in arms + ['mavg_e127']:
    r = load(a)['runs'][0]
    steady[a] = {
        'stages': {s: r['stages'][s]['gpu_ms_median'] for s in STAGES},
        'stages_p90': {s: r['stages'][s]['gpu_ms_p90'] for s in STAGES},
        'sum_gpu_ms': r['step_gpu_sum_ms'],
        'selector_total_ms': r['step_gpu_sum_ms'] - r['stages']['eager_decoding']['gpu_ms_median'],
        'cpu_wall_ms': r['step_cpu_wall_ms'],
        'first_step_prepare_ms': r['first_step']['prepare_index']['gpu_ms'],
        'gate': r['gate'],
        'triton_ok': r['triton_ok'], 'inc_cache_built': r['inc_cache_built'],
    }

# cavgsim 80 步补充（far 增量续跑 spike 实测）——【119】同样走校验加载
d80 = load('cavgsim_steps80')
r80 = d80['runs'][0]
prep80 = r80['per_step_gpu_ms']['prepare_index']
cav_far_spike = max((x, i) for i, x in enumerate(prep80) if i > 1)
# cclustersim 38 步内 near 重建 spike 实测
rcc = load('cclustersim')['runs'][0]
prepcc = rcc['per_step_gpu_ms']['prepare_index']
cc_near_spike = max((x, i) for i, x in enumerate(prepcc) if i > 1)

E127_LOCAL = 1.5201119966804981  # e127_stage_bench_final_64k.json normal 层（本机 H20）
calib = steady['mavg_e127']['sum_gpu_ms'] / E127_LOCAL

report = {
    "meta": {
        "probe": "E131 阶段 1：E125 五臂 E127 fast path（TLI_E127_FAST=1）S=64K decode 分阶段延迟（开箱即用口径）",
        "user_directive": "后续你这些子冠军补充 64K 上他们各自极致优化完后的时间延迟精度数据（2026-10-12）——本阶段先测 fast path 开箱即用，识别未覆盖瓶颈段，供阶段 2 专项优化",
        "generated": time.strftime("%F %T"),
        "measurement_environ": {
            "host": ".251 (10.238.139.251) GPU1（CUDA_VISIBLE_DEVICES=1，独占锁卡，与 GPU0 E129 任务物理隔离）",
            "gpu": "NVIDIA H20-3e", "torch": "2.8.0+cu128", "triton": "3.4.0",
            "env": "TLI_E127_FAST=1（harness 启动断言）",
            "git_sha": "387b7242e8bcccbc071fab3ccea9174372fbe08a（.251 worktree ~/two-level-attention_e131，detached @ bee/main；主树 a096d2b 有 E129 在飞未提交改动，未触碰）",
            "tli_indexer_md5": "bf5e26c4d6123875b2786b9ed886da5f（E127 锚点零改动）",
            "harness": "exp/bench/e131_arm_bench.py（五臂配置逐字段克隆 e126_stage_timing.py + mavg_e127 锚点臂克隆 e127_stage_bench.py make_args）",
        },
        "caliber_notes": [
            "全部数字 = TLI_E127_FAST=1 优化后口径（E126 优化前基线数字不得与本表混用；E126 对比仅在 stage_baseline_e126_ref 字段作跨口径参考，其阶段口径为 AST 插桩 span、数据为 structured+singleton 双臂，与本表五阶段 event 口径 + randn 数据不同）",
            "每臂单进程串行独占 GPU1；S0=65536 起 decode 模拟每步 +1 token，steps=30 recorded / warmup=8，median 为稳态口径（E127 定案同款）",
            "合成数据 = torch.randn KV/Q（E127 harness 同口径，singleton 型 = 贪心簇量最坏情况；cluster 臂冷建/重建代价在真实数据下更低，见 cluster_arm_amortization 推算）",
            "selector_total = 五阶段 GPU median 和 − eager_decoding（互斥阶段和；与 E126 selector（S0..S7）同语义）",
            "性能点估计如实报：aavg 与 mavg 稳态 SUM 点估计差 0.0007ms（0.04%），按纪律不写「持平/相当」，如实列差值",
        ],
    },
    "identity": {
        "result_files": {f"e131_arm_bench_{a}.json": sha(f"e131_arm_bench_{a}.json")
                         for a in arms + ['mavg_e127']},
        "e131_arm_bench_cavgsim_steps80.json": sha("e131_arm_bench_cavgsim_steps80.json"),
    },
    "steady_state_64k_gpu_ms": steady,
    "five_arm_table": {
        "note": "五臂 = E125 冻结配置（E126 COMMON_ARGS+ARM_CONFIGS 逐字段克隆）；单位 ms/层/decode-step（GPU median，S=64K）",
        "rows": [],
    },
    "machine_calibration": {
        "purpose": "E127 定案 1.52ms 产自本机 H20；.251 同配置（mavg_e127 锚点臂 = e127_stage_bench.py make_args 原样克隆）复测分离「机器漂移」与「配置差」",
        "e127_definitive_local_h20_gpu_ms": E127_LOCAL,
        "mavg_e127_on_251_gpu_ms": steady['mavg_e127']['sum_gpu_ms'],
        "calibration_ratio_251_over_local": calib,
        "calibration_pct": f"+{(calib-1)*100:.1f}%（.251 比本机慢）",
        "per_stage_ratio": {s: round(steady['mavg_e127']['stages'][s] / {
            "prepare_index": 0.4471, "compute_score": 0.3496, "compute_mask": 0.4736,
            "metrics": 0.0676, "eager_decoding": 0.1822}[s], 4) for s in STAGES},
        "config_delta_same_machine": {
            "note": "同机（.251）内 mavg(E126 冻结配置) vs mavg_e127(E127 harness 配置) 的配置差（cmp_ratio 4 vs 2 / subspace tail vs full / layer_skip off vs on）",
            "mavg_e126_config_gpu_ms": steady['mavg']['sum_gpu_ms'],
            "mavg_e127_config_gpu_ms": steady['mavg_e127']['sum_gpu_ms'],
            "delta_pct": f"{(steady['mavg']['sum_gpu_ms']/steady['mavg_e127']['sum_gpu_ms']-1)*100:+.1f}%",
        },
    },
    "cluster_arm_amortization": {
        "note": "簇臂冷建簇在首步（E126 同款口径分开报告）；块边界前移触发 far 增量续跑/near 全量重建（每 ~64 decode 步一次），全步原始样本已持久化，spike 为实测值",
        "cavgsim": {
            "first_step_cold_build_ms": steady['cavgsim']['first_step_prepare_ms'],
            "far_incremental_spike_ms": {"value": round(cav_far_spike[0], 2), "at_step": cav_far_spike[1],
                                         "source": "e131_arm_bench_cavgsim_steps80.json（80 步窗口捕获 step64 far 边界）"},
            "amortized_far_rebuild_ms_per_step": round(cav_far_spike[0] / 64, 4),
            "no_near_rebuild": "near_select=4bit → 无 near 侧聚类",
        },
        "cclustersim": {
            "first_step_cold_build_ms": steady['cclustersim']['first_step_prepare_ms'],
            "near_full_rebuild_plus_far_incremental_spike_ms": {"value": round(cc_near_spike[0], 2), "at_step": cc_near_spike[1],
                                                                 "source": "e131_arm_bench_cclustersim.json 38 步窗口（step36 = far_hi 块边界 → far 续跑 64 tok + near 全量贪心重建 ~8.2K tok）"},
            "amortized_rebuild_ms_per_step_randn_worst": round(cc_near_spike[0] / 64, 4),
            "structured_data_estimate": "按 E126 冷建比例 structured/singleton ≈ 1679.7/7915.9 ≈ 0.212 → 重建估 ~33ms/64 步 ≈ 0.52ms/步（推算值非实测；randn 最坏 2.45ms/步为实测）",
        },
        "cold_build_root_cause": "_greedy_cluster_pass 为 Python 逐 token 循环（for i in range(T)：每 token ~7 次 kernel launch：bmm/argmax/gather/3×index_add_），65280 token 冷建 ≈ 45 万次 launch（launch-bound 非 FLOP-bound）——E127 fast path 未覆盖（预期内，如实记录）",
    },
    "bottleneck_vs_mavg": {
        "note": "同机同卡同口径直接可比（无需校准）；Δ = 该臂 − mavg 稳态 SUM",
        "rows": [],
    },
    "fastpath_coverage": {
        "prepare_index_inc_cache": "六臂全覆盖（inc_cache_built=True；稳态 0.48-0.53ms vs E126 优化前 S0+S1≈2.19ms）",
        "compute_score_triton": "六臂全覆盖（triton_ok=True；fine GQA K 复用 + coarse minmax/avg，0.38-0.46ms vs E126 S3 三段和≈3.05ms）",
        "compute_mask": "E127 优化点（skip 层 .item() 消除/far 切片零同步）在 normal 层基准不受益；cluster 臂簇分数段（_far_token_score/_near_token_score einsum+gather）完全未覆盖 → cclustersim mask 段 +0.257ms 主项",
        "metrics_lazy": "六臂全覆盖（GPU 0.09-0.11ms；baseline CPU 3.7ms/步停顿已消）",
        "eager_gather": "六臂全覆盖（0.22-0.23ms vs E126 优化前 S8≈3.05ms）",
        "kmeans_greedy_build": "未覆盖（预期内）：冷建 8.4-11.2s + 重建 spike 见 cluster_arm_amortization",
    },
    "stage_baseline_e126_ref": {
        "note": "跨口径参考（E126 优化前基线 exp/trace/results/e126_stage_timing_*.json S=64K structured：阶段=AST span；勿与本表精确对比）",
        "selector_stage_sum_ms": {"mavg": 5.55, "aavg": 5.54, "mminmax": 5.37, "cavgsim": 5.48, "cclustersim": 5.73},
        "s8_eager_ms": 3.05,
    },
    "phase2_recommendations": [],
}

for a in arms:
    s = steady[a]
    row = {"arm": a, **{k: round(v, 4) for k, v in s['stages'].items()},
           "sum_gpu_ms": round(s['sum_gpu_ms'], 4),
           "selector_total_ms": round(s['selector_total_ms'], 4),
           "cpu_wall_ms": round(s['cpu_wall_ms'], 4),
           "first_step_prepare_ms": round(s['first_step_prepare_ms'], 1)}
    report['five_arm_table']['rows'].append(row)
    d = s['sum_gpu_ms'] - steady['mavg']['sum_gpu_ms']
    report['bottleneck_vs_mavg']['rows'].append({
        "arm": a, "delta_sum_gpu_ms": round(d, 4),
        "delta_pct": f"{d / steady['mavg']['sum_gpu_ms'] * 100:+.2f}%",
        "stage_deltas": {st: round(s['stages'][st] - steady['mavg']['stages'][st], 4) for st in STAGES},
    })

report['phase2_recommendations'] = [
    {"arm": "cclustersim（E125 冠军臂）", "priority": 1,
     "target": "near 侧全量重建（实测 156.71ms/64 步 ≈ 2.45ms/步摊销 @randn 最坏；structured 推算 ~0.52ms/步）+ near 簇分数段（mask 段 +0.257ms vs mavg）",
     "actions": [
         "① _greedy_cluster_pass CUDA graph 化或 Triton 化：当前逐 token Python 循环 launch-bound（~7 launch/token），graph replay 估 3-10× 削减；冷建 8.4s / 重建 156.7ms 同源受益",
         "② near 侧改增量：右缘新增段贪心续跑语义已被 far 侧证明逐位一致（_update_far_greedy 增量=全量重放）；左缘撤销不可精确 → 混合策略（定期全量 + 间隙细筛分回退，生产代码已有回退语义）拉长重建周期",
         "③ _near_token_score/_far_token_score 的 einsum+gather 套用 E127 iter8 同款 GQA K 复用 Triton kernel（[Hkv,K_c,dd] 小 GEMV），预期 mask 段 −0.15~0.20ms",
     ],
     "expected_gain": "摊销口径（randn 最坏）4.55→~2.1ms/步；稳态口径 2.10→~1.9ms/步"},
    {"arm": "cavgsim", "priority": 2,
     "target": "冷建 11.18s（请求级首 token 延迟）+ far 增量续跑 25.36ms/64 步 ≈ 0.40ms/步摊销",
     "actions": ["同① greedy pass 加速（同源）；稳态 SUM 已比 mavg 低 0.060ms（(0,0) 单池简化路径），mask 段无需专项"],
     "expected_gain": "冷建 11.2s→~1-3s；摊销 +0.40→~+0.1ms/步"},
    {"arm": "aavg / mavg", "priority": 3,
     "target": "稳态 1.80ms（两臂点估计差 0.0007ms=0.04%）；score 段 0.45ms 为最大段",
     "actions": ["E127 iter9 fused mask 前段在 64K 已实测净回归（-3%）、仅 16K +3.7%——不推荐重试同方向；64K 档维持 iter8 结论"],
     "expected_gain": "有限；两臂吃 E127 既有优化已接近饱和"},
    {"arm": "mminmax", "priority": 4,
     "target": "稳态 1.65ms（五臂最快；无 k_avg 物化天然占优）",
     "actions": ["无簇开销无 avg 物化，阶段 1 无专项需求"],
     "expected_gain": "—"},
]

out = 'e131_phase1_report.json'
with open(out, 'w') as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
print("saved", out)

print("\n===== E131 阶段 1 五臂 64K 稳态表（.251 GPU1，TLI_E127_FAST=1，GPU ms median，ms/层/步）=====")
print(f"{'arm':<13}{'prep':>7}{'score':>7}{'mask':>7}{'metrics':>8}{'eager':>7}{'SUM':>8}{'sel_tot':>9}{'wall':>7}")
for a in arms + ['mavg_e127']:
    s = steady[a]
    print(f"{a:<13}" + "".join(f"{s['stages'][st]:7.3f}" for st in STAGES[:4])
          + f"{s['stages']['eager_decoding']:7.3f}"
          f"{s['sum_gpu_ms']:8.4f}{s['selector_total_ms']:9.4f}{s['cpu_wall_ms']:7.3f}")
print(f"\nmavg_e127 锚点校准: .251 {steady['mavg_e127']['sum_gpu_ms']:.4f} vs 本机 E127 定案 {E127_LOCAL:.4f}"
      f" → ×{calib:.4f}（.251 慢 {(calib-1)*100:.1f}%）")
print(f"配置差（同机）: mavg E126 冻结 {steady['mavg']['sum_gpu_ms']:.4f} vs mavg_e127 {steady['mavg_e127']['sum_gpu_ms']:.4f}"
      f" → {(steady['mavg']['sum_gpu_ms']/steady['mavg_e127']['sum_gpu_ms']-1)*100:+.1f}%")
print(f"簇臂摊销: cavgsim 冷建 {steady['cavgsim']['first_step_prepare_ms']/1e3:.2f}s"
      f" + far 续跑 {cav_far_spike[0]:.1f}ms/64步(≈{cav_far_spike[0]/64:.2f}ms/步)；"
      f"cclustersim 冷建 {steady['cclustersim']['first_step_prepare_ms']/1e3:.2f}s"
      f" + near 重建 {cc_near_spike[0]:.1f}ms/64步(≈{cc_near_spike[0]/64:.2f}ms/步 @randn 最坏)")
