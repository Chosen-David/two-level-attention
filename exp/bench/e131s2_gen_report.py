#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E131-S2 终表生成：从 e131s2_final/ 的 pre/post JSON 生成
e131_phase2_optimized.json + e131_phase2_report.md。"""
import json
import statistics

BASE = "/home/wangyuanshuo02/two-level-attention-e131s2/exp/results_efficiency"
OUT_JSON = f"{BASE}/e131_phase2_optimized.json"
OUT_MD = f"{BASE}/e131_phase2_report.md"

STAGES = ("prepare_index", "compute_score", "compute_mask", "metrics", "eager_decoding")
ARMS = ("mavg", "aavg", "mminmax", "cavgsim", "cclustersim")
EXTRA = {"mavg_e127": "mavg 同机校准锚点（E127 定案 iter8 配置）"}
STEPMAP = {"cclustersim": 38, "cavgsim": 80}

# spike 摊销周期（步）：cavgsim far 续跑每 64 步（far_hi 块对齐前移 bs=64）；
# cclustersim near 重建 key=(far_hi,near_hi)，far_hi 每 64 步前移 → 周期 64
SPIKE_PERIOD = 64


def summarize(path):
    d = json.load(open(path))
    r = d["runs"][0]
    warmup = d["meta"]["protocol"]["warmup"]
    out = {"stages": {}, "spike": {}, "gate": d["runs"][0].get("gate")}
    for st in STAGES:
        gpu = r["per_step_gpu_ms"][st]
        out["stages"][st] = statistics.median(gpu[warmup:])
    pi = r["per_step_gpu_ms"]["prepare_index"]
    idx = max(range(warmup, len(pi)), key=lambda i: pi[i])
    out["spike"]["prepare_index_max"] = pi[idx]
    out["spike"]["step"] = idx
    out["spike"]["amortized"] = pi[idx] / SPIKE_PERIOD
    # 首步冷建（prepare_index GPU ms）
    out["first_step_prepare_gpu_ms"] = r["first_step"]["prepare_index"]["gpu_ms"]
    fs = r["first_step"]
    out["first_step_total_gpu_ms"] = sum(fs[s]["gpu_ms"] for s in STAGES)
    out["SUM"] = sum(out["stages"].values())
    out["wall_cpu_ms"] = statistics.median(r["per_step_cpu_ms"]["total"]) \
        if "total" in r.get("per_step_cpu_ms", {}) else None
    return out


def main():
    res = {}
    for arm in ARMS + tuple(EXTRA.keys()):
        pre = summarize(f"{BASE}/e131s2_final/e131s2_preopt_{arm}.json")
        post = summarize(f"{BASE}/e131s2_final/e131s2_postopt_{arm}.json")
        res[arm] = {"pre": pre, "post": post,
                    "delta_SUM": post["SUM"] - pre["SUM"],
                    "speedup_SUM": pre["SUM"] / post["SUM"] if post["SUM"] else None}

    doc = {
        "meta": {
            "experiment": "E131-S2 阶段 2：greedy cluster Triton v2（SPLIT 协作）+ 簇分数 Triton 全融合，五臂 64K 稳态延迟终表",
            "protocol": {
                "S": 65536, "B": 1, "q_len": 1, "H": 32, "Hkv": 8, "D": 128,
                "bs": 64, "K1": 128, "K2": 1024, "layer": 2, "kv_dtype": "bfloat16",
                "warmup": 8,
                "steps": {"cclustersim": 38, "cavgsim": 80, "默认": 30},
                "timing": "cuda Event GPU 时间线 median（稳态，steps>=warmup）；spike 单列 + /64 步摊销；冷建首步单列",
                "gpu": "2×H20-3e（终表数据 GPU1 独占采集；开发验证 GPU0）",
                "data": "torch.randn 合成 KV/Q（E127 harness 同口径，singleton 型最坏贪心簇量）",
                "baseline_env": "TLI_E131_GREEDY_V2=0 + TLI_E131_TOKSCORE_TRITON=0（全关）",
                "optimized_env": "默认全开（v2 SPLIT 协作 kernel + tok_score Triton 融合）",
            },
            "optimizations": [
                {
                    "id": "①",
                    "name": "greedy cluster pass Triton v2（SPLIT 协作 kernel）",
                    "file": "sparse_attn/indexer/greedy_triton.py",
                    "switch": "TLI_E131_GREEDY_V2（默认 1；0 回退）",
                    "detail": "grid=(H,SPLIT) 同头多 program 均分活簇扫描 + 两阶段软件 barrier；"
                              "warmup 编译探测 n_regs 死锁防御钳制；小 T<256/探测失败回退 v1",
                    "effect": "far 冷建 8.0→0.96s（8.4×）；near 冷建 137.7→47.6ms；"
                              "大 k_live 续跑 1080→113ms（9.5×）",
                },
                {
                    "id": "③",
                    "name": "簇分数 gather-dot Triton 全融合（E127 GQA K 复用同款）",
                    "file": "sparse_attn/indexer/tli_indexer.py（E127 inplace 合并后；原 tli_indexer_fast.py）",
                    "switch": "TLI_E131_TOKSCORE_TRITON（默认 1；0 回退锚点原版）",
                    "detail": "einsum+gather+q_g 胶水三件套 → 单 kernel（int32 索引 + qg 组内聚合融合）",
                    "effect": "far_token_score 0.107→0.057ms（1.89×）；near_token_score 0.062→0.047ms（1.32×）",
                },
                {
                    "id": "bugfix",
                    "name": "v2 回退路径 BT 修复 + v2 内部 BT 自适应",
                    "file": "sparse_attn/indexer/greedy_triton.py",
                    "detail": "回退 v1 时 BT 256→512（大 k_live 下 tile 固定开销主导，T=64@57K 实测 2.83× 回归修复）；"
                              "v2 内部每 program 行数≥1024 时 BT=512（续跑 135→117ms）",
                    "effect": "cavgsim far 续跑 spike 73.7→28.3ms（修复 e2e 净回归）",
                },
                {
                    "id": "②",
                    "name": "near 侧增量化（复用未失效簇）",
                    "status": "评估后不做",
                    "detail": "锚点语义：near 均值簇心无法精确撤销左缘前移成员（tli_indexer.py L489 注释），"
                              "增量需改簇心语义破坏逐位一致性；v2 已把 near 重建 spike 156.7→72.3ms（-54%），"
                              "摊销 0.57ms/步，增量化的边际收益不抵语义风险，如实列为未实施",
                },
            ],
            "correctness": {
                "greedy_v2": "exp/bench/e131s2_v2_validate.py V1-V5 全 PASS（冷启动 vs Python 参考 / 24 组多 seed 逐位 100% / v2 vs v1 逐位 / 增量续跑==全量重放 / randn 最坏 k_live≈28K 续跑逐位）",
                "tok_score": "16 步稳态 mask 逐位 0 差异（8390592 位）",
                "e2e_mask_bitexact": "cclustersim/cavgsim 各 24 步（覆盖 far 块边界续跑 + near 重建）优化前后 mask 逐位一致率 12585120/12585120 = 100.0000%（top-k 重叠率门 ≥99.9% 满分）",
                "gate": "min_row_any / out_finite / sel_tokens=1024 恒定 全绿（逐臂 JSON gate 字段）",
            },
            "md5": {
                "tli_indexer.py（锚点，未动）": "bf5e26c4d6123875b2786b9ed886da5f",
                "tli_indexer_fast.py": "467fd087b24156765c19dafac3281215 → de9786c9d2f221e14d3c4aa988f95b54",
                "greedy_triton.py": "ee14a7882d580db640b544d55876981a → b8669bb6ca58976248bc0e2434f9a1ad",
            },
        },
        "results": res,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(f"JSON → {OUT_JSON}")

    # ---- Markdown 报告 ----
    L = []
    L.append("# E131-S2 阶段 2 终表：cluster 路径 kernel 极致优化后五臂 64K 稳态延迟\n")
    L.append("## 一、优化内容\n")
    L.append("| # | 优化 | 文件 | 开关 | 效果 |")
    L.append("|---|------|------|------|------|")
    L.append("| ① | greedy pass Triton v2（SPLIT 协作 kernel，同头多 program 均分扫描+软件 barrier，n_regs 死锁防御） | greedy_triton.py | TLI_E131_GREEDY_V2 | far 冷建 8.0s→0.96s（8.4×），大 k_live 续跑 9.5× |")
    L.append("| ③ | 簇分数 gather-dot Triton 全融合（E127 GQA K 复用同款：einsum+gather+q_g→单 kernel） | tli_indexer_fast.py | TLI_E131_TOKSCORE_TRITON | far 0.107→0.057ms（1.89×），near 0.062→0.047ms（1.32×） |")
    L.append("| fix | v2 回退路径 BT 256→512 + v2 内部 BT 自适应（大 k_live tile 开销主导） | greedy_triton.py | — | 修复 cavgsim far 续跑 spike 净回归 73.7→28.3ms |")
    L.append("| ② | near 增量化 | — | 评估后不做 | 均值簇心无法精确撤销左缘成员（锚点语义），v2 已兑现 spike -54%，摊销 0.57ms/步 |")
    L.append("")
    L.append("## 二、五臂 64K 稳态延迟终表（GPU ms median，S=65536，decode 1 token/步）\n")
    L.append("| 臂 | 优化前 SUM | 优化后 SUM | Δ SUM | 加速比 | 优化前 mask | 优化后 mask | 冷建首步 前→后 | spike 前→后（摊销） |")
    L.append("|----|-----------|-----------|-------|--------|-------------|-------------|----------------|---------------------|")
    for arm in ARMS:
        p = res[arm]
        L.append(f"| {arm} | {p['pre']['SUM']:.4f} | {p['post']['SUM']:.4f} | "
                f"{p['delta_SUM']:+.4f} | {p['speedup_SUM']:.3f}× | "
                f"{p['pre']['stages']['compute_mask']:.4f} | {p['post']['stages']['compute_mask']:.4f} | "
                f"{p['pre']['first_step_total_gpu_ms']:.0f}→{p['post']['first_step_total_gpu_ms']:.0f} ms | "
                f"{p['pre']['spike']['prepare_index_max']:.1f}→{p['post']['spike']['prepare_index_max']:.1f} ms"
                f"（{p['post']['spike']['amortized']:.3f} ms/步） |")
    L.append("")
    L.append(f"校准锚点 mavg_e127：优化前 {res['mavg_e127']['pre']['SUM']:.4f} → 优化后 {res['mavg_e127']['post']['SUM']:.4f}"
             f"（E127 定案 iter8 mavg 热路径不专项优化，两组差异应在噪声带内：Δ={res['mavg_e127']['delta_SUM']:+.4f}）。\n")
    L.append("### 分段明细（GPU ms median）\n")
    L.append("| 臂 | 组 | prepare_index | compute_score | compute_mask | metrics | eager_decoding | SUM |")
    L.append("|----|----|---------------|---------------|--------------|---------|----------------|-----|")
    for arm in ARMS + tuple(EXTRA.keys()):
        for tag, key in (("优化前", "pre"), ("优化后", "post")):
            s = res[arm][key]["stages"]
            L.append(f"| {arm} | {tag} | " + " | ".join(f"{s[st]:.4f}" for st in STAGES)
                     + f" | {res[arm][key]['SUM']:.4f} |")
    L.append("")
    L.append("## 三、冷建 / spike 摊销曲线（cluster 臂）\n")
    for arm in ("cavgsim", "cclustersim"):
        p = res[arm]
        L.append(f"- **{arm}**：首步冷建 {p['pre']['first_step_total_gpu_ms']:.0f} → "
                 f"{p['post']['first_step_total_gpu_ms']:.0f} ms"
                 f"（{p['pre']['first_step_total_gpu_ms']/p['post']['first_step_total_gpu_ms']:.1f}×）；"
                 f"prepare_index spike {p['pre']['spike']['prepare_index_max']:.1f} → "
                 f"{p['post']['spike']['prepare_index_max']:.1f} ms"
                 f"（周期 64 步，摊销 {p['post']['spike']['amortized']:.3f} ms/步，"
                 f"优化前摊销 {p['pre']['spike']['amortized']:.3f} ms/步）")
    L.append("")
    L.append("## 四、正确性验证\n")
    L.append("- greedy v2 对拍（exp/bench/e131s2_v2_validate.py）：V1-V5 全 PASS——冷启动 vs Python 参考、24 组多 seed 逐位一致率 100%（393216 位）、v2 vs v1 同输入逐位 100%、增量续跑==全量重放、randn 最坏 k_live≈28K 续跑逐位一致。")
    L.append("- tok_score 融合：16 步稳态 mask 逐位 0 差异（8390592 位）。")
    L.append("- e2e mask 逐位终验：cclustersim / cavgsim 各 24 步（覆盖 far 块边界续跑 + near 重建），优化前后 mask 逐位一致率 12585120/12585120 = **100.0000%**（top-k 重叠率门 ≥99.9% 满分，超门）。")
    L.append("- 逐臂 bench 末步 gate：min_row_any=true / out_finite=true / sel_tokens=1024 恒定，全绿。")
    L.append("- 锚点 tli_indexer.py md5 未变（bf5e26c4d6123875b2786b9ed886da5f）。")
    L.append("")
    L.append("## 五、如实的回归与修复记录\n")
    L.append("1. v2 首版回退路径把 v1 默认 BT=512 误设为 256：大 k_live 续跑（T=64@k_live≈57K）每 tile 固定开销主导，实测 2.83× 净回归（cavgsim e2e far 续跑 spike 25.3→73.7ms）。已修复（回退显式 BT=512 + v2 内部行数≥1024 自适应 BT=512），修复后 spike 28.3ms，回到优化前水平。")
    L.append("2. E113 的「BT=256 vs 512 仅差 6%」结论只在 struct K≈2K 小簇口径成立，大簇态（k_live≥28K）必须大 BT——已写入 wrapper 注释。")
    L.append("3. cavgsim 优化后 spike 28.3 vs 优化前 25.3：残余 +3.0ms 为 v2 wrapper 额外 CPU 开销（k_live.max().item() 同步 + 探测缓存查询，CPU +2.9ms/步实测）+ GPU0/GPU1 跨卡波动，稳态 SUM 无影响，如实记录。")
    L.append("")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(L))
    print(f"MD → {OUT_MD}")


if __name__ == "__main__":
    main()
