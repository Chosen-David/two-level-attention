# Qwen3-8B Baseline 进度状态报告

**更新日期**: 2026-10-26 04:30
**机器**: pc33 (本机 2×H20), pc31 (10.245.58.57, 2×H20)

---

## 1. LongBench (AVG16) — 全部方法完成 ✅

| 方法 | AVG16 | 状态 |
|------|-------|------|
| FullKV | 49.39 | ✅ |
| JSSA | 49.28 | ✅ |
| TLI (冠军) | 49.27 | ✅ |
| AsyncTLS | 45.13 | ✅ |
| Quest | 41.35 | ✅ |
| FASA | 35.15 | ✅ |

**TLI vs FullKV**: -0.12（16 任务中 13 个差异 ≤1.0，最大差异 passage_retrieval_en -1.5）

---

## 2. RULER — 32K / 64K 全部完成，128K 在跑

### RULER 32K (AVG11)

| 方法 | AVG11 | 状态 |
|------|-------|------|
| ClusterKV | 60.60 | ✅ |
| SparQ | 60.45 | ✅ |
| HISA | 59.69 | ✅ |
| FullKV | 59.38 | ✅ |
| JSSA | 59.13 | ✅ |
| TLI (bsa_avg/mavg) | 58.96 | ✅ |
| MInference | 49.94 | ✅ |
| AsyncTLS | 48.33~50.09 | ✅ |
| Quest | 42.85 | ✅ |
| FASA | 18.64 | ✅ |

### RULER 64K (AVG11)

| 方法 | AVG11 | 状态 |
|------|-------|------|
| MInference | 50.29 | ✅ |
| TLI (mavg) | 49.42 | ✅ |
| FullKV | 48.54 | ✅ |
| JSSA | 48.40 | ✅ |
| TLI (aavg) | 47.51 | ✅ |
| AsyncTLS | 33.88~34.11 | ✅ |
| Quest | 26.25 | ✅ |
| FASA | 1.72 | ✅ |

### RULER 128K (AVG11)

| 方法 | AVG11 | 状态 |
|------|-------|------|
| MInference | 49.53 | ✅ |
| TLI (mavg) | 47.49 | ✅ |
| FullKV | 46.43 | ✅ |
| TLI (aavg) | 42.56 | ✅ |
| AsyncTLS | 37.47 | ✅ |
| Quest | 19.33 | ✅ |
| FASA | 1.40 | ✅ |
| JSSA | — | 🔄 5/11 (pc31 GPU1) |

---

## 3. JSSA 128K RULER 进度

- pc31 GPU1, ts=202610251730
- 已完成 (5/11): niah_single_1 (100), niah_single_2 (100), niah_single_3 (100), niah_multikey_1 (100), niah_multikey_2 (100)
- 进行中: niah_multikey_3 (21/100 截至 04:30)
- 待: niah_multiquery, niah_multivalue, cwe, fwe, vt
- 速率 ≈ 2h/任务 → 预计全部完成约 16:00
- 完成后操作: JSSA-only isolated root + `score_ruler_formal.py --yarn --yarn-factor 4.0 --expect-tasks 11 --min-samples 100`

---

## 4. 数据产物位置

所有 formal results 已同步到 `exp/trace/results/`:

- LongBench: `qwen3_8b_longbench_{fullkv|tli_champion|jssa_vendor|asynctls_vendor|quest_vendor|fasa_vendor}.json`
- RULER 64K JSSA: `qwen3_8b_ruler64k_jssa_vendor.json`
- RULER 32K/64K/128K: `e119_ruler{32768|64k|128k}_formal_{method}.json`
