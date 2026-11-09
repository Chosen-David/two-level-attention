# GLM-4.7-Flash 32 维索引子空间选举判决（2026-11-08）

任务：`TL-GLM-SUBSPACE-32D-20261105`。
本报告补齐上个会话未完成的 **RULER 诊断锚点**，并据此给出最终子空间选择。

## 结论（先说）

**选定 `glm_nope_random32`，`--nfa-subspace-seed 1`**，即当前 LongBench 英文 16 项
全量（`longbench_16en_full_nope_random32_s1`）正在使用的臂。

**但必须同时登记一个更重要的发现**：RULER 64K `niah_multikey_2` 的退化主要
**不是** 32 维子空间造成的——完整 256 维索引（`--nfa-subspace full`）在同口径下
只有 **48.0**，而 native fullKV 是 **90.0**。即：

- NFA 集成/选择器本身在 GLM 上的缺口：**−42.0**（90.0 → 48.0）
- 32 维子空间在此之上的额外缺口：**−30.0**（48.0 → 18.0）

因此"换一组 32 维"无法修复 RULER；继续调子空间之前必须先归因 full-index 缺口。

## 证据口径

- LongBench v1 `hotpotqa` + `gov_report`，每任务 n=200，全量非抽样；
  每请求同时跑 NFA `full` 臂与 `native_full` 对照臂（同输入、同 seed=42）。
- RULER 64K `niah_multikey_2`，n=100，seed=42，`string_match_all`；
  所有 NFA 臂 `--nfa_frozen_region_mode full`（47 层全部 keep，无 near/far skip，
  逐行校验通过），即**纯选择器子空间对照**。
- 主线 producer：`benchmark/RULER/pred_ruler.py` sha256 `802d3ed9…`；
  LongBench 审计 `schema=glm_subspace_longbench_full_audit_v1`，全部 `status=PASS`。
- 审计 JSON：`~/ak_verify_results/glm_subspace_ablation_20261106/full_audits/`；
  诊断评分与脚本：本目录 `ruler_diag_scores.json` / `score_ruler_diag.py`。

## LongBench（分数越高越好；Δ 为相对同请求 native_full）

| 子空间臂 | hotpotqa | gov_report | macro | macro Δ |
|---|---:|---:|---:|---:|
| native_full（对照） | 62.50 | 33.42 | 47.96 | — |
| **nope_random32 s1** | 62.41 | 32.88 | **47.65** | **−0.31** |
| nope_uniform32 | 62.65 | 32.22 | 47.44 | −0.52 |
| jssa_basis32 ※ | 62.17 | 32.76 | 47.46 | −0.50 |
| nope_random32 s0 | 61.63 | 32.85 | 47.24 | −0.72 |
| fasa calibrated32 ※ | 60.51 | 32.05 | 46.28 | −1.68 |
| rope_uniform32 | 59.23 | 29.90 | 44.56 | −3.40 |
| rope_random32 s0 | 57.72 | 28.62 | 43.17 | −4.79 |

## RULER 64K niah_multikey_2（string_match_all，n=100）

| 臂 | 分数 | 说明 |
|---|---:|---|
| native fullKV（`--method none`，eager） | **90.0** | 本次新评分，receipt SHA 校验通过 |
| NFA full 256 维（`--nfa-subspace full`，graph） | **48.0** | 本次新评分，100 行 full-region 校验通过 |
| jssa_basis32 ※ | 23.0 | 不同 producer 版本，**不参与排名** |
| **nope_random32 s0 / s1** | **18.0 / 18.0** | 同版本可比臂中最高 |
| nope_uniform32 | 15.0 | |
| rope_uniform32 / rope_random32 s0 | 0.0 / 0.0 | |
| fasa calibrated32 ※ | 0.0 | |

※ `jssa_basis32` 与 `fasa calibrated32` 的 RULER/LongBench 产物来自另一 producer
（`pred_ruler.py` sha256 `8fd4472d…`，treatment `b8431ac9…` / `75208ad3…`），与主线
`802d3ed9…` 不是同一份代码，**分数不可与主线臂直接比较**。若要考虑 jssa 基，
必须在主线版本重跑后再排名。

## 三条可写入方法段的判断

1. **索引子空间应只取 NoPE 维**。所有含 RoPE 维的候选在两项 LongBench 上
   一致退化（−3.4 ~ −4.8），RULER 直接归零；NoPE 类候选 LongBench 损失 ≤0.72。
2. **32 维随机 NoPE 子空间不劣于等间隔/首尾确定性选维**，且随机种子间稳定
   （s0/s1 RULER 同为 18，LongBench macro 差 0.40，hotpotqa 差 0.78）。
3. **RULER MultiKey 的瓶颈在 full-index 路径而非子空间维度**：full256 已丢 42 分，
   子空间只解释其余 30 分。把 32 维分数（15~23）当作"选维失败"是错误归因。

## 遗留缺口（新任务，不阻塞本次选择）

- `GLM NFA full-index RULER 归因`：native 90 → NFA full256 48 的 −42 缺口，
  需在固定 full region、固定 256 维下分解（选择器召回 / 评分统计 / decode 路径 /
  graph 与 eager 差异）。该缺口大于任何子空间维度收益，优先级高于继续扫子空间。
- jssa/fasa 基需在主线 producer 版本重跑，才可纳入子空间排名。

## 证据分布（跨机同路径，避免误判"证据丢失"）

| 机器 | LongBench 完整臂（各 400 请求） | RULER 臂 |
|---|---|---|
| 本机 pc33 | nope_uniform32、rope_uniform32 | 同左两臂 |
| pc02 | fasa、nope_random32_s0 | fasa、nope_random32_s0、**diag_native_fullkv、diag_nfa_full256** |
| pc40 | jssa、nope_random32_s1 | jssa、nope_random32_s0/s1、rope_random32_s0 |
| pc31 | rope_random32_s0 | nope_random32_s1、rope_random32_s0、rope_uniform32 |

各机另有 7/116/192 行的跨机残片，属于分片 rescue 前的中间产物，正式值以
`full_audits/*.json`（PASS，400 行齐全）为准。
