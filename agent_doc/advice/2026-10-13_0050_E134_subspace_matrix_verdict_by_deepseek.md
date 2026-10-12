# E134 三方法子空间矩阵判决（AVG5 + 性能）+ 128K Quest 收官

> 作者：deepseek · 2026-10-13 00:50 · 用户核心指令（2026-10-11）延续收官

## 1. 背景与三种方法

用户指令：实验「粗筛子空间 + 细筛子空间 / 粗筛 full + 细筛子空间 / 粗筛子空间 + 细筛 full」
三种方法在 **AVG5** 上的精度（其中已测过的一种不需重测），任务**拆多卡**跑，并给**性能（@64K，ms/步/层）**。

| 方法 | L1 粗筛 | L2 细筛量化 | 状态 |
|---|---|---|---|
| ① full+sub | 全 128 维 | tail32 | E123 已有 AVG5 47.91（不重测） |
| ② sub+sub | tail32 | tail32 | E133 批 + E134 l1sub 补（本轮打分） |
| ③ sub+full | tail32 | 全 128 维 | E134 新代码 + 远程 5 卡派单（本轮打分） |

## 2. 精度判决（AVG5，Qwen3-8B LBv1 5 任务）

| 方法 | AVG5 | Δ vs ① | 95% CI（bootstrap 10000/seed42） | 判决 |
|---|---|---|---|---|
| ① full+sub（冠军默认） | **47.91** | — | — | E123 基线 |
| ② sub+sub | 47.588 | -0.322 | [-0.510, -0.106] | **显著下降**（上界<0） |
| ③ sub+full | 47.666 | -0.244 | [-0.636, +0.148] | 未显著（CI 含 0 ≈①） |

per_task 配对差（arm − baseline）：
- l2full ③：qasper -0.16 / hotpotqa -0.64 / gov_report +0.28 / musique -0.87 / repobench +0.17
- l1sub ②：qasper -0.50 / hotpotqa -0.22 / gov_report +0.07 / musique -0.37 / repobench -0.59

**解读**：
1. **L1 粗筛子空间（tail32）引入可检测精度损失**：②③ 全任务净差为负，② 达统计显著（-0.32，CI 上界 -0.106）。L1 粗筛覆盖 128→32 维丢失了逐维分块 minmax 的召回。
2. **L2 细筛全维/子空间对精度几乎无影响**：③（L2 全 128 维）≈ ①（L2 tail32），未检出显著差异（CI 含 0）。把 ②→③ 的恢复（-0.322→-0.244）全部来自 L1 维度不变、L2 变化，而③ vs ① 仍未显著 → **L2 子空间不是精度损失来源**。
3. 结论：**论文方法维持 full+sub（mavg 冠军默认）**；L1 子空间掉点不可取；L2 子空间省不出精度收益。

## 3. 性能（@64K，同基三臂，H20-3e steps=60/warmup=40）

| 方法 | ms/步/层 |
|---|---|
| ① full+sub | **1.5284** |
| ② sub+sub | 1.5773 |
| ③ sub+full | 1.5342 |

- 三者差距 ≤3.2%，L1 子空间（tail32 少算维度）理论应省算但 H20 上**未体现为速度收益**（L1 粗筛非瓶颈，② 反而略慢）。
- 精度/性能双口径下 **full+sub 均占优**，无子空间扩展必要。

## 4. 打分工程细节（cap 102 混挂解法）

- 新臂（096 后）runcap sidecar cap_identity 含 **run_timestamp** → 同 treatment 跨批次/跨任务 cap identity 天然不同 → 单目录聚合打分必触发 **102 混挂 fail-closed**（实测 5 个 hash：23be7eb895/9bc79ade82/5d17a1cf12/ba5cc649bf/94280e0078）。
- **解法 = 逐任务独立目录打分**（每目录唯一 strict 文件 → cap_hashes 长度 1 → 通过），再合并 per-task 分数算 AVG5。基线 E123（096 前 legacy 无 sidecar）聚合 + `--legacy-no-runcap` 降级。
- ②③ 每任务 cap_identity_hash 落盘（l2full 23be7eb895/9bc79ade82/5d17a1cf12/ba5cc649bf/94280e0078；l1sub a224375bf0/dda5feef09/62c6940f74/cd90f25f17/4643ba839a）。
- 教训：**跨任务 cap 混挂无法用 --legacy-no-runcap 绕过**（只降级无 sidecar 文件），必须分目录打分。

## 5. RULER 三档 Quest 掉点收官

| 长度 | Quest | FullKV | Quest Δ |
|---|---|---|---|
| 32K | 42.85 | 59.38 | -16.53 |
| 64K | 26.25 | 48.54 | -22.29 |
| 128K | **19.33** | 46.43 | **-27.10** |

- 掉点随长度放大（-16.5 → -22.3 → -27.1）**完全贴合 Quest 论文趋势** ✓。
- 128K 打分前置修复：4 个 .187 重跑产物补 `.tli_gen` 指针（gen 目录 + jsonl + yarn_receipt 齐全但指针未随 rsync 落地）→ score_ruler_formal 双通道识别 11/11。**教训：跨机回收 gen 目录后必须核对 .tli_gen 指针。**

## 6. 产物与脚本落盘

- ② sub+sub 5/5：`exp/trace/results/e134_l1sub/pred_mavg_l1sub/`（qasper/hotpotqa/musique TS 1012180214 + gov_report/repobench E133 批 1012175451 已归集，hash h0a1708cf4e）
- ③ sub+full 5/5：`exp/trace/results/e134_l2full/pred_mavg_l2full/`（hash h2d1dc6ec58）
- 基线 E123：`exp/trace/results/e123_trial_raw/pred_mavg/`
- 判决脚本 `/tmp/e134_subspace_pertask_analyze.py`；verdict `/tmp/e134_subspace_verdict.json`
- 主表更新：`exp/results_ruler/main_table_quest_vs_tli_draft.md` §3 128K 行 + 新增 §6 子空间消融

## 7. 待办

- MoBA/ClusterKV 接入后补主表对比臂（阻塞：MoBA 训练依赖、ClusterKV pylibraft 版本障碍）
- E118 sglang C1/C2/C3 修复（回主仓集成时做）
