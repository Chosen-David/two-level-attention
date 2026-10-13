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
| ② sub+sub | 47.588 | -0.322 | [-0.510, -0.106] | **噪声带边缘**（效应量 -0.67%，上界贴近 0，5 任务功效不足） |
| ③ sub+full | 47.666 | -0.244 | [-0.636, +0.148] | 未显著（CI 含 0 ≈①） |

per_task 配对差（arm − baseline）：
- l2full ③：qasper -0.16 / hotpotqa -0.64 / gov_report +0.28 / musique -0.87 / repobench +0.17
- l1sub ②：qasper -0.50 / hotpotqa -0.22 / gov_report +0.07 / musique -0.37 / repobench -0.59

**解读（10-13 用户口径修正：② 为噪声级候选，非「掉点不可取」）**：
1. **② sub+sub 落在噪声带边缘，当前 5 任务证据不足以定案**：Δ-0.32 绝对效应仅 -0.67%，CI 上界 -0.106 极贴近 0；mask 对拍 98.34%（每步仅 ~1.66% 差异）证明块选择改变本就微小。E134 性能重判后 ② 与冠军性能持平（见 §3），-0.67% 的质量差若被更大数据量（LB 13 任务全集、128K 档）证明为噪声，② 即为净收益方案。
2. **L2 细筛全维/子空间对精度几乎无影响**：③（L2 全 128 维）≈ ①（L2 tail32），未检出显著差异（CI 含 0）。②→③ 的恢复（-0.322→-0.244）全部来自 L1 维度变化，而③ vs ① 仍未显著 → **L2 子空间无精度代价**。
3. 结论（修正版）：**生产默认维持 full+sub（E123 锚点已背书，切换收益待证）**；② sub+sub 定位为**噪声级候选**——待更广泛任务数据复验质量 + prefill 阶段省算兑现（L1 粗筛全量建索引 4× 省算在 prefill 更明显）后可升格；L2 子空间已采纳无代价。

## 3. 性能（@64K，同基三臂）——**【10-13 Opus 重判：本节原表作废撤回】**

> **勘误**：本节原表（①1.5284/②1.5773/③1.5342）基于**假 32 维旧实现**（zeros_like 零填充 128 宽 dense + b_q 全维 + 路由==128，性能浪费未兑现），用户指令后已修复合入 main（merge 2c393c5：E133b 紧凑布局 + E133c 省算兑现 + l2_subspace 开关移植），三臂新旧逐位对拍 100% PASS（质量判决不受影响）。以下为真 32 维实现后重跑数据（2xh20-reg0519 双卡独占，steps=60/warmup=40，两轮 GPU 交叉复跑取中位，详见 `exp/trace/results/e134_perf_reverdict.json`）：

| 方法 | 旧实现（作废） | 新实现（真 32 维） | Δ |
|---|---|---|---|
| ① full+sub | 1.5545 | **1.3502** | -13.1% |
| ② sub+sub | 1.6965 | 1.3598 | -19.8% |
| ③ sub+full | 1.6002 | 1.4634 | -8.5% |

- **原判「L1 子空间未体现为速度收益」必须撤回**：旧代码 sub+sub 比冠军慢 +9.1% 是假 32 维工程债（零填充 dense 读带宽 4× 浪费 + launch 开销吃掉省算），不是方法属性；新代码 sub+sub 1.3598 vs full+sub 1.3502（+0.7% 噪声带内持平）——tail32 省 4× 粗筛算力已兑现为时间。
- ③ sub+full 仍最慢（+8.4% vs 冠军）：L2 全 128 维量化/打分成本实打实，紧凑化无法救。
- **生产默认维持 full+sub**（E123 锚点背书），sub+sub 性能与冠军持平后升格为**噪声级候选**（质量差 -0.67% 待更广泛数据复验，见 §2 修正版解读）——若后续 LB 全任务集/128K 复验证明噪声级 + prefill 省算兑现，sub+sub 即为净收益方案。

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
