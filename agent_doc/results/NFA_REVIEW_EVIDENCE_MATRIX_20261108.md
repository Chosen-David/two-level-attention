# NFA 审稿证据矩阵与最小补测清单（2026-11-08）

本文只登记证据、语义差异和最小动作，不修改论文、图或 Overleaf，也不把历史计划当作完成回执。目标是回答具体审稿问题；已有 raw 先审计复用，只有语义不符或 raw 不可恢复的单元才补测。

## 1. 分类与统一口径

### 1.1 证据处理分类

- **A：可直接复用。** raw、逐样本结果、配置、源码身份和完成/验收记录齐全，并且语义与目标问题一致；仍需在数据包中登记路径与 SHA。
- **B：已做，需找回或重汇总。** 已有生成结果，但需找回 raw、补配置映射、重评分、重算统计或转换数据格式；不因缺一张汇总表而重跑 GPU。
- **C：必须补测。** 目标样本/配置没有生成结果，或缓存、gate、预算、源码语义变化使旧结果不能回答当前问题。
- **D：增强项。** 有助于解释机制，但不阻塞核心质量与性能结论。

“A/B”不是免验收标签。任何复用都必须先通过输入身份、样本数、配置、源码、评分和完成状态检查；检查失败才转入 C。

### 1.2 当前投稿工作点

- 模型：Qwen3-8B；最终 attention 使用完整 K/V。
- sink=4、SWA=128、K1=128、K2=1024（含保护区）、near pages=32、near length=4096、page=16。
- 子空间：生产 tail32；Near/Far 报告顺序为 **Avg/MinMax**。
- strict-skip：gate 从第一次 decode forward 起生效并冻结；不加入旧首步 seed/core-token 补偿；物理 KV `retain`。
- no-skip 必须显式关闭 gate；自然未触发不等于强制 no-skip。
- Full-dimensional indexing 是同一稀疏算法的全维索引，不是 dense FullKV；历史 Qwen `Full=TLI gates-off` 也不是 dense。
- 当前研究仓证据发布快照为 `fbe455951333f264288699d81aac4c98c14fd20a`；strict-skip 功能验收锚为 `2f702c222186bdf17de996e981c52af467957ee5`，旧 seed-on 对照锚为 `f5f3df3c4fe4c4af39c9074db0baaa5e00770eab`。新实验必须另存实际 HEAD 与 diff，不用本清单日期代替源码身份。
- SGLang `two-level-indexer=4f226f1f0a51e9fa6890d7d992466db9a4128f97` 已包含在远端 `main=5ba097eb3c15cb3d265bf33d9da474854ec52fe2` 中；远端工作分支已删除，后续复现基线指向远端 main，而不是假设仍存在独立分支。

## 2. 核心问题逐项矩阵

| ID | 审稿问题与目标单元 | 已有 raw / 配置 / 源码身份 | 当前证据能回答什么 | 分类 | 最小动作 |
|---|---|---|---|---|---|
| Q0a | FullKV、旧 seed-on、当前 strict-skip 的语义桥接；HotpotQA 200、GovReport 200、RULER 64K MultiKey-2 100 | `nfa_strict_skip_seed_removal/` 保存旧/新两臂的逐样本 token、预测、gate counts 与输入 SHA；旧 `f5f3df3c`，新 `2f702c22` | HotpotQA 仅 5 行；full/far/dual 均 token 一致。RULER 仅 32K `niah_multikey_1` 与 `cwe` 各 5 行，不是 64K MultiKey-2；MultiKey-1 有 3/5 行 token 改变，不能概括为完全无差异 | B（已有小规模）；C（目标三臂） | 保留 5 行桥接作为 smoke。冻结目标 500 个输入后，逐臂审计是否已有同输入 raw；只补缺失臂。输出逐题分数、完整 token、四态和首步支持集 |
| Q0b | 三臂中的 dense FullKV 质量锚 | `e109_full_lbv1.json` 及对应 LongBench raw 含 HotpotQA/GovReport；`e116b_ruler65536k_formal_fullkv...` 含 64K MultiKey-2 的 100 条预测与分片/合并文件 | 是真实 dense FullKV 历史生成候选，不是 TLI gates-off | B | 对目标输入逐行核 dataset row、input/answers SHA、生成上限、模型 revision、预测和官方评分；一致则直接重评分，不重跑 |
| Q0c | 当前 strict-skip 的真实正确性与首步语义 | `nfa_strict_skip_seed_removal_verdict.json`：99 tests + 5 subtests；32K B4/B1 eager/Graph、自然 far skip、集合/地址 oracle、token identity；大 raw 有路径和 SHA | 证明指定 smoke 输入上的首步 strict、KV retain、eager/Graph 一致，不证明目标 500 样本质量 | A（限定 smoke） | 直接作为实现正确性证据；不得外推为全量质量或所有长度性能 |
| P0a | 完整 decode TPOT：32K/64K/128K × B1/B4 × eager/Graph × FullKV(FA2)/NFA；512 tokens；5 固定输入 × 3 独立重复 | 现有 `benchmark/efficiency/results/.../e2e_*` 与 `performance_abba_manifest.json` 保存单/少输入 ABBA、逐步数据、token 和 hygiene；四配置六长度文件明确是纯算子 | 可回答已测输入/源码上的局部性能，不能满足新矩阵的输入数与独立重复数 | B（既有单元）；C（完整矩阵） | 先建立 24 个比较单元的已有覆盖表，保留合法单元；缺失单元按同输入 ABBA、独立进程、3 warmup、锁卡和 30 秒旁路监控补齐。纯算子只作 breakdown |
| P0b | skip benefit：no-skip vs prefill-guided adaptive skip，32K/64K/128K，优先 B1 | 四配置纯算子矩阵含 full/tail × no-skip/natural-skip、真实输入 SHA 和 gate epsilon；已有 e2e ABBA 含自然 skip 样本 | 有局部自然 skip 和真实计算路径证据；不能拿强制 both-skip 或纯算子替代完整 adaptive decode | B（局部）；C（主面板） | 复用合法自然 skip 输入；每个输入保存 gate mass、epsilon、sN/sF、四态、首/后续一致性和实际跳过 kernel/region work。无自然 skip 的输入如实记 zero-skip，不改 flags 制造加速 |
| A1a | 子空间 × skip 四格质量：full/tail × no-skip/natural-skip | 四配置结果 `nfa-four-config-six-length-operator-summary.json` 的 scope 是 `pure CUDA operator chain; not end-to-end decode TPOT`；没有对应目标质量 raw | 纯算子四格已完成；质量四格未完成 | A（纯算子）；C（质量） | 在相同诊断输入上仅跑缺失质量四格；no-skip 显式 `gate-epsilon=0`，natural-skip 保存实际四态，不重跑纯算子矩阵 |
| A1b | `(c) Subspace choice`：Full、Low-frequency、High-frequency、Random；no-skip、相同 `dc` | E90/E134 与 GLM scan/sweep 有历史子空间 raw；当前 `nfa_indexer.py` 的 tail/highfreq 遵守 RoPE 配对，random 为固定 `seed=20261002` 的 32 维坐标 | 历史结果可作有限配置候选；GLM screen/固定 10 例不能充当全量。当前 random 只有一个硬编码种子 | B（已有单种子/旧配置）；C（统一直接对照与三种子） | 先核 E90 的维度、no-skip、输入和 raw，合格则重评分。补齐同 `dc` 的缺失臂；Random 至少 3 个预登记固定坐标种子并保存实际坐标，不能事后挑最好种子 |
| A1c | GLM-4.7-Flash NoPE/RoPE 子空间证据 | 已发布 `glm_subspace_scan_20261101/`、`glm_subspace_sweep_20261031/`、`glm_nope_last32_ruler_aligned_20261101/`、`glm_nfa_ruler32k_20261105/` 及 SHA256SUMS | screen、固定 10 例、全量 NoPE-last32 和 32K gate 对照分别成立；旧 RULER 默认 dual，不是纯子空间-only；NoPE 不得解释为 GQA 低频 RoPE | A（各自范围）；D/C（若要求新的干净全量子空间面板） | 主文优先复用已完成且语义匹配的 LongBench/诊断结果；只有审稿明确要求干净 RULER 子空间-only 时，才用 gate `full` 补缺失 finalist，不重跑已有 NoPE 全量 |
| A2a | `(d) Regional scoring`：Near/Far = MinMax/MinMax、Avg/Avg、Avg/MinMax；固定其余配置、关闭 skip | 已发布 `nfa_ablation_20261030/B_final/`：5 cells × 2 tasks × 200 = 2000 raw；`B_final_review/` 独立官方重评分误差 0、paired bootstrap B=10000，并有可重跑数据包。manifest 明示 tail32/sink4/SWA128/K1=128/K2=1024/near4096/mode=full、git `4dab6e8c` 和逐文件 SHA | 与当前区域评分问题匹配；生产默认不是 dense。CI 为单次生成、未校正多重比较，CI 含 0 不证明等价 | **A（限定原协议）** | raw、manifest、review、配置映射和 tidy 数据包已归档并做 SHA 复核；直接抽取所需三臂及全局对照，不重跑 2000 请求 |
| A2b | 区域配额、边界页和摘要的机制解释 | B_final 另含全局 MinMax/MinMax、无边界页等 cell；历史 trace/observation 有局部分数与 mass | 可解释特定干预，但不同 scorer 混池时不能声称只改变一个数学因子 | B/D | 从同一 B_final manifest 导出 tidy rows 和配置差分；不新增与核心三臂无关的全笛卡尔积 |
| A3 | 真正 no-L1 单级对照 | 当前参数没有 `no_l1`/`disable_l1`/`bypass_l1`；历史单池 aavg 仍有 L1。`near_pages <= K1` 构造约束也意味着调大 K1 不等于绕过 L1 | 真正 no-L1 未实现、未验收 | C（实现 + 小对拍 + 实验） | 先实现显式单级路径，记录全合法候选、实际 L2 工作和成本；用小输入证明 L1 page stats/Top-K 确实消失且最终预算/tie 合法，再只跑既定诊断集。不得把单池或大 K1 改名 no-L1 |
| A4 | K2=256/512/1024/2048 消融 | E139/E177 等有历史 K2 性能/质量结果，但参数集合和旧工作点不完全等于当前 sink4/page16/strict-skip | 历史趋势可参考，不能直接填入当前四点表 | B（历史）；C（当前缺失点） | 先按实际 manifest 映射已有 1024/2048；固定同一批输入与其余参数，只补 256/512 及未通过映射的单元，不扩到 4096/8192 |
| M1 | 真实 L1/L2/Gather 选择轨迹 | `tests/test_e140_observation.py`、selector/cache 中有集合与地址 oracle；strict verdict 已验证 P/R/X/A 和逻辑 token→物理 slot 反演 | 有 correctness oracle，没有面向论文数据包的完整 `selection_trace.npz` | C（导出）；D（更多样本） | 在少量预先固定的真实请求/层/decode step 上导出：逻辑位置、区域、页映射、L1 scores/入选页/保送来源、联合候选、L2 scores、最终 IDs、Gather 物理位置与原生 KV 对应。优先 2–3 个代表轨迹，不做全数据集 trace |
| O1 | observation 原始数组与 GQA/MLA 机制解释 | `TL-NFA-OBSERVATION-PLAN-20261104.md` 和 E140/E141/E158 trace 有历史数据；旧生产观察配置含 sink128/page64 | 可回答旧配置下的机制观察，不能无条件套到当前 sink4/page16 | B/D | 先发布原始数组、坐标和实际配置；仅当核心图必须当前工作点时补一个代表输入，不恢复大规模 observation 矩阵 |
| O2 | gate mass、四态和 MultiKey-2 失败案例 | strict seed-removal 有小样本 gate/token 轨迹；历史 RULER/LongBench raw 有逐样本预测；GLM/Qwen 归档含负面结果 | 可构造真实失败案例候选，但当前 64K MultiKey-2 目标案例需与 Q0 同批对齐 | B，随 Q0 落地 | 从同一 manifest 选预先定义规则的失败/成功例，保存 mass、阈值、四态、首步及后续支持集；不事后挑有利案例 |
| B1 | baseline 官方版本与可复现性 | `research/docs/论文baseline矩阵补齐报告_20261031_by_claude.md`、E109/E116/E119 等 raw/manifest 记录模型、配置和运行身份；SGLang 最新 main 已核实 | 有部分版本记录，尚需统一成机器可读索引 | B | 汇总官方仓库/版本或 commit、模型 revision、数据 SHA、命令和 scorer SHA；缺哪项补哪项，不重跑已完整 baseline |
| B2 | LongBench v2 实际长度与截断 | `e145d_lbv2_verdict.json` 明示 503 MCQ、official-strict parser、31.5K truncation；已有 503 题 raw/汇总 | 已有总体截断口径，但审稿所需逐样本实际长度分布需从 raw 重算 | B | 从冻结输入导出原始 token 数、截断后 token 数、被截断样本数与输入 SHA；不重跑 503 题生成 |
| S1 | SGLang 集成基线 | 本地分支是远端 main 祖先，`origin/main` 含 `python/sglang/srt/layers/attention/tli/`；远端 main=`5ba097eb3` | 代码基线明确，不包含本清单要求的新性能结论 | A（源码身份） | 新复现只记录实际远端 main、环境和支持矩阵；不再寻找已删除的远端工作分支 |

## 3. 已发布证据的边界

以下结果已经推送研究仓 main，并有逐文件 SHA256SUMS；它们是复用输入，不代表当前 P0 已全部完成：

1. **Qwen Full5 / LongBench v1 16 任务**：旧 TLI gates-off、Far 物理淘汰、Near staging 语义。可回答历史 near/far 策略质量，不能直接改名当前 strict-skip；AVG 区间和显著负差按原报告保留。
2. **GLM RULER 32K full/dual**：两臂都是 NFA NoPE 索引；`full` 仅关闭 gate，不是 dense。
3. **GLM 子空间 scan/sweep**：分别是 screen 和固定 10 例诊断；没有受测 32D 变体达到 10/10，不能写“32D 无损”。
4. **GLM ablation/full256 目录**：分别只有范围/pending 材料和脚本，不是已完成实验。
5. **四配置六长度矩阵**：只支持纯 CUDA operator chain；不能替代完整 decode TPOT 或任务质量。

完整归档说明见 `NFA_EVIDENCE_ARCHIVE_20261108.md`。复用时必须同时携带该目录的 scope/validation，不得只摘平均数。

## 4. 最小执行顺序

### 阶段 R：CPU 复用与数据找回，不启动 GPU

1. 为 Q0 冻结 500 个目标输入清单，逐臂连接已有 FullKV、seed-on、strict-skip raw；输出 `reuse / rescore / rerun` 判定及原因。
2. 将 B_final 的 raw、manifest、独立 review 和配置映射纳入版本化证据包；重算所需三臂 tidy data，不重跑 2000 请求。
3. 建立 24 个性能比较单元的覆盖表，只把同源码语义、同输入、同 decode 长度且有 hygiene/token 的历史单元标为可复用。
4. 导出 LongBench v2 长度分布、baseline 版本索引和已有 observation 原始数组。

### 阶段 I：最小入口与语义对拍

1. Random 多种子：参数化种子并落盘实际坐标；旧单种子结果保留其原坐标，不回填成新种子。
2. 真正 no-L1：先做集合/成本对拍，证明 L1 工作消失，再允许进入质量或性能实验。
3. selection trace：先在单请求单层单步验证逻辑 ID、页和物理地址一致，再扩到少量代表轨迹。

### 阶段 E：只执行判定为 C 的单元

1. Q0 三臂目标诊断集，优先完成质量/语义桥接。
2. 当前 P0 性能缺失单元，严格锁卡 ABBA；性能与 profile/traffic 分进程，避免互相污染。
3. 子空间×skip 质量四格、当前 K2 缺失点和区域评分中未通过复用映射的单元。
4. 每个单元完成后立即保存 raw、配置、失败记录和独立验收；负面结果不触发事后扩参或换输入。

## 5. 数据包契约

最终每个目标问题至少能追溯到以下字段；缺失单元显式标 `missing/blocked`，不生成占位曲线。

| 文件 | 最小内容 |
|---|---|
| `manifest.json` | run/cell ID、分类、状态、源码 HEAD+diff SHA、模型/数据/输入 SHA、命令、全部参数、环境、样本清单、父 raw 路径与 SHA |
| `quality.jsonl` | task、row/sample ID、arm、prediction、answers、score、完整 generated token IDs、首个分叉位置、官方 scorer 版本 |
| `latency.jsonl` | input ID、repeat、ABBA position、warmup、decode steps、逐步/汇总 TPOT、execution、batch、method、gpu_hygiene、token identity |
| `component_times.csv` | page stats/L1/L2/Top-K/union/gather/attention 的 device 时间、调用数和测量层级；明确 operator/module/decode-runtime |
| `gates.jsonl` | request×layer 的 near/far mass、epsilon、sN/sF、keep/near/far/both、freeze timing、首/后续一致性、实际 skip work |
| `selection_trace.npz` | 逻辑/物理 ID、区域、页、L1/L2 scores、候选来源、最终集合、Gather 地址及 shape/dtype/坐标元数据 |
| `metadata.json` | 单位、评分与统计协议、CI 方法、限制、负面结果、复用谱系 |
| `plot_data.csv` | 由上述 raw 程序生成的 tidy 数据，不手抄汇总数字 |
| `README.md` | 复现命令、校验方法、适用范围、known missing；区分 smoke/screen/diagnostic/formal/planned |

## 6. 当前结论

- **不需要重跑的项目**：B_final 区域评分 2000 请求、已有 FullKV raw、已发布 Qwen/GLM 历史结果、四配置纯算子、strict-skip 正确性 smoke、LongBench v2 生成本身。
- **真正阻塞 P0 的项目**：目标 500 输入的当前三臂桥接缺单元、完整 5×3 decode TPOT 覆盖、质量四格、当前 K2 缺失点、真正 no-L1、Random 多种子和真实选择轨迹导出。
- **不自动扩张的项目**：GLM 新全量矩阵、更多 observation 样本、96K/256K/512K e2e、额外 K2/阈值/保护层网格，均不是当前核心问题的默认补测范围。
