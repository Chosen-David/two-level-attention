# NFA 子空间与 Near/Far Skip 证据归档（2026-11-08）

本批把此前分散在本机、共享盘和远端主机的已有结果纳入版本化归档，包含原始预测、逐请求记录、配置、评分汇总、运行/完成回执、独立验收报告、负面结果以及仍未完成项。归档本身不重跑 GPU、不替代各目录中的独立验收，也不改变原实验语义。

## 统一口径边界

- 历史 Qwen `Full` 是 **TLI gates-off**，仍使用 TLI 稀疏路径，不是 dense FullKV。
- 历史 Qwen `Far-only` 会物理淘汰 Far KV；`Near-only` 保留 attention 不可见 staging，随后可老化进入 Far。该语义不同于当前 strict-skip 的“保留 KV、跳过区域计算”，不能直接换名或拼接成同一对照。
- GLM `full` 指 frozen gate mode `full`，用于关闭 Near/Far skip；它不是 dense attention 参照。dense 对照只存在于明确保存 dense arm 的目录。
- 子空间只改变索引/选择所用维度；最终 attention 仍使用完整 K/V。GLM NoPE 坐标不能套用 GQA 的低频 RoPE 解释。
- `screen`、固定 10 例诊断、正式全量和仅有运行计划是四种不同证据等级。`CI` 含 0 不证明等价，点估计落在工程容忍范围内也不证明无损。
- 节点容量、attention-token 比例和候选数量不是实测显存或 TPOT。本批质量结果不提供性能结论。

## 归档组与状态

| 目录 | 已有证据与完成状态 | 主要语义限制 |
|---|---|---|
| `frozen_near_far_full5_independent005_20261029/` | Qwen3-8B，5 任务 1300 配对请求、3900 次三臂生成、64 分片完成；保存 raw、配置、评分、门控/缓存回执和独立 CPU 验收。源码 `3e4a6885604ac3e719ee92a4906f87b532535314`。 | `Full=TLI gates-off`，不是 dense；旧 Far 淘汰/Near staging 语义；AVG5 区间含 0，Qasper 有显著小幅下降；未独立重跑 GPU。 |
| `frozen_near_far_v1_16_independent005_20261030/` | Qwen3-8B，LongBench v1 英文 16 任务 3750 配对请求、11250 次三臂生成、128 分片完成；保存正式 raw、配置、报告及两轮独立验收材料。源码 `08ce44d6027a235080e7ff4127754af6b1ae91e0`。 | 同样不是 dense 或当前 strict-skip；Far/Dual 相对 Full 的 AVG16 下降区间均低于 0，不能称无损；无性能结论。 |
| `glm_nfa_ruler32k_20261105/` | GLM-4.7-Flash，RULER 32K 11 任务，每任务 `n=100`，`full` 与 `dual` 两臂均完整；保存逐样本预测、生成回执、正式评分、源数据 SHA、manifest 和失败历史。AVG 分别为 35.01/34.82。 | 两臂都是 NFA `nope` 索引，不是 dense 对照；`full` 关闭 skip，`dual` 开启 Near/Far 0.05；历史 frozen 语义，不等同当前 strict-skip。 |
| `glm_nope_last32_ruler_aligned_20261101/` | GLM-4.7-Flash，RULER 64K MultiKey-2 100 + LongBench AVG5 1300，共 1400 请求、2800 dense/selected arms，0 失败；有独立验证与重评分。源码 `42821f81a4739d05d8bc81e1d4cc58307e34bb7e`。 | LongBench 为 gate `full`；旧 RULER 未显式指定 gate，实际默认 `dual`，所以 RULER 的 -62 分同时受子空间和 skip 影响，不能作为纯子空间结论。 |
| `glm_subspace_scan_20261101/` | 已有逐请求 raw、配置、日志和 10 份 summary；screen 为 LongBench 每任务前 20 条、RULER 前 24 条，gate mode `full`。源码 `42821f81a4739d05d8bc81e1d4cc58307e34bb7e`。 | 只支持 screen 范围；README 的 `Status: running` 是启动时状态，不表示后续所有计划均完成，也不否定已经落盘的 summary。 |
| `glm_subspace_sweep_20261031/` | RULER 64K MultiKey-2 固定前 10 例，8 个变体，80 cells 完整；独立复核 `status=pass`，保存 dense parity 和 all-token control。源码 `0a72672f3aa3d37f12545eeb8f397bbcc4c91d0b`。 | 固定 10 例诊断，不是全量 benchmark；没有受测 32D 子空间达到 10/10，不能据此宣称 32D 无损。 |
| `glm_subspace_ablation_20261104/` | 保存范围说明、既有证据索引、RULER gate 纠错和 pending 状态。 | 新候选 screen 尚未完成；其中主机不可达/GPU 占用是 2026-11-05 的历史状态，不是当前资源巡检结论。 |
| `glm_full256_ruler_aligned_20261101/` | 只保存运行脚本。 | 没有可确认 raw 或完成回执，不得写成已测完。 |

`agent_doc/task/task_details/` 中对应任务文件记录目标和当时状态；任务计划本身不是实验完成证据。

## 原始结果与重复副本处理

- 两个 Qwen 源目录原为共享盘符号链接，本批归档的是实际文件内容，不是链接本身。共享盘默认 `du` 显示压缩后物理占用，文件完整性以实际字节和 SHA256 为准。
- V1 独立验收目录 `independent_final_20261030T0715/fixed/v1/` 曾重复携带正式 `requests/`、`predictions.jsonl` 和 `raw_index.json`。独立报告已记录 3750 条 raw SHA 与正式 raw 的全量绑定，发布前另抽样 `hotpotqa/0000.json`，两端均为 566625 bytes，SHA256 均为 `1c3ae2e0cd396fb798c6ed2bf74aecefb41b746a47271a378ff355e88eec67a8`。因此本批只保留正式 raw 与独立报告/manifest，不重复收录这三项副本。
- GLM 32K 两个 `*_formal.json.run-*/pred_root/` 是评分 staging 副本。已逐任务核对：`full`、`dual` 各 11 个文件均为 100 行，staging、正式 raw 与 `full_formal.json`/`dual_formal.json` 中登记的 SHA256 三方一致。因此本批删除这两处重复 `pred_root`，保留正式 raw、评分 manifest/receipt、结果和失败历史。
- 复制时排除 `*.lock`、`*.pid`、`__pycache__/` 和 `*.pyc`；没有把这些运行控制/缓存文件当作科学证据。

## 完整性与复现边界

每组目录下的 `SHA256SUMS` 覆盖该目录内除清单自身以外的全部归档文件，可在该目录运行：

```bash
sha256sum -c SHA256SUMS
```

原始 manifest/receipt 中的绝对路径记录历史生产位置，用于追溯，不要求复现机器具有相同目录。模型权重、原始外部数据集和部分历史环境不随本批重复归档；其身份由已有模型/数据 SHA、配置和回执绑定。复现前应使用对应目录记录的源码提交、模型与数据版本，并先阅读该目录的 scope/validation，而不是只读取汇总分数。

## 发布前完整性复核

- 八组 `SHA256SUMS` 共覆盖 11892 个归档载荷文件；连同 8 份清单共 11900 个文件、3367677740 bytes。发布前已逐组执行 `sha256sum --quiet -c SHA256SUMS`，8/8 返回码均为 0。
- 清单与目录实际文件逐集合一致，无遗漏、重复登记、符号链接、`*.lock`、`*.pid`、`__pycache__/` 或 `*.pyc`；最大单文件为 18172546 bytes。
- Qwen Full5 的 64 份 manifest 与 64 份 worker-done identity 完全一致，协议请求数与 `requests/` 实际文件均为 1300；V1 的 128 份 manifest 与 128 份 worker-done identity 完全一致，协议请求数与实际文件均为 3750。
- GLM NoPE 独立报告登记 1400 个请求、2800 个 arm score、7 份 worker-done、0 failed；发布目录实际请求文件也是 1400。GLM 32K `full`/`dual` 的 `incomplete_cells` 均为空，11 任务 × 100 条正式 raw 已在副本去重前逐任务核对。
- 这些检查证明本批发布副本的文件完整性与已有完成回执一致，不等于重新执行历史 GPU 生成，也不扩大各组原有的科学结论范围。
