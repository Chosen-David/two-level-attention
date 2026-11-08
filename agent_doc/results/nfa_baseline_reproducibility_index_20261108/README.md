# NFA baseline 版本与可复现性索引

本包回答审稿问题 B1：现有 Qwen3-8B baseline 的官方来源、固定版本、适配边界、formal 结果与 sidecar 身份分别是什么，哪些只需找回同步，哪些确实是实验缺口。它只读取当前发布快照并计算索引，**不重跑生成、不重评分、不启动 GPU，不修改论文、图或 Overleaf**。

## 口径

- 官方代码直跑、修改版 Qwen3 quality adapter、论文协议语义复现和 source-only 候选分开登记。
- `archived-complete` 要求 result、manifest、receipt 均在快照中，任务数完整，receipt 为 success，且 result/manifest SHA 回指一致。
- `reported-not-archived` 表示报告引用存在但当前快照缺文件，动作是找回或同步，**不等于实验没做**。
- `legacy-partial` 和 `archived-partial` 不用于冒充完整 formal 单元。
- 本地模型路径不等于 Hugging Face revision；历史 scorer/formal script SHA 不用当前文件哈希替换。未闭合的生成命令、模型 revision 保持显式缺失。
- AsyncTLS 是论文协议语义复现；HISA 是语义复现而非官方 kernel 执行；FFD 不支持 Qwen3；Quest/FASA/MInference 的 matched-files 范围不扩张为整个 adapter 的未修改证明。

## 文件

- `baselines.jsonl`：逐方法上游仓库/commit、matched-files、适配与执行边界、来源文件 SHA。
- `evidence.jsonl`：逐结果单元的状态、任务和样本数、result/manifest/receipt SHA、数据 SHA、模型与 scorer 身份、恢复动作。
- `summary.json`：由上述行程序统计的完整单元、恢复单元、integration-only 和未闭合身份数量。
- `manifest.json`：发布快照 HEAD、来源文件和构建脚本身份、运行命令。
- `metadata.json`：状态定义、身份规则与覆盖范围。
- `SHA256SUMS`：本包除自身外全部文件的完整性清单。

## 重生成与校验

在研究仓根目录运行：

```bash
python3 agent_doc/results/nfa_baseline_reproducibility_index_20261108/build_baseline_index.py
sha256sum -c agent_doc/results/nfa_baseline_reproducibility_index_20261108/SHA256SUMS
```

脚本只依赖 Python 标准库。结果文件缺失时保留恢复项，不因缺一张汇总表而触发任何实验。

## 边界

- 本包不生成新的质量均值，不拼接不同 generation、预算或模型口径。
- RULER 完整单元为 11 任务，LongBench v1 为 16 任务；样本数来自结果 JSON，而非从报告手抄。
- 当前快照中 Quest 128K 缺 formal sidecar；部分报告引用的 LongBench 正式 JSON 和共享 RULER 汇总尚待找回同步。它们是 B 类恢复工作，不自动转为 GPU 补测。
