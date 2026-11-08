# LongBench v2 实际长度与截断审计

本包回答一个限定问题：既有 Qwen3-8B LongBench v2 503 题实验在 `cap=31500` 下，逐样本原始 prompt 有多长、多少样本被中部截断、实际送模长度是多少。它只做 CPU 重建与身份核验，**不重跑 503 题生成，不启动 GPU，不修改论文、图或 Overleaf**。

## 口径

- 冻结源输入：503 个官方 `_id`，源文件 SHA256 固定在 `manifest.json`。
- 原始长度：官方 0-shot prompt 在 Qwen3 tokenizer 下的 token 数，尚未做中部截断和 chat 包装。
- 截断：原始 prompt 超过 31500 tokens 时，保留首 15750 与尾 15750 tokens，按生产代码 `decode(..., skip_special_tokens=True)` 拼接。
- 实际送模长度：截断后 prompt 再经过 Qwen3 chat 包装，并按生产 `Choices:` 锚点拆成 material/question 后的 token 总数。
- 每行保存 `actual_input_ids_sha256`，编码复用生产 `input_ids_sha256` 的 signed-int64 little-endian 口径；不复制整段上下文或完整 token 数组。
- 数据集的 `length` 字段只是 coarse label，不能替代数值 token 长度。

旧预测 JSONL 没有保存逐样本 token 数或输入 IDs，因此这里是**从冻结输入、tokenizer 和生产函数确定性重建**，不是声称旧 raw 原本含有这些字段。历史模型 revision 不从本地目录名猜测；身份以 tokenizer/config 文件 SHA256 为准。

## 文件

- `lengths.jsonl`：503 行逐样本长度、截断标记、material/question 长度和输入 IDs SHA。
- `plot_data.csv`：同一逐样本数据的 tidy 表，可直接用于分布图；本包不生成或修改论文图。
- `summary.json`：由上述行程序计算的分位数、截断计数、按源 `length` 标签分组统计，以及三臂 raw 身份/官方 parser 复核。
- `manifest.json`：源数据、tokenizer、代码、raw 的 SHA，运行命令和环境覆盖项。
- `metadata.json`：字段定义、单位、E109/E145d 适用范围与限制。
- `raw/{fullkv,mavg,aavg}.jsonl`：既有 E109 三臂 503 行预测，仅用于 `_id`、answers、行序和官方 parser 闭环；不是新生成结果。
- `build_length_audit.py`：唯一派生入口；所有统计均由代码从 raw 和冻结输入实时计算。
- `SHA256SUMS`：本包除自身外全部文件的完整性清单。

## 重生成与校验

在研究仓根目录运行：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/wangyuanshuo02/two-level-attention/.venv/bin/python \
  agent_doc/results/nfa_lbv2_length_audit_20261108/build_length_audit.py
sha256sum -c agent_doc/results/nfa_lbv2_length_audit_20261108/SHA256SUMS
```

默认读取：

```text
/home/wangyuanshuo02/datasets/LongBench/data/lbv2.json
/home/wangyuanshuo02/Qwen3-8B
```

可用 `NFA_LBV2_SOURCE_JSON` 和 `NFA_QWEN3_TOKENIZER` 指向字节相同的副本。脚本会 fail-closed 校验源数据、三臂 raw、样本数、行序、answers 和历史官方分数，不会联网下载模型或数据。

## 边界

- 该口径是历史内部 31.5K 实验，不是 LongBench v2 官方 120K 协议；绝对分不能与官方榜单直接比较。
- E109 raw 用于身份和评分复核；E145d 与源 503 题及 cap 相同，但本包不重评分、不替换 E145d 结果。
- 长度重建不能补足旧生成的完整 generated token IDs，也不能用于反推首个 token 分叉。
