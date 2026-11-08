# Regional scoring 数据包

本目录由 `../build_data_package.py` 直接读取 `../../B_final/quality/` 的 2000 份原始请求生成，没有重新执行 GPU，也没有手抄均值。

## 文件

- `quality.jsonl`：2000 行逐样本分数、预测、输入身份、实际输入长度、生成 token 数、完整 token 数组 SHA、gate counts，以及 raw 文件相对路径和 SHA。
- `plot_data.csv`：2000 行 tidy 数据；每行一个 `cell × dataset × row_index`，携带完整区域选择配置和逐样本分数。
- `manifest.json`：5 个 cell 的真实 spec、源码提交、manifest identity 和完整性范围。
- `metadata.json`：评分、统计和复用边界。

完整 generated IDs 仍保存在 `raw_request` 指向的原始 JSON 的 `result.generated_ids`；本包只重复保存其 SHA256，避免再复制约 1 GiB token 数组。预测文本和分数保留在 `quality.jsonl`，官方重评分与配对统计见上级目录的独立验收文件。

## 重生成与校验

在 `nfa_ablation_20261030/` 目录运行：

```bash
python B_final_review/build_data_package.py
```

当前独立复核结果：2000 行唯一键完整，10 个 cell×task 均值与原独立报告四舍五入值全部一致；每 cell×task 首尾共 20 份请求的 raw SHA、完整 token SHA 和分数回读一致。

## 解释边界

- `2d2a466b4e9a` 是 NFA tail32、Near/Far=Avg/MinMax 的生产默认对照，不是 dense FullKV。
- 五格均为 gate `mode=full`，即关闭 Near/Far skip；`reclaim=true` 是旧路径原始配置，不改称当前 strict-skip 新实测。
- 该配置与当前 sink4、SWA128、K1=128、K2=1024、near4096、no-skip 的区域评分问题匹配；复用结论仍限定于旧源码 `4dab6e8c` 及独立报告声明的范围。
- 每格每任务只有单次生成；paired bootstrap CI 不含重复运行方差，未做多重比较校正，CI 含 0 不证明等价或无损。
