# TL-LBV1-INPUT-UNIFORMITY-20261021

Date: 2026-10-21

## Plan

来源：用户要求固定输入、同预算和同语义公平比较，以及
`TL-LBV1-INPUT-IDENTITY-20261020` 的未决聚合闭包。基线
`ca1b8c15a0e8c31e91dd50f98c1dfd5676292457` 已为每个预测文件记录
LBv1 v2 源行、模板与实际 token 身份，但需核对聚合器是否真的消费这些证据，
而不只把它们抄入结果元数据。不得伪造历史身份，不启动GPU任务。

## Progress

确认 `benchmark/LongBench/eval.py` 修前只检查 cap 单一性，并把每文件输入身份
写入 `_meta`；相同 dataset 的不同 method 即使源行、模板或实际 token 哈希不同，
仍会写出同一 `result.json`。入口 AST 基线实际得到 `baseline_gate_calls=0` 并失败。

已在标准库 `run_cap_manifest.py` 增加按真实 manifest dataset 分组的 fail-closed
门禁，并在任何结果发布前调用：同一 dataset 的多个 strict 产物必须三轴完全一致；
v2 与 legacy strict 混挂、v2 闭包缺字段或任一轴漂移均拒绝。不同 dataset 可使用
各自身份，单文件及全 legacy strict 历史目录保持可读，不将 legacy 冒充 v2。

专项回归普通/-O各7/7，通过相邻 v2 身份与回填回归普通/-O各15/15；
`compileall` 与 `git diff --check` 通过。完整 `eval.py` 运行仍受当前环境缺少
`jieba` 阻断；节点也无 `torch`/GPU/LongBench源/模型，故历史重评分、质量、
端到端时延、显存和性能均未验证或主张。详见
[中文证据](../../advice/2026-10-21_lbv1_input_uniformity_by_gpt.md)。
