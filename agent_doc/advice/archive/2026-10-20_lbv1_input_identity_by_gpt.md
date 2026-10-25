# LB v1 数据指纹未绑定问题与上下文

问题ID：TL-LBV1-INPUT-IDENTITY-20261020；main基线
`5276987c8df987d0620d023cded19a7a8b78e396`。本次是有复现的新发现，
生产身份协议尚未修改，不是已修复或历史污染判定。

## 已确认的边界

`benchmark/LongBench/pred.py:911–914`仅hash逐行`answers + length`，取前12位；
`:448`将其与dataset/行号组成v1样本ID，`:955`传给run-cap。
但`:247`用官方模板格式化整个样本中的`input/context`。
`run_cap_manifest.py:208`沿用该指纹，输出文件SHA只能绑定预测文件字节，
不能反推未保存的源正文。T2的完整ID配对和预测快照检查并不补足这个缺口。

[复现](../results/lb-input-identity93/probe.py)提取实际AST表达式与官方qasper
模板，无Torch/GPU。保留声明的answers/length，改变context、问题、交换两条
同投影样本或使用不同Unicode正文：4例格式化prompt SHA不同，但指纹与
逐行ID全相同；不变/答案变化/length变化3个对照符合预期。
普通/-O输出逐字节一致，结果见[原始观察](../results/lb-input-identity93/observed.json)，
输入SHA为`b6e7faaca3b44c4a1e923cb28deb9819b35dfb67c413fc7c6f488c7984ff0ede`。
这不是SHA随机碰撞，而是被hash的投影本来就忽略正文。

执行：`python [-O] agent_doc/results/lb-input-identity93/probe.py`；CPython3.12.14、
Linux6.18.55/x86_64，仅标准库CPU，无编译器/kernel/驱动测量。独审另用
实际cap_identity_hash固定参数确认同一输入投影仍得到相同cap身份。

不要扩大结论：未tokenize或执行模型，不能声称截断后的token一定改变；
row-swap反例限定两条answers/length相同。没有真实收集快照，尚无证据说明
当前87格混用输入、冠军改变或需要全量GPU重跑。既有评分/CI修复仍有效。

## 复用现有机制后再修复

`exp/paper/receipt.py:83–96`已有`input_ids_sha256`，由可选Qwen timing捕获生成；
默认off，且T2当前不消费它。不能说全库没有输入hash，也不能直接开启重型
timing捕获充当所有模型的通用身份方案。

建议下一版显式区分源样本、格式化/模板、tokenizer/截断及实际token身份，
复用已有token字节编码规范；为新记录增加版本和正文绑定，不静默重解释旧ID。
保留旧预测与失败证据；无正文或token证据的旧数据标为该层身份未知，
不能回填一个hash就追溯认证。先固定真实输入，再验证新旧混用拒绝、同输入
跨臂可比、问题/正文/模板/截断变化失效及sidecar/CI消费链，之后才切换生产。

独立检查见[回执](../results/lb-input-identity93/independent-review.md)。仅标准库
CPU源码绑定复现；无质量、速度、内存、kernel或真实历史重评收益主张。
在飞网格、模型、kernel、人类guide和旧结果均未改。
