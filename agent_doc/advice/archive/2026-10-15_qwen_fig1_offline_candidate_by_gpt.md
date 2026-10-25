# Qwen Fig1 固定归档离线入口：CPU候选，非实测图

任务来源：恢复 `GPT-PAPER-20261015`，仅已登记距离分布和near/far token-page
集中度。基线dev `3281913`；整合main `b6f39b2`。没有GPU、付费、权重下载、
采集、MLA/动态研发、Claude、SGLang操作或公开稿件上传。

## 已核缺口与实现

原 `audit_qwen_archive.py` 只做字节/schema校验，`observations.o1_mass`已有
单分布距离记账；O3入口还带自己的quantization/selection arms，不能替代当前
Fig1已登记的区域90%集中度。新增 `exp/paper/qwen_fig1.py`，保留原采集器和
历史数据；workflow的Fig1计划映射改指新入口，旧observations/O3命令保留兼容。

- 固定hotpotqa/musique 0/1、层8/24、原始16中点anchors，不使用256 tail替代。
- 先逐Q-head、128维、FP32 causal softmax，再GQA组均值。两个统计层级独立保存。
- sink/SWA各128并集去重；near边界与修复后的E140 regions逐项测试一致。
- 固定64-token距离箱，每箱mass及有效token数/密度；只保留有贡献箱。
- near/far区域内90%质量的最小token/page数和比例，页为绝对64-token块，
  部分页保留；区域mass另外报告，空或零mass区域为undefined。
- 按source-sequence/layer/head/region/bin聚合有贡献query，保存实际query
  位置及数量，不对缺失距离补0，不跨序列造CI或把head扩展当独立样本。
- 成功的字节/schema校验后再消费Q/K，并再次绑定实际读取字节的SHA256；
  输出始终paper_ready=false，模型/输入/HF实现及文档独立性仍未验收。

## 本机真实状态与执行

既有归档路径仍缺失；没有借用历史聚合值替代原始张量。本机实际执行：

```sh
python exp/paper/qwen_fig1.py --analyze --out /FRESH/status.json
```

退出2，12文件均missing、rows为空、paper_ready=false，完整回执见
`exp/paper/evidence/qwen-fig1-cpu-20261015/local-archive-status.json`。
有原件时可显式加`--root /EXISTING/qwen3-8b`，仍只产诊断，默认不加
`--analyze`只审计。无绘图/重采入口，旧输出拒绝覆盖。没有V，不能声称
attention-output误差或任务质量；未增加paired-random臂或复活旧E140聚合。

## main E139并发整合

两处冲突已合并双方门：`batch_select`保留dev的schema/几何/stream/dtype/
backend signature、fallback统计，同时传播main require_graph异常；benchmark
所有Graph调用显式require_graph=True，仍保留capture/replay与9次计时replay
计数门。新main CPU测试夹具最初缺CUDA上下文及tile/backend元数据，独审
定位后仅补测试替身，不放宽生产门，不把CPU spy当GPU capture证明。

## 验证与限制

相关CPU套件普通/-O各40项通过（另150子例）：Qwen Fig1、归档校验、Graph
契约、required dispatch、E139脚本失败门及paper contract。新增Qwen9项含
均匀/尖峰/零mass、部分页、保护重叠、非法dtype/shape/NaN/Inf/overflow、
因果未来token、归档缺失退出、输出覆盖拒绝、读取后hash变化、anchor子集
和有效query计数。synthetic案例不充当真实Qwen数据。

额外ops全套及共享GQA回归普通/-O各39项通过（另109子例），包括原decode mask、
单token生命周期、异构KV与新增required Graph派发门。

独立只读reviewer审查通过；发现的CPU夹具问题已修正并独立复测。证据、
精确命令、源码hash与边界见上述evidence目录。仍缺实际归档和来源身份，
无真实图、科学结论、CUDA replay数值/性能或完整模型验证。仅推原dev，
用户审核后决定合main。
