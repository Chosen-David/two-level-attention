# 680配对固定子集质量观察（非最终AVG5）
独立证据：[REPORT](independent_partial/REPORT.md)，实际独立会话与hash见independent_sessions.json。仅消费其usable-with-scope范围，未引入活跃新增样本。冻结阈值0.05、协议和源未改。

|任务|固定样本数|Far−Full (pp)|Dual−Full (pp)|
|---|---:|---:|---:|
|Qasper|107|−0.957999|−0.957999|
|HotpotQA|107|+0.467290|+0.467290|
|Musique|107|+0.002148|+0.002148|
|GovReport|107|+0.278864|+0.250850|
|RepoBench-p|252|−0.059524|−0.059524|
|子集任务等权宏均值|680|−0.053844|−0.059447|

该固定子集确有观测均值下降，不能报告“完全不掉点”。它不代表全1300条结果，也没有事前等价界值或全量推断结论。Full指TLI gates-off，不是dense attention。QA任务为F1、GovReport为ROUGE-L、RepoBench-p为官方代码相似度，不把异质指标统称准确率。

Musique 107条：两个处理臂均0降/1升/106平，唯一row82微小F1改善但回答仍错误，不能解释为多跳推理提升。Both自然覆盖row56第11层，near mean0.0462979525、far mean0.0344907008，均严格小于0.05；三臂评分0，Dual/Far输出相同。只有这一个Both样本不支持一般质量保证。

Dual相对Far仅GovReport98变化，样本分数下降2.997513pp。第30层Near-only；Near丢attention但保留staging，与Far路径区别相关，不将单一案例外推一般因果规模。

Dual门控：Near-only 27/24480层—请求（0.1103%）、13/680请求（1.9118%）；Both 1/24480（0.0041%）、1/680请求（0.1471%）。Far-only3174/24480；逐任务分母和记录见独立报告。缓存220320层快照账面守恒检查通过；这不等于GPU张量或生成复现。

独立评分2040/2040与记录误差0，746绑定文件及72源码hash通过。输入token和路径补验另行开展，未完成前不扩大可用范围。全量尚未完成，最终bootstrap与全量AVG5等待自动汇总后独立核验。

## 本轮独立输入/路径补验已完成
[独立报告](independent_inputs/REPORT.md)：680输入token重建、2040输出解码/停止条件、73440末步TLI回执通过；7个Near-only层—请求出现Far增长记录，GovReport98对应8个64-token页迁移。仍不等于GPU张量或生成复现。完整权重、逐步选择集合及部分运行环境限制见原报告。
