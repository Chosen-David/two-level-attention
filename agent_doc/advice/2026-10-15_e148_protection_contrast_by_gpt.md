# E148 保护带联合干预的解释边界

稳定 ID：`TL-E148-PROTECTION-CONTRAST-20261015`。固定原件：main `44a98682538209e3fcd44cd12b703012f277b58b` 的 `exp/trace/results/e148_sinkswa_verdict.json`，blob `8f03b452325f75030668633e2ee499503ce8b442`，SHA256 `dd1d92c0aceef24142ab8566ec857d1bbb8e4b29bb2521c17b514501d1eea183`。本条只核来源、算术和解释，不重跑模型或修改原始记录。

B（sink64/SWA64）相对 A（sink128/SWA128）的报告 AVG5 为 **43.88 vs 44.65**；按五项已报告小数复算为 **43.876 vs 44.650，差 −0.774 分**。逐任务 B−A：hotpotqa **−3.22**、musique **−0.26**、gov_report **+0.21**、qasper **+0.44**、repobench_p **−1.04**。保留正负两类结果；A 的 hotpotqa/musique 明示复用 E138，不能把五任务都称为本轮新配对实验。

改变 sink/SWA 同时改变了保护覆盖与非保护预算，因此目前只能描述为**这次联合干预的 aggregate 净负**，不能普遍排除保护几何或 far 饥饿机制。`K2_mid` 的 **768→896** 是 `1024−sink−SWA` 的非保护容量，不是实测 far 选择数量；没有实际分配/选集记录，不能据此宣称 far 已增加同样数量。

**47.91 的原键名是 `prefill_dense_mavg`，不是已核实 FullKV。**实际 producer、模型、输入、scorer 与预测匹配尚未闭合；保留作者的键名和配置描述，不将该数字升级为已验 FullKV 锚点。请补齐已有运行的不可变预测、评分、配置与代码身份映射，再验收跨臂可比性及机制归因；不自动要求新矩阵或第二轮。

E149“近无损”边界沿用已发布的 [TL-E149-AGGREGATE-QUALITY-BOUNDARY-20261015](2026-10-15_e149_aggregate_quality_boundary_by_gpt.md)，mass 探针沿用 [既有 sink 协议说明](2026-10-15_sink_trace_protocol_interpretation_by_gpt.md)，不重复立项。本条不改变默认配置、不触发 sink0 或其他实验；JSON 内的 `user_directive` 是历史记录，不构成新的执行授权。原报告及其正负结果均保留，建议收窄其机制排除表述，而非宣布全部数据无效。
