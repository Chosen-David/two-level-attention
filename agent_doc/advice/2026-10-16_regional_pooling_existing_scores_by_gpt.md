# 区域池化：既有页分数复算与身份纠正

稳定ID：GPT-PAPER-20261015 / O-POOL-REGION-20261016。
基线main `416232222f429e53e6b30fd95704258fe56dd11d`。已核statistics followup、修正回复及tail advice；本条只补新区域比较，不重复旧请求，旧开放项保留。

## 固定来源
- `exp/trace/results/e140b_pooling_anatomy.json` SHA256 `9ba426e66b56786adc375f11b82fa4bb86fe8be257fbf1f23a249f58feeea0ca`。
- `agent_doc/guide/paper_figs/trace/e140b_pooling_anatomy.py` SHA256 `9d410ec5739d4fc7398d4d7b647a39c3f255044020a76560c67f14f62e086ee3`，匹配JSON记录。

**纠正：avg/minmax都用同32个低频坐标及32查询头均值，不能称二者表示混杂。** 全维softmax页权重参考与两评分器目标不同；漏参考页不等于漏最高分词元。当前被导入的采集器输出float32，但历史manifest未单独绑定该依赖hash。

## 已完成的有界CPU复算
全部4序列×2层×16query保留。沿既有2048/64，预先固定每区域最多32页，分别按avg/minmax降序、页ID破同分；far按页末≤near_lo、near按页首≥near_lo，跨界拒绝。指标为选中页权重/该区域总权重。两区域不拼作同总K方法，亦非生产16/112配额；无L2、质量或GPU测量。独立实现逐行复核128条一致。

按序列聚合，minmax−avg的far均值为+.1033/+.1339/+.1698/+.1483；near为−.0016/+.0210/+.0092/+.0140（HotpotQA0/1、MuSiQue0/1）。支持有限的区域收益差异，**不支持avg近区普遍更优**。far/near饱和26/70条；near实际词元数不同32条，far为0。far正/负/平53/49/26，near18/39/71；均值不等于逐query胜出。128条相关观测不是128个独立样本，反例全部保留。

## 最小后续与验收
优先复用已有Q/K归档及原128锚点，先登记同表示、同区域page cap、同最终K/L2规则的离线“两区域×两评分”对照，分清页权重与token参考；无需另采或扩展全任务矩阵。当前JSON仅两类按结果挑选页含token细节，缺全体token分数/概率，不能据此补算完整L2。先核归档可取性、模型/input/producer及依赖hash、有效位置、保护、精度、页尾与选中IDs。验收需序列级方向、饱和、实际预算及负例；若不支持预期分工就保留结论，不挑样本逼出方向。尚未启动此后续，不变更默认或恢复动态研究。
