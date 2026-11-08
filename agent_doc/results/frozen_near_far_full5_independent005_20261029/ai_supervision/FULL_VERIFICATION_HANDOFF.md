# 全量结果独立验收交接

PROJECT_ROOT=/home/wangyuanshuo02/two-level-attention。Task: TL-FROZEN-FULL5-QUEUE-20261029。此文件为既有verification_plan.json的执行清单，不改变实验协议。

生产者为三机冻结worker；独立verify_experiment_result必须由不同会话/owner执行，之后才生成科学结论。当前全量pending。旧screen仅辅证，身份/范围不同不能替代。

1. implementation：以dispatch_plan source/HEAD 3e4a6885604ac3e719ee92a4906f87b532535314与protocol.source_sha256核对实际源码；阅读末256query、mean严格<0.05、独立Near/Far、decode冻结与cache路由。优化关闭；full=TLI gates-off而非dense。检查实际receipt，不以字段名代替路径证据。
2. reference_boundary：检查既有原生模型对照、单元/边界验证证据来源；无Near-only/Both触发时明确该分支未被本轮自然样本实证覆盖，不追加放宽阈值实验。
3. data_integrity：64分片同一identity、各完成标记对应manifest；完整唯一(qasper/hotpotqa/musique/gov_report各200、repobench-p500)，三臂3900。复核数据/模型/配置哈希、prompt token身份、截断、答案和输出token序列。按实际文件SHA256绑定，活跃snapshot不可冒充最终数据。
4. numerical_sanity：独立重算官方评分、每任务配对差与任务等权AVG5（百分制百分点）；核对summary/scores/per_request/raw_index/predictions一致。保存每任务退步/进步/相同数。Musique独立列出，不能用AVG5抵消。Bootstrap沿用已实现5000次、NumPy default_rng(42)、任务分层宏平均；95%CI含0不等于等价，无事前界值不能称无损。
5. measurement_validity：准确率协议同输入同生成上限；不同任务分别QA F1/ROUGE-L/代码相似度，应注明指标。KV比例为节点容量，attention tokens为工作量代理，不主张速度/总显存收益。
6. reproducibility：执行独立评分/门控重算及cache守恒核查，核对实际冻结源；记录无法独立重跑的GPU生成与外部依赖限制，不伪称重复生成验证。

全量到齐后先等待自动validate→analyze→export，检查FINISHED标明pending independent。若汇总异常，精确看daemon错误及finalize锁所有权，再在冻结源和对应远端python环境恢复汇总；不能和自动finalize争写。

门控报告至少包括每臂每任务：request层触发数、layer决策keep/near-only/far-only/both数及分母；Far决策在dual与far-only一致性；实际Near staging不可见/老化迁移证据及未覆盖分支。

最终中文报告绑定独立验收路径、数据manifest及局限。只有全量生产、汇总和独立验收完成才能写COMPLETE.json；掉点也是可接受的真实完成结论。

## 2026-10-29第三轮新增辅证（不替代全量检查）
round_20261029T151704/independent_partial/REPORT.md：固定680配对2040官方评分；independent_inputs/REPORT.md：680输入token独立重建、2040输出解码、73440末步TLI回执、源码路径与资产hash。会话身份见independent_sessions.json。Near-only27层—请求，Both已出现Musique56层11，不能继续沿用旧500子集的Both零覆盖描述。GovReport98 Near-only有8页老化转Far记录，另6个层—请求有1页迁移。
最终验收仍须覆盖1300，核对新字节/资产绑定；完整权重历史hash、逐步GPU张量/选择轨迹/生成复现及部分消费者环境缺口必须保留。不把native_full_parity布尔值解释成dense attention一致，不把32维选择解释成attention也只32维。
