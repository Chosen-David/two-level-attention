# RULER 收口证据等级补充建议
作者：GPT；日期：2026-10-11 UTC；接续既有 051/057/059 身份验收，不新建实验任务。
核查源码与产物：`093e44c21e8d2b40cbae1978abbfcf521d72f5d9`；发布前复核 `be3f7a3368017ca9ab6be7a2c46992c0fdc9f187`。

## 问题与已核事实
[ERRATA](https://github.com/Chosen-David/two-level-attention/blob/093e44c21e8d2b40cbae1978abbfcf521d72f5d9/exp/trace/results/ERRATA.md#L7-L20) 将 E116b 的 producer 11/11 证据用于“三臂同批同档”“分数无污染”及旧 E119 排序继续有效，结论需要按臂、按代限缩。
- E116b mavg/aavg 的 32/64/128K 六份 receipt 均登记 native、producer 11/11；128K 登记有效 factor=4.0。这里核的是已发布回执字段，不冒充独立宿主执行验证。
- FullKV [32K](https://github.com/Chosen-David/two-level-attention/blob/093e44c21e8d2b40cbae1978abbfcf521d72f5d9/exp/trace/results/e116b_ruler32768k_formal_fullkv.json.receipt.json#L20-L38)、[64K](https://github.com/Chosen-David/two-level-attention/blob/093e44c21e8d2b40cbae1978abbfcf521d72f5d9/exp/trace/results/e116b_ruler65536k_formal_fullkv.json.receipt.json#L20-L38)、[128K](https://github.com/Chosen-David/two-level-attention/blob/093e44c21e8d2b40cbae1978abbfcf521d72f5d9/exp/trace/results/e116b_ruler128k_formal_fullkv.json.receipt.json#L20-L38) 均为 legacy-partial、producer 0/11、operator_declared，不能概括为三臂 producer 闭合。
- 对比 E119 与 E116b 的 128K receipt，同名任务的 `source_sha256` 字段：FullKV 11/11 相同，mavg/aavg 各 0/11。本项仅比较已发布字段，未重新哈希 raw；新稀疏臂回执不能追溯绑定旧稀疏臂。
- 既有 [E119 同时代佐证](https://github.com/Chosen-David/two-level-attention/blob/093e44c21e8d2b40cbae1978abbfcf521d72f5d9/exp/trace/results/e119_128k_arm_provenance_corroboration.md#L14-L25) 保留运行佐证，同时明确 051 仍 provisional，不能升级 closed。

## 最小处置与验收
保留原始产物及历史分数，补充分层说明：修改 factor 声明不改变固定预测文本的评分运算，但这不足以证明生成协议一致或比较归因闭合。当前证据没有证明数值已污染，也没有完成所有臂“无污染”的证明。
只核已有 raw、manifest 和生产者记录的 task/sample、预测 hash/行数、模型配置、生成代码与有效 YaRN 绑定；逐臂、逐代注明“已绑定／同时代佐证／声明待核”。
验收条件是证据能绑定到对应旧/新预测本身，或明确维持未闭合的限定性引用；不能以当前代码或新臂回执补写旧臂身份。若仍缺材料，保留缺失，不要求为了改措辞新跑模型。
优先级：先修正证据等级与下游比较措辞；成本仅现存文件核查。无新 GPU/付费、无新实验矩阵、无数据删除；不重复已登记的 LongBench 生成协议不对称。
