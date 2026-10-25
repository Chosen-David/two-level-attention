# T2固定配额目录必须绑定冻结几何

问题ID：TL-E155D-GEOMETRY-20261020；基线main
`8d541d4001b184c2c14062e5e7303dce9b7b12a2`。

原`exp/trace/analyze_e155d_grid.py:175–220`从目录取nb/nt/nl并计入排名，
eval只核预测/runcap，不比对已复制的`tli_manifest.json`。将声明nb4的合法
预测放入nb8目录，即使文件名仍匹配nb4 manifest，也会被当作nb8计分；
缺失manifest同样通过。修前8项中7失败、合法87格控制通过。

新增`cell_treatment_hashes`（102–133行）复用原生产者规范JSON/hash10，
要求TLI文件名、manifest对象、整数几何及64块大小与目录一致；CTRL须三个
显式几何为null且alpha=beta=0。评分前后校验并绑定sidecar完整字节SHA256；
无效格保留`evaluation_errors`，不进入排名/CI。原始预测不改，配额算法、
chunk2048、子空间、D′ gate、sink/SWA、kernel和bootstrap算法均未改。

[验证记录](../results/e155d-geometry94/verification.md)包含普通/-O各57项、
实际metadata生产者87格/51错标签及稳定输入各24组差分。这里验证的是声明
provenance一致性，不是manifest真实反映GPU执行的证明，也不是LongBench
实际评分验收。新增两次小sidecar读取/解析/哈希属于正确性开销，未实测提速。

旧T2结果用于T3–T6前应以原预测和原sidecar重汇总；不应为过门而手填或重算
缺失manifest。发现错标签先追溯来源；预测和身份仍完整时不因本项盲重跑GPU。
本次未取得真实收集快照，不能声称历史冠军/论文已受影响。并发staging/ABA、
跨主机同预测异sidecar、非几何超参/运行环境一致性及跨格样本完整身份仍需
分别核验；本门禁不是完整治疗配置或执行真实性审计。near宽度候选、T7和
prefill性能改造仍缺当前可用GPU验证。
