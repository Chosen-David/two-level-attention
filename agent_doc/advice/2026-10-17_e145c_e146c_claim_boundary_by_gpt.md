# E145c/E146c：保留新结果，收缩比较结论

固定 main d1489fc8a1280eec4a263b9c66b4b08e7e2f51bb，新结果提交 d9181d76dcceaf3750ab4d69f0a439a2f73e004a。只读原JSON与manifest/receipt，Python重算已运行；未重跑模型、未重打分预测。

- E145c 的8任务是中英混合子集（含SAMSum、TREC），均值46.00375；对verdict内dense数字46.0025，差+0.00125。4任务下降，最差passage_count为0.75对2.00（−1.25）。98对98只说明该任务汇总得分一致，不代表每样本输出相同。完整dense原件及新预测未在该SHA的树找到，保留作者报告身份。
- 旧新配置同时改变sink128→4和beta.375→.125，verdict还披露评分环境差异。因而“中英通用成立”“确证”单一配置因素的结论过强；请改为此混合子集的描述性恢复，保留负任务。
- E146c 32K正式汇总重算58.30，对已发布dense JSON59.381818，差−1.081818；最差cwe−14.7，fwe+8.0。相对E146b均值+0.670909，但multivalue−3.75。不能只用均值讲普遍改善。
- E146c result/manifest/Markdown哈希与receipt一致，三臂source-data哈希图一致，属于有价值的可比性证据。新臂及E146b为native身份，dense为legacy-partial且YaRN状态仅operator-declared；原始JSONLs/独立producer receipts未取得，不能称完全配对复现或非劣性。

原始来源：exp/trace/results/{e145c_zh_result.json,e145c_zh_verdict.json,e146c_r32k_formal.json,e146c_r32k_formal.json.manifest.json,e146c_r32k_formal.json.receipt.json,e146b_r32k_formal.json,e116b_ruler32768k_formal_fullkv.json}。E146c结果SHA256 fef62458e98ce567cbb2ac8fb4063f4a0eca4e876319655626212384c4842c59。

最小验收：复用现有同身份预测及dense运行原件补齐配对/评分器与配置身份；报告逐任务退步、独立样本不确定性及事前容差。若原件仍缺，保留描述性结果，不新增矩阵或重跑已完成格。不用同一确认集反调参数。本条无停止现有作业或新增GPU要求。
