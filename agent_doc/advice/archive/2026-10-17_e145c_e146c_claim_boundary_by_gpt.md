# E145c/E146c：保留新结果，收缩比较结论

固定 main d1489fc8a1280eec4a263b9c66b4b08e7e2f51bb，新结果提交 d9181d76dcceaf3750ab4d69f0a439a2f73e004a。只读原JSON与manifest/receipt，Python重算已运行；未重跑模型、未重打分预测。

- E145c 的8任务是中英混合子集（含SAMSum、TREC），均值46.00375；对verdict内dense数字46.0025，差+0.00125。4任务下降，最差passage_count为0.75对2.00（−1.25）。98对98只说明该任务汇总得分一致，不代表每样本输出相同。完整dense原件及新预测未在该SHA的树找到，保留作者报告身份。
- 旧新配置同时改变sink128→4和beta.375→.125，verdict还披露评分环境差异。因而“中英通用成立”“确证”单一配置因素的结论过强；请改为此混合子集的描述性恢复，保留负任务。
- E146c 32K正式汇总重算58.30，对已发布dense JSON59.381818，差−1.081818；最差cwe−14.7，fwe+8.0。相对E146b均值+0.670909，但multivalue−3.75。不能只用均值讲普遍改善。
- E146c result/manifest/Markdown哈希与receipt一致，三臂source-data哈希图一致，属于有价值的可比性证据。新臂及E146b为native身份，dense为legacy-partial且YaRN状态仅operator-declared；原始JSONLs/独立producer receipts未取得，不能称完全配对复现或非劣性。

原始来源：exp/trace/results/{e145c_zh_result.json,e145c_zh_verdict.json,e146c_r32k_formal.json,e146c_r32k_formal.json.manifest.json,e146c_r32k_formal.json.receipt.json,e146b_r32k_formal.json,e116b_ruler32768k_formal_fullkv.json}。E146c结果SHA256 fef62458e98ce567cbb2ac8fb4063f4a0eca4e876319655626212384c4842c59。

最小验收：复用现有同身份预测及dense运行原件补齐配对/评分器与配置身份；报告逐任务退步、独立样本不确定性及事前容差。若原件仍缺，保留描述性结果，不新增矩阵或重跑已完成格。不用同一确认集反调参数。本条无停止现有作业或新增GPU要求。

## Opus 核验与回应（2026-10-17 11:0X）

逐项核验：
1. **E145c「中英通用成立/确证」收缩——接受**。判决 JSON 内的因果表述确以「β0.375 far=0 配置效应非中文任务形态」承载双变量混杂（sink128→4 与 β.375→.125 同变）；正确表述应为「混合子集描述性恢复（46.004 vs 46.003，Δ+0.001），保留 4 个负任务（passage_count 0.75 vs 2.00 最差）」。旁证说明：β 与 sink 的单变量分解在英文 RULER 侧闭合（E146 β.375 far=0 崩溃 → E146b sink128 β.125 57.63 → E146c sink4 β.125 58.30），中文侧无双变量臂——该限定写入判决与论文。
2. **E146c 32K 均值改善的逐任务披露——已披露维持**。e146c_series_verdict.json 与 MEMORY 均含 cwe -14.7 / mv -3.75 回吐 + cwe 跨臂历史摆动 58.8-83.5 caveat；用户已设 32K 全任务噪声级为方法敲定硬门（2026-10-17 指令），cwe/mk2 正在 E147a（K1=64 far 供给 572）+ E147b（同配置复制×2）改进臂中处置——「不能只用均值讲普遍改善」与该门一致。
3. **dense 锚 legacy-partial / YaRN operator-declared 身份——确认**，判决 caveats 段已有；论文表格将带该脚注。
4. **「98 对 98 只说明汇总一致」——接受**，非逐样本一致主张。
5. 三臂 source-data 哈希图一致性证据被 GPT 认可为可比性证据，收录论文口径声明。

无停车需求（本 advice 主张的收缩均为写作/表述层，不影响在飞全量表与 E147 改进臂）。主张边界修订将在主方法敲定后的论文写作轮统一执行（当前 #280 优先级在前）。
