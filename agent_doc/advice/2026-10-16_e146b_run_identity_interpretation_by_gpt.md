# E146b 32K：数值差异成立，跨机噪声／top-k tie 归因尚未成立

只读复核基线：`539342c686f8f97dda6849faa91f1ae900bf89c2`；待核解释来自提交 `d39d7d71a79730455c6b084c7217e45699c47f14`。未运行模型、未更改配置或调度任务。已查阅该基线下 advice 当前层全部61个文件及索引，未见同项 E146b32K 归因审查；既有 E139 topk 并列序注意事项不能替代本次因果证据。

## 可以保留的结论

- 对11任务分数独立重算：E146b 为57.6290909，E116b mavg 为59.6518182，差−2.0227273分；CWE差−12.5、多键2差−12属实。FullKV历史汇总均值为59.3818182。
- 两份manifest的11个源数据SHA256、任务IDs、答案hash和lengths一致；新结果的11/11生成回执同代绑定及配置检查由formal manifest记录。这增强输入与产物关联证据，但不等于独立重算全部原始预测。
- 目前应称“名义配置相近的两次运行存在未解释的2.02分差异”。单次差值不能估计“跨机复制噪声约2分”，汇总分数相等也不能证明FullKV预测逐字确定性复现。

## 尚未闭合的身份与归因

两次回执的producer脚本SHA256及effective_config_sha256不同（完整值见下）；方法标识亦由`h562141be42`变为`h3b26085047`。这证明记录的字节身份不同，**不直接证明有效配置语义或执行行为不同**，也不能直接认定它们导致分数差。当前可查材料未给出导致该分数差的并列候选轨迹、跨机首次分歧或受控tie-break干预，因此“top-k tie跨机分歧”为未检验解释。

最小下一步先使用已有产物：对两份实际effective-config JSON做逐字段语义差分，区分路径/记录字段与会改变选择的字段；按回执SHA定位两份实际producer脚本并diff，核对依赖的indexer实现、模型/分词器及运行环境身份；随后比较已有逐样本预测和分数。优先复用保存产物，不先派重跑。仅凭hash不同，不预判语义差异；若要归因top-k tie，仍需直接的候选分数/索引或等价受控证据。现有材料不授权恢复或新增实验。

## 可追溯来源（均相对上述固定基线）

- `exp/trace/results/e146b_r32k_formal.json`：Git blob `e7ee04d3fd97d36b88d866e1921592f1934390b2`；SHA256 `4bbbd9a28b4c8c209557dc2f164952a760f57502a86f4f643af6239a0abd3c12`。
- `exp/trace/results/e146b_r32k_formal.json.manifest.json`：blob `e3d61a3a237cf17ae5113139fe6f6585c79f65db`；SHA256 `40a5d8faa31a6f031bc4a2bf2885ad581d5d9ac7dd310563a7d5c4180f714936`。
- `exp/trace/results/e116b_ruler32768k_formal_mavg.json`：本次读取的归档副本blob `c83031de2a09e12d8d2715c2db7ad1d730dda2c4`；SHA256 `b8ec523b84110389958a6df9c59f0a19bb3490361a9ac4627dd028009edac8ed`。
- `exp/trace/results/e116b_ruler32768k_formal_mavg.json.manifest.json`：blob `f9f5c55441fc3efc76f8ddf4f79f64dec9b62f32`（已与fresh main核同）；SHA256 `66c608950c7e5889ccb8852db100db4b231ade500a9ea6d48e3db7a20726a090`。
- 回执producer SHA256（旧→新）：`9e68bdd1e84621e725c3c2d7151415d2adb0afa4f50d26c7b077e117cb03af37` → `ab5c7d6307a14e4d69c51271f69358271aaac4adb46347a5239766504e26524b`。
- 回执effective-config SHA256（旧→新）：`0c355ec6c898d4f6adee434956a9d9b43c243268e9b450fb9377eea3976df899` → `6d2f2d632d2075a04e12b74a1ab323a49f6db9f382bc796b7614c21905da1eb3`。

