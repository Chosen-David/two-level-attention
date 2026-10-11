# two-level-attention 项目任务索引（AI 维护）

这是 two-level-attention 主项目的唯一 AI 任务索引（agent_doc 口径），
由主 AI 串行维护。
**优先级从属关系：sglang 主仓根目录 `/home/wangyuanshuo02/sglang/TASK.md`
（用户私人设定，AI 只读）是指南层，本文件是执行索引层**——每项登记的
任务须能追溯到用户指令或根 TASK.md 条目；两者冲突时以根 TASK.md 为准。

每项任务的方法、进度、证据、阻塞与下一步写 `task_details/<任务ID>.md`，
不在此堆日志。日期是任务来源/组织日期，不自动表示完成日期。

本索引自 sglang 主仓 `agent_doc/task/TASK.md` 迁移（2026-10-11 主项目
移交）时只保留**未完成**任务（[ ] 在飞 / [~] 暂停），已收官任务不在此
重建；其历史详情保留在 sglang 主仓 `agent_doc/task/task_details/`。

两个身份，DeepSeek和opus，分别负责不同的内容，一个是开源其他论文的对比也即我们的对手，opus负责我们的论文的推进和优化

## 2026-10-08

- [ ] [S-T001] E109 method×(α,β,γ) 海选-全量两级扫描（进行中：AVG5 6 臂
  入账，榜首 aavg(.125,.125,.375)=46.43；三机 12 卡满载推进；旧口径已被
  S-T022/E125 新口径取代，剩余臂作排序先验）
  ([详情](task_details/S-T001.md))
- [ ] [S-T003] agent 库升级 + 项目 agent_doc 布局建立（2026-10-11 已接入
  agent 库 39c6d42 并建立本目录；sglang 主仓存档化处理）
  ([详情](task_details/S-T003.md))

历史任务（E1-E113 全链）的原始记录位于 `exp/trace/` 与 sglang 主仓
`research/docs/`，不在本索引重建；新任务自此按本格式登记。

## 2026-10-09（补记，主 AI 维护义务恢复）

- [ ] [S-T007] E119 四机 RULER 64K/128K 汇总 + 正式入口收口
  （**64K 三臂已收口**：mavg 49.42 > FullKV 48.54 > aavg 47.51，
  bb4cb21ac；128K 在飞四机 20 卡满载，全齐后同口径收口）
  ([详情](task_details/S-T007.md))
- [ ] [S-T009] 论文对比口径讨论（dense vs 单级稀疏）+ E120 单级无粗筛
  方案探索（用户 13:2X 指令；讨论已发起 advice，待 GPT 回复）
  ([详情](task_details/S-T009.md))

## 2026-10-11（上午）

- [ ] [S-T019] E116b 新口径三件套全量重跑（#167；mavg+aavg 双臂 ×
  LB v1 16 任务 + LB v2 503 + RULER 32/64/128K；fullkv 复用仅补三缺任务；
  **四机 20 卡 09:47 全量起跑**；动态主线转向后不停——静态锚点 + fullkv
  基线 + 动态 vs 静态消融素材）
  ([详情](task_details/S-T019.md))
- [~] [S-T021] E124 论文主线转运行时逐序列逐层动态参数（用户 10-11 指令 +
  GPT Runtime 方案 f304ec979 合流；静态 mavg 留作对照/回退，Observation
  改每层差距，D' gate 动态版；A0-A7 阶段制）
  **2026-10-11 15:2X 用户指令：动态线暂停**——E124 全部保留为未验收候选，
  恢复等用户明确通知；GPT 090/091（seq 身份可移植 + DecisionLog 预检）
  为恢复前置门。A1 已收官（087/088/089 修复 + v3 sample）。
  ([详情](task_details/S-T021.md))
- [ ] [S-T022] E125 静态 method 海选收官 → 冠军选举（用户 15:2X 指令 +
  GPT 2bbb9a763 优先级切换；E109 海选 31 臂为旧口径，mavg/fullkv 复用
  E123 新口径数据，4 method 最佳臂 .187 在飞补齐）
  ([详情](task_details/S-T022.md))
- [ ] [S-T023] 竞品复现与整请求公平比较：Quest/FASA/SparQ/ClusterKV + MInference，
  HiP/MoBA条件候选；主表统一整请求质量及实测E2E，各方法保留原生另一阶段。
  阶段子表仅归因；我方海选/冠军/MLA扩展由Opus独占，DeepSeek仅竞品。
  sanity核语义/配置/打分/计时，不能强制结果符合论文方向；Quest新版已补dense分支，decode缺Q/K norm及验收问题待修。
  本轮文档纠偏不新增跑批矩阵、不启动GPU。([详情](task_details/S-T023.md))

## 待办池（未排期，均有用户指令或审计来源）

- [ ] E118：sglang C1/C2/C3 三源码 bug 修复（SG e2e taskmd+CUDA graph
  前置门；C3 修复改行为须换 postfix 重跑；sglang 主仓集成相关，后续回归
  sglang 时处理）
- [ ] E120：粗筛阶段价值消融 + 单级方案对比（依赖 S-T009 讨论收敛）
