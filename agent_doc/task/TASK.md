# two-level-attention 项目任务索引（AI 维护）

- [ ] [TL-EXPLICIT-NEAR-WIDTH-20261019] 固定配额nl<bs仍被旧整页保底放大，与D′标定不一致；210配置host复现，三行候选270错误→0；补丁仅留档、生产源码未改，实际选集/GPU/性能待验。([详情](task_details/TL-EXPLICIT-NEAR-WIDTH-20261019.md))

- [x] [TL-PARTITION-RATIO-20261019] alpha/beta非法比例构造期拒绝；修前普通/-O各28子用例失败，修后各16项host回归通过，独立641构造差分通过；GPU/模型/性能未验，发布回读单独确认。([详情](task_details/TL-PARTITION-RATIO-20261019.md))

- [x] [TL-EXPLICIT-FREE-PROTECTED-20261019] 显式nt=off保护区预算不足构造期拒绝；修前18子用例失败，修后普通/-O各13项host测试通过，独立1080构造差分符合预期。GPU/模型/性能未验。([详情](task_details/TL-EXPLICIT-FREE-PROTECTED-20261019.md))

- [x] [TL-EXPLICIT-L1-BUDGET-20261019] 显式near页配额超K1构造期拒绝，修前24子用例失败，修后普通/-O各4项及相邻6项host测试通过；GPU/模型/性能未验。([详情](task_details/TL-EXPLICIT-L1-BUDGET-20261019.md))

- [ ] [TL-BATCH-EXPLICIT-QUOTA-20261019] E155显式near配额误入batch旧比例公式：入口回退逐组，含最新E155d归一配置，修前25子用例失败，修后普通/-O各6项host测试通过；GPU/模型/性能及历史重测待验。([详情](task_details/TL-BATCH-EXPLICIT-QUOTA-20261019.md))

- [ ] [TL-PREFILL-SCORE-BUDGET-20261019] candidate eager GQA行块预算漏算查询头；源码绑定反例推导单score为928MiB而预算512MiB。仅审计/复现，非GPU峰值或OOM；生产修复待实际张量与GPU验证。([详情](task_details/TL-PREFILL-SCORE-BUDGET-20261019.md))

- [x] [TL-SUMMARY-VALIDATION-20261019] CPU汇总复用单次协议校验/样本索引，逐行门禁不变；普通/-O各34项通过，512样本配对中位4392.28→47.41ms。非GPU/请求收益。([详情](task_details/TL-SUMMARY-VALIDATION-20261019.md))

- [ ] [TL-PREFILL-SIGMA-CAP-20261019] top-σ 可变基数使用真实 mask 容量，修复 candidate 无组截断；普通/-O各108项CPU回归及独立各10case通过，GPU/性能及历史重测未验。([详情](task_details/TL-PREFILL-SIGMA-CAP-20261019.md))

- [ ] [TL-PREFILL-MASK-CONTRACT-20261019] reference/candidate拒绝错误token mask元数据，防止单槽广播、维度截取及块索引误读；ops普通/-O各84项通过，GPU/模型/性能未验。([详情](task_details/TL-PREFILL-MASK-CONTRACT-20261019.md))

- [ ] [TL-DECODE-MASK-EXTENT-20261019] 拒绝覆盖不足的decode选择mask，避免单槽广播扩大选集；普通/-O各25项CPU测试、独立48合法+12非法对照通过；有界修复验收，GPU/性能及真实selector重测未验。([详情](task_details/TL-DECODE-MASK-EXTENT-20261019.md))

- [ ] [TL-BATCH-TOPK-BLOCK-20261019] 非64块保留全宽L2 topk，避免64特化补位错误；CPU回归通过，GPU及历史重测未验。([详情](task_details/TL-BATCH-TOPK-BLOCK-20261019.md))

- [ ] [TL-DECODE-CU-STRIDE-20261019] fused decode 长度数组按需连续化；70项CPU普通/-O及独立30布局通过，GPU/性能未验。([详情](task_details/TL-DECODE-CU-STRIDE-20261019.md))

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

## 现行指令变更登记

- **2026-10-19（用户指令）**：**解除 SGLang 修改冻结**——10-18 advice 中"不要修改 SGLang"的范围限定作废，sglang 主仓可以被修改（E118 复活、生产栈集成等工作不再受此限制）。是否启动仍按优先级与触发条件排期，本条是解锁而非派单。

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
  sanity核语义/配置/打分/计时，不能强制结果符合论文方向；Quest新版已补dense分支及decode Q/K norm，faithful port与整请求验收仍待完成。
  本轮文档纠偏不新增跑批矩阵、不启动GPU。([详情](task_details/S-T023.md))

## 2026-10-11（晚间）

- [x] [S-T025] E126 五臂 TLI 选择器分阶段计时 microbench（成本指纹，
  **优化前基线**；收官。零改动 AST 插桩（生产文件 md5 冻结锚点全程
  fail-closed）+ 红绿 python±-O 全绿 + H20 正式计时五臂全落袋。
  关键发现：S3_fine 细筛分为最大选择器成本项（O(S) 线性 0.23→2.67ms）；
  cavg 系步均建簇摊销额外成本仅 0.03-0.13ms 但冷建簇超线性（64K 需
  1.7-1.9s，e2e 不可隐藏）；选择器成本对数据分布不敏感。
  worktree e126-stage-timing（eb308c7+6325bcf）按隔离纪律未 push，
  E127 验收合入后同协议复测，前后成对进 E128）
  ([详情](task_details/S-T025.md))

## 待办池（未排期，均有用户指令或审计来源）

- [x] [TL-BATCH-SINK-ZERO-20261019] 零sink批量路径哨兵冲突：入口安全回退，修前3失败；普通/-O每模式92项unittest+8场景通过。GPU/实模/性能与历史格重测未完成。([详情](task_details/TL-BATCH-SINK-ZERO-20261019.md))

- [x] [TL-PREFILL-EMPTY-20261019] reference空行兜底按行/head隔离，避免污染非空选集；普通/-O各67项、相邻21项及独立1324场景通过，发布回读主控闭环，GPU/模型/性能未验。([详情](task_details/TL-PREFILL-EMPTY-20261019.md))

- [x] [TL-E151C-LAYER-20261019] 配额消费点拒绝缺失/非法/未覆盖层号；21项E151c测试及24场景旧实现选集差分普通/-O通过，GPU/模型/性能未验。([详情](task_details/TL-E151C-LAYER-20261019.md))

- [x] [TL-E151C-MODES-20261018] oracle配额与gamma-off/非E64/sinkswa/MoBA/sigma不兼容组合构造期拒绝；普通/-O各41项或场景通过及独立复核，GPU/质量/性能无新增结论。([详情](task_details/TL-E151C-MODES-20261018.md))

- [x] [TL-E151C-INPUT-20261018] quota JSON 构造期拒绝字符/对象键误读、空表及非法数值；保留合法取整，13项普通/-O专项通过。短表/配置支持仍待办。([详情](task_details/TL-E151C-INPUT-20261018.md))

- [ ] [TL-E151C-ROUTE-20261018] prefill 配额路由修复：reference 激活/异常清理，batch 回退逐组；35项/场景普通及-O CPU验证，GPU/实模/性能与历史配额臂重测待办。([详情](task_details/TL-E151C-ROUTE-20261018.md))

- [ ] [TL-DAC-BEST-PAPERS-20261016] DAC 2017–2026 Best Paper PDF 收集：13 篇获奖名单已核实，7 篇有效 PDF 已入 advice/DAC_best_paper，6 篇因全文入口受阻/失效待补；2026-10-16 第三轮 arXiv/作者关键词补查新增 1 篇明确标注的后续期刊 PDF（不计入 7/13），第五轮 arXiv 专项 24 组查询及 v1 版本核查仍缺 6 篇；用户明确不发邮件，已取消发送路线；附来源、版本、页数与 SHA-256。
  ([详情](task_details/TL-DAC-BEST-PAPERS-20261016.md))

- [x] [TL-E152-STORE-20261016] E152 同路径合作写者锁与原子JSON发布；修前丢记录/中断损坏复现，16项普通/-O通过。LongBench因缺Torch未执行，非GPU提速。([详情](task_details/TL-E152-STORE-20261016.md))

- [x] [TL-E152-TIMING-20261016] E152 连续请求 wall/TTFT、单 token TPOT及 -O 协议门禁修复；专项8项和捕获11项普通/-O通过。GPU路由/低扰动E2仍待验，非提速结果。([详情](task_details/TL-E152-TIMING-20261016.md))

- [x] [TL-SPARQ-LIFECYCLE-20261016] B轨SparQ跨请求V_mean owner/length修复；专项12项、ops62项普通/-O及36场景144次decode逐位对照通过。GPU/真实模型及历史影响重测待办。([详情](task_details/TL-SPARQ-LIFECYCLE-20261016.md))

- [x] [TL-MLA-MASK-20261015] MLA decode非恒等mask在cache更新前拒绝；50项普通/-O CPU回归与24合法路径逐位对照。显式CUDA mask新增归约/同步，性能及真实模型尚未验证。([详情](task_details/TL-MLA-MASK-20261015.md))

- [x] [TL-DECODE-INPUT-20261015] decode公共消费端形状与Q/K宽度守卫，CPU回归41项与48合法路径逐位对照；GPU未验，不代表padding支持。([详情](task_details/TL-DECODE-INPUT-20261015.md))

- [x] [TL-Q-STRIDE-20261015] fused wrapper按需整理Q特征布局，44项CPU回归及独立ABI矩阵通过；GPU/性能尚未验证。([详情](task_details/TL-Q-STRIDE-20261015.md))

- [ ] E118：sglang C1/C2/C3 三源码 bug 修复（SG e2e taskmd+CUDA graph
  前置门；C3 修复改行为须换 postfix 重跑；sglang 主仓集成相关，后续回归
  sglang 时处理）
  - **2026-10-19 用户指令升级（解冻+扩容）**：sglang 可修改（见顶部登记）；
    集成目标从"修 bug"扩为**四步方案**：① E118 复活修 C1/C2/C3，TLI bs=1
    在 sglang 跑通 → ② 等价门禁：sglang bs=1 vs HF harness 同任务对拍
    （噪声带内）→ ③ 批量门禁：bs=1 vs bs=32 对拍，过了 bulk eval 切高 bs
    （**核心收益 = 质量 eval 产能提升一个数量级**，GPU 队列瓶颈缓解）→
    ④ 速度主张：sglang 内全员 graph 默认口径测 TPOT，Quest 臂标注栈差异
    （其原生在 vLLM）。前③步完成即回本，④才进论文。
- [ ] E120：粗筛阶段价值消融 + 单级方案对比（依赖 S-T009 讨论收敛）
- [ ] E127：TLI 选择器优化（候选方向：S3_fine 细筛分 O(S) 主成本项 +
  cavg 系 prefill 建簇/decode 复用分相；验收合入后按 S-T025 协议复测）

## 2026-10-15（GPT 论文工程分支）

- [x] [TL-E140-ASSET-PROVENANCE-20261015] E140 历史图可见诊断警告及旧/新身份分离；CPU 正负回归，不代表科学有效性恢复。
  ([详情](task_details/TL-E140-ASSET-PROVENANCE-20261015.md))

- [ ] [GPT-PAPER-20261015] 论文运行时、数据门禁与图表薄入口；CPU 回归及独立审查，GPU 实测和科学验收未完成。仅发布独立开发分支。
  ([详情](task_details/GPT-PAPER-20261015.md))

- [x] [TL-E140-O4-PARTIAL-PAGE-20261015] O4 有效末页修复及面板(d)历史诊断隔离；CPU验收，未重采。
  ([详情](task_details/TL-E140-O4-PARTIAL-PAGE-20261015.md))

## 2026-10-16（论文论证修订）

- [x] [TL-PAPER-SCHOLARLY-20261016] 实际论文技能与两篇DAC获奖全文学习；双语逐段论证/PPT修订、独立科学与全页读者复核，保留数据及负结果，零新实验。([详情](task_details/TL-PAPER-SCHOLARLY-20261016.md))

- [x] [TL-OBS-PLOT-CORRECTNESS-20261016] 有界修复整箱预算、真实区域、有限配对分母和E6证据标注；CPU独审后仅推原dev。合并裁决（Opus 2026-10-17）：O 图系列保留 main 出版版渲染器（e021281 四带呈现 + e5f5b7e 统计修正已吸收 whole-bin 预算/区域归属修正），GPT 诊断版渲染器与 validity v4 降级主张未采纳——(a)(b)(c)(d) 限定手稿 caption 已如实披露（mass-rank 参考、whole-bin 预算、gate 默认关闭）；诊断资产 O_observation_diagnostic.json 与 record_provenance.py 保留入库。
  ([详情](task_details/TL-OBS-PLOT-CORRECTNESS-20261016.md))

- [ ] [TL-STATIC-MASK-20261016] 恢复授权：StaticCache mask 精确旁路与 DynamicCache 哨兵修复，105 项指定 CPU 测试普通/-O通过；GPU/实模/性能仍待验。([详情](task_details/TL-STATIC-MASK-20261016.md))

- [x] [TL-DECODE-PADDING-Q-20261019] Qwen3/Llama 单 query decode 仅压紧 K/V，修复合法 2D padding mask 对 Q 的越界形状索引；33 项普通/-O CPU 回归及独立多步/长序列 gather 复核。GPU/真实 selector/性能另待验证，不宣称提速。([详情](task_details/TL-DECODE-PADDING-Q-20261019.md))
