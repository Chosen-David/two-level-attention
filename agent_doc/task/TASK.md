# two-level-attention 项目任务索引（AI 维护）

## 2026-10-21

- [ ] [TL-TLI-INC-SIGNATURE-20261021] 增量索引缓存绑定device/dtype/KV头几何/页大小，不兼容时全量重建；CPU张量精确对照及普通/-O各42回归通过，独立审查通过，CUDA数值和性能待验；仅修复分支/draft PR，不合main。([详情](task_details/TL-TLI-INC-SIGNATURE-20261021.md))

- [x] [TL-TLI-RUNTIME-SHAPE-20261021] TLI 公共入口在增量/快捷分流前显式拒绝多请求、多 query、错误 K 宽度及缺失 MoBA 分数，避免增量路径静默只消费 `k[0]`，且校验不再被 `python -O` 删除；修前普通4失败、-O为3失败1错误，修后普通/-O各13项host回归通过。当前无torch/GPU，生产张量、质量、显存与性能未验。([详情](task_details/TL-TLI-RUNTIME-SHAPE-20261021.md))

- [x] [TL-LB-EMPTY-HANDOFF-20261021] LongBench 手动生成在短 prompt 全部进入 question 时不再向模型提交 0-token material；空 question 的 BOS 归一化也不再越界。修前基线确定触发空输入调用，修后普通/-O各28项相邻回归通过；当前无torch/GPU/真实tokenizer，实模输出和性能未验。([详情](task_details/TL-LB-EMPTY-HANDOFF-20261021.md))

- [x] [TL-LB-SHORT-PROMPT-SPLIT-20261021] LongBench 手动生成入口将无锚点短 prompt 的 `-1` 切片修为零边界，恢复完整问题阶段并保持长输入 100 字符上限/锚点优先级；修前最小复现与新增回归失败，修后普通/-O各22项通过。当前无torch/GPU/真实模型，端到端token、质量与性能未验。([详情](task_details/TL-LB-SHORT-PROMPT-SPLIT-20261021.md))

- [x] [TL-LB-DATASET-BINDING-20261021] 评分文件前缀绑定sidecar数据集，拒绝错集评分/输入身份分组，保留repobench-p前缀兼容；修前3失败，修后普通/-O各26项通过。完整评分依赖、历史重评及GPU未验。([详情](task_details/TL-LB-DATASET-BINDING-20261021.md))

- [x] [TL-LBV1-INPUT-UNIFORMITY-20261021] LBv1 聚合按数据集拒绝跨 method 的源行、模板、实际 token 身份漂移及 v2/legacy strict 混挂；修前入口无门禁，修后普通/-O各7项及相邻各15项通过。当前无jieba/torch/GPU，完整eval、历史重评、质量与性能未验。([详情](task_details/TL-LBV1-INPUT-UNIFORMITY-20261021.md))

- [x] [TL-BATCH-TREATMENT-SNAPSHOT-20261021] RULER batch TLI 的 runtime、输出名与完成回执改为共享模型加载前冻结的唯一 treatment 快照；修前入口级复现证明快照晚于模型且回执无 manifest，修后普通/-O各1项通过。当前无torch/GPU，真实模型、历史产物与性能未验。([详情](task_details/TL-BATCH-TREATMENT-SNAPSHOT-20261021.md))

- [x] [TL-BATCH-ADDITIVE-MASK-20261021] FASA batch decode 的加性 mask 有效长度由统计非零修为统计零值；修前真实契约例把 `[3,4]` 反算为 padding 长 `[2,1]` 并拒绝，修后普通/-O各4项通过并保留bool语义；当前无torch/GPU，实模质量与性能未验。([详情](task_details/TL-BATCH-ADDITIVE-MASK-20261021.md))

- [x] [TL-BATCH-CONTEXT-ISOLATION-20261021] 批量decode请求上下文由进程全局改为ContextVar并用scope恢复owner；修前双线程确定串扰，修后线程/async/嵌套异常普通及-O各3项通过，独立复核无blocker；当前环境缺torch，GPU/实模/性能及显式线程池传播未验。([详情](task_details/TL-BATCH-CONTEXT-ISOLATION-20261021.md))

- [x] [TL-E155D-STAGING-20261021] T2每次打分独占快照，避免重复/并发调用删除前次证据；新增5反例、普通/-O各72项及各24入口对照通过；独立双进程/删源后20预测20manifest及CI通过，发布回读另核，真实T2/GPU未验。([详情](task_details/TL-E155D-STAGING-20261021.md))

- [x] [TL-E155D-ATOMIC-VERDICT-20261021] T2写入中断保留旧判决、避免公开半JSON；修前10项9失败，普通/-O各67项CPU回归通过，独立24入口完整JSON及87几何对照通过；真实T2/GPU/并发staging未验，发布回读另核。([详情](task_details/TL-E155D-ATOMIC-VERDICT-20261021.md))

## 2026-10-20

- [x] [TL-BACKFILL-ID-FORMAT-20261020] 回填旧格式回退前检查全行ID格式，修复首行掩盖后续混行；普通/-O各8方法及旧105各10项通过，独立200行默认计数CLI/sidecar保留通过；非正文/token身份或GPU/性能验收。([详情](task_details/TL-BACKFILL-ID-FORMAT-20261020.md))

- [x] [TL-E155D-GEOMETRY-20261020] T2目录几何绑定冻结manifest/文件名hash；修前8项7失败，普通/-O各57项、87合法producer/51错标签及24组差分与独审通过；仅汇总门禁完成，真实T2/GPU/性能未验。([详情](task_details/TL-E155D-GEOMETRY-20261020.md))

- [ ] [TL-LBV1-INPUT-IDENTITY-20261020] 修前4组实际AST反例；已实现v2完整源行+模板+实际token身份闭环，普通/-O各15/15。真实Qwen3/GPU端到端与历史影响待验。([详情](task_details/TL-LBV1-INPUT-IDENTITY-20261020.md))

- [x] [TL-E155D-SNAPSHOT-20261020] 固定配额排名/CI绑定暂存预测字节，修复源替换造成方向反转；普通/-O各41项及各24组差分通过，独审通过；真实T2/GPU/性能未验，发布回读另核。([详情](task_details/TL-E155D-SNAPSHOT-20261020.md))

- [x] [TL-E155D-SCORE-UNITS-20261020] 固定配额 CI 与 AVG5 统一为分值单位并显式标注；修前5失败，普通/-O各32项、相邻17项及24组基线差分通过；真实T2重评/GPU/性能未验。([详情](task_details/TL-E155D-SCORE-UNITS-20261020.md))

- [x] [TL-E155D-PAIRING-20261020] 固定配额top-N CI拒绝样本ID错配、缺失和重复；修前5失败2错误，普通/-O各25项及17项相邻host回归通过；独立24合法差分和20主入口失败场景通过，真实评分/GPU/性能未验。([详情](task_details/TL-E155D-PAIRING-20261020.md))

- [x] [TL-E155D-ADMISSION-20261020] 固定配额网格汇总拒绝歧义文件和冲突完整副本；修前3失败、修后普通/-O各13项及40项相邻host回归通过；独立87格合法对照与冲突过滤通过，真实快照/GPU/性能未验。([详情](task_details/TL-E155D-ADMISSION-20261020.md))

- [x] [TL-VENDOR-FROZEN-MANIFEST-20261020] 修复E155e重读已冻结源造成vendor结果身份漂移；4反例修复、7项普通/-O host测试及独立32配置身份对照通过；固定配额/kernel未改，GPU未验，发布回读另核。([详情](task_details/TL-VENDOR-FROZEN-MANIFEST-20261020.md))

- [x] [TL-EXPLICIT-ORACLE-20261020] 固定配额拒绝E151c prefill oracle混配；9个错误接受配置修复，普通/-O各25项host测试及212合法状态差分通过；GPU/质量/性能未验。([详情](task_details/TL-EXPLICIT-ORACLE-20261020.md))

- [x] [TL-EXPLICIT-MODES-20261020] 固定配额拒绝sinkswa_only/MoBA/sigma旁路组合；15反例拒绝，普通/-O各21项host测试及181合法构造状态对照通过，GPU/性能未验。([详情](task_details/TL-EXPLICIT-MODES-20261020.md))

- [x] [TL-DEBUG-OFF-20261020] 六处调试关闭契约统一：54次误入→0，普通/-O各24项host测试通过；固定配额与算法未改，实际GPU/性能待验。([详情](task_details/TL-DEBUG-OFF-20261020.md))

- [x] [TL-ENUM-CONTRACT-20261020] 四个选择枚举不再被-O取消校验；32非法构造拒绝，336合法对照不变，普通/-O各21项host测试通过；Torch依赖/GPU/性能未验。([详情](task_details/TL-ENUM-CONTRACT-20261020.md))

## 2026-10-19

- [x] [TL-EXPLICIT-INTEGER-20261019] 固定配额nb/nt/nl禁止小数静默截断；修前21失败3错误，修后普通/-O各19项host回归通过，6反例拒绝、108对照不变；完整依赖/GPU/性能未验。([详情](task_details/TL-EXPLICIT-INTEGER-20261019.md))

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

## 2026-10-21

E155d 主任务链并轨节（T1–T8/TDS，源自根 TASK.md 字节级归档迁移，见
`legacy/TASK.7679d9be4f96….md`）。

- [x] [TL-E155D-T1-20261021] E155d 网格监督 87/87 收官（缺口格补跑过门，六机进程清零核实） ([详情](task_details/TL-E155D-T1-20261021.md))
- [x] [TL-E155D-T2-20261021] E155d 网格打分终版判决：冠军 nb32nt448nl4096 AVG5 48.668（+1.21 vs CTRL 单池；top-3 配对 CI 全含 0，同族几何稳健） ([详情](task_details/TL-E155D-T2-20261021.md))
- [ ] [TL-E155D-T3-20261021] nl=全mid 退化端点补跑（top-2 几何叙事用，各 1 格，未跑） ([详情](task_details/TL-E155D-T3-20261021.md))
- [ ] [TL-E155D-TG-20261021] eval harness CUDA Graph 化 + 大 bs 化（用户提级为 T6 前置硬门；阶段 0/1 已入库，实现由 TG 会话承担，本链只做协调审计） ([详情](task_details/TL-E155D-TG-20261021.md))
- [ ] [TL-E155D-T4-20261021] nt 轴可分性 RULER 64K 七臂实验（预登记判定规则；TG 升级完成后跑；LongBench 先验 nt448 vs ntoff +0.405 排除 0） ([详情](task_details/TL-E155D-T4-20261021.md))
- [ ] [TL-E155D-T5-20261021] ablation 论文部分写入（TLI_paper_en.tex / TLI_paper.tex 双语镜像） ([详情](task_details/TL-E155D-T5-20261021.md))
- [x] [TL-E155D-TDS-20261021] DeepSeek T7 fix 三重独立复跑审查通过（E127 15/15 + batch 门禁 26/26 + E133c 开关对照轮廓逐步全等） ([详情](task_details/TL-E155D-TDS-20261021.md))
- [ ] [TL-E155D-T6-20261021] 主方法敲定 + 全量测试（用户改序暂缓：等 TG 提速确证；六机派单已预置未启动，零进程零清理） ([详情](task_details/TL-E155D-T6-20261021.md))
- [ ] [TL-E155D-T7-20261021] 性能浪费点修复（E133c A1-A4/A6 已入库；剩余 .187 独占窗口稳态 bench 同机对照 + 判决 JSON） ([详情](task_details/TL-E155D-T7-20261021.md))
- [ ] [TL-E155D-T8-20261021] 推送远端 main（全部判决/论文/代码提交后 origin push，含历史挂账） ([详情](task_details/TL-E155D-T8-20261021.md))
