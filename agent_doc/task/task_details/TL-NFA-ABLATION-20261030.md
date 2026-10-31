# [TL-NFA-ABLATION-20261030] NFA消融A–F：HotpotQA＋GovReport

Task-ID: TL-NFA-ABLATION-20261030
Date: 2026-10-30

## Plan

按用户最新决定只用HotpotQA/GovReport，不增加数据集。实现并实跑A低频/高频/随机/全维与维数；B全局MinMax、分区MinMax/MinMax、Avg/Avg、Avg/MinMax及边界页；C四种skip允许模式与阈值、同gate的缓存管理比较；D基础/融合/并行/Graph，B1/4/8；E近区1024/2048/4096/8192；F sink0/4/16/64×swa0/64/128/256全部16组合，总注意力预算含保护区严格1024且动态候选去重。质量/延迟/内存指标分开，不支持的路径先修/标未执行，不伪造结果。先用现有已验证缓存语义，新的Paged实现独立验证后才能作为新增版本比较。

PROJECT_ROOT=/home/wangyuanshuo02/two-level-attention；结果agent_doc/results/nfa_ablation_20261030/。读取人类GUIDE和适用AGENTS，绝不改guide。各用独立worktree、固定git源码。已在飞任务不停止、不原地修改。先审复用证据，实验先冻结协议、后独立verify_experiment_result，未验收不结论。总索引仅root维护，任务owner可更新本详情。

资源：优先113/125两台各8卡；先查真实占用。规划软分工：两机GPU0–5优先消融，6–7优先缓存验证；已有GLM/MK2/他人占用不可抢。性能卡独占。双方登记/沟通后才领取空卡；本机GPU1禁止。不足再由root协调其他卡。

## Progress

已获用户明确授权创建agent执行；待owner核查当前代码并细化可执行矩阵/接口与验证。

- Root已启动owner /root/nfa_ablation，授权独立实现与实测；先回报矩阵/接口后进入独立审查。用户再次确认113/125优先，不改变现有任务协议。

### Owner进展（2026-10-30，首轮实现）

- worktree `/home/wangyuanshuo02/tli-nfa-ablation`，branch `experiment/nfa-ablation-20261030`，base `8718c90`，首版commit `d060723`。
- 已读并核现有调用链。复用 `prefill_frozen_skip.prepare_ids/capture_profiles/generate`、`TLIIndexer.compute_score`、官方sample_score；旧E90是Hotpot/Musique旧ratio，不直接复用结论；E135旧grid只mass且K2=2048，不替代本轮16组合；paired harness可适配但当前hardcodedGPU1/far-only/B1。
- 真实发现：旧boundary bypass不占K1，SWA=0缓存不支持。新增独立strict_ablation_v1 inclusive候选/最终预算，生产默认不变；完整/高频/随机实际坐标、SWA0 FIFO、retain/evict相同可见KV路径已实现。
- CPU 53 tests passed，证据 `agent_doc/results/nfa_ablation_20261030/validation/cpu_pytest.txt`；尚待独立verify_experiment_result，不提前声称正式结果。
- 矩阵47个唯一cell，两任务各200=18800 request；F16完整，C阈值.01/.025/.05/.1且.05默认。协议和矩阵落PROTOCOL.md/matrix.json。
- 明确采用full-prompt prefill冻结参考协议；不是Quest-prefix。质量墙钟不是部署性能；B自主生成轨迹mass不是matched-query对照，仍需固定轨迹replay。D单独尚未执行。
- 资源盘点113八卡GLM占用，125八卡MK2已完成且空闲。cache agent计划GPU6独占验证，我方0–5质量。正在同步独立源码和Hotpot/Gov数据到125，先GPU smoke后独立验收派正式。

## Plan v2（用户最新steering，覆盖未执行的47-cell全量计划）

用户允许探索性消融不追求统一新口径，优先直接复用已有数据；进一步明确真实sink128+swa128旧消融可以作为该组独立设置，论文注明即可。禁止把已知其他参数或未知参数改标4/128。未知保持unknown；不再采用默认假定。不同组参数如实说明，不作严格单因素因果归因。

已暂停18800-request新矩阵正式派单；没有GPU进程需要停止，125仅同步了独立源码/data，CPU成果与commit d060723保留。继续任务：历史原始预测重评分、runcap绑定与答案样本完整性核查，A–F草表，真正缺格再做最小补测。复用不取消独立真实性审查。

当前发现：A/E90原始Hotpot full/high/random；B/E72原始Hotpot三种区域打分；E/E155原始Hotpot/Gov的nb32/ntoff下near2048/4096/8192（manifest证实4/128）；C/full5独立冻结三臂full/far/dual各200真实TLI请求；F/E135是不同保护网格且K2=2048的mass，须按真实参数作为历史机制证据，不冒充目标1024质量网格。D需从已验收paired结果选取有效域。

## Plan v3（最新执行顺序：先E，选定nl，再F）

先固定sink4/swa128检查近长旧数据并做任务内逐请求配对bootstrap，确认/选择nl之后才固定nl开展F保护区实验。C near-only补测不抢E/F优先级。

E旧数据三档2048/4096/8192已实际重评，producer同ecfbd4f、pred.py同hash、全部其他manifest一致（4/128、nb32/ntoff、K1=128/K2=1024、gateoff）、逐_id答案/输入长度对齐。完整token哈希和完整旧环境未记录。HQ相对4096：2048 −0.173pp CI[-1.667,1.294]，8192 +.776[-1.065,2.753]；Gov：2048 −.185[-.558,.182]，8192 +.263[-.130,.653]。这是任务样本重采样不确定性，不是运行随机噪声或等效证明。已交独立verify_pool_initial复核。结果near_length_paired.json。

1024已有历史nb4/8/16格，但未找到nb32/ntoff匹配格，不能混当同quota。若补这档需复用ecfbd4f原始pred.py（历史Quest式最多100字符后缀），不能与新full-prompt或strict选择器混池。F待nl登记；新F若采用1024含保护预算则独立声明，不假称历史旧grid也是同口径。

- E独立验收已完成，`reuse_E_review.md` + `reuse_E_independent_checks.json`：1200逐请求score、6组bootstrapCI全部精确一致，usable-with-scope。补充关键限制：1024在固定nb32下违反生产构造约束32×64<=nl，不是普通缺格，不派非法任务。8192点估计更高但CI含0，4096可按现有默认/工程折中保留，不能宣称统计等效或最优。

- 已根据E独立证据登记F固定near_len=4096（NEAR_LENGTH_DECISION.md）：保留既有默认、合法测试中档、避免依不确定小幅质量差异扩大Near边界；不声称最优/等效/实测更快。接下来F16独立CPU/GPU准入，其他参数固定，代码review前不正式铺量。

### 最新硬范围确认

消融选择、统计、作图、补测只用HotpotQA和GovReport；其他任务即便在历史AVG5源manifest出现也不进入本轮分析。当前E每任务200独立CI、C400请求×3臂、22个历史质量文件重评、全部F smoke均已遵守。nl4096决策不使用AVG5；若需汇总只允许明确标注两任务等权AVG2，不混称AVG5。历史源manifest按原样留存供身份核对，原始任务列表不是本轮分析集合。

## Plan v4：F取消，所有性能评测暂缓

最新用户决定：sink4/SWA128视为经验默认，不再做F取值消融；保留已有历史128/128各组真实参数。F正式6400请求从未启动，已完成23条smoke与69CPU测试保留；精确检查125的8个已知父PID及ablation runner/pool均不存在，GPU0–7空闲，未杀他人任务。

D执行优化性能消融以及A/B/E筛选延迟、C缓存处理/更新开销全部暂停，等优化完成；不得新增任何latency、throughput、ABBA、timeline或Graph/fusion/overlap速度测试。本owner未启动过活性能任务，无需终止。继续仅HotpotQA/GovReport质量与gate/cache语义，旧数据优先、只补必要缺格。E质量/CI独立验收与图保留，不把smoke墙钟当性能结果。

## Progress v5：C 最小低阈值质量补测启动

E 独验完成，固定4096；F取消、所有性能仍暂停。C默认.05已获独立 usable-with-scope：Near-only由5条真实paired与395条同请求all-keep复用组成，非400条新生成。Gov98单样本下降2.9975pp保留，不宣称无损。

低阈值仅.01/.025：127个请求重做历史no-skip精确桥接、139个Far分支真实生成；near均不触发，其他未变化请求可按准入条件复用。设计见C_LOW_THRESHOLD_PLAN.json，C_reuse_admission.json绑定其SHA。每请求Full token及完整profile必须精确匹配，否则全局STOP，不做条件复用。

125 GPU0–5已启动，pool PID73106、worker73109–73114；进程/GPU/日志三重核查通过（六卡各约17.5GB、日志START full_bridge）。公共UUID锁与NFA卡锁同时持有，任何既有未released租约拒绝覆盖；GPU6operator和GPU7cache排除。最多3次重试、checkpoint可恢复，结果每60秒自动回传C_low_threshold，所有raw在独验前仍pending。113另建Qwen路径传输中，不改GLM环境，尚未计为可运行。

### v5实际门禁停止（不可把启动当完成）

首批6条Gov（32/81/99/102/110/130）Full桥接全部token不一致，profile完全一致；首个分歧108–257步。所有Far分支均未运行。全局STOP使六worker全部exit3，不放宽门禁；已核原PID不存在并取得公共锁，只释放本owner六lease。原始文件C_low_threshold/requests与bridge_diagnostic.json保留，结论为不能按该环境直接条件拼接低阈值结果。已知旧torch2.8.0+cu128、新2.9.0构成候选原因，未归因。125另建venv-torch28，准备Gov32单请求定位；不改他人环境。此前默认.05的5条精确桥接仍仅在已验收局部scope内有效，不能推广到本次6条。


## Progress v6 — 2026-10-30T18:07:52.290289+08:00 root接手与正式A运行

执行agent/reviewer服务额度中断后，root接手已有经源码及smoke核查的A队列。源码4dab6e8ccb84bde38cfd8b9a06d65e7f766629a6；113 controller252159、125 controller90979，当前113八卡、125零至五卡共14卡。125 GPU6既有operator预留不碰，GPU7保留缓存正确性路径。10个配置×任务单元，各200请求，共2000请求；仅HotpotQA/GovReport，真实任务生成上限，非4-token smoke。A关闭区域skip以隔离子空间影响。两机自动接力controller已运行，本机collector3128478每120秒回传。最新本地已回传55完整请求（113:33，125:22），原始数据pending，不据此提前作质量结论。健康证据A_HEALTH_ROOT_CONTINUE.json、A_progress.json、A_LAUNCH_ROOT.json；外置controller字节留存quality_pool_dispatched.py。

C低阈值六条旧桥接失败仍STOP，隔离Torch2.8依赖正在安装，未宣称解决；Paged65步尚未准入。30分钟AI监督可能同受额度限制，不能宣称持续决策正常；远程派单和本机回传不依赖AI推理，已核持续推进。F取消、所有性能暂停保持。


## Progress — 2026-10-30T18:32:30.869968+08:00 监督执行

A原14卡队列继续，最新回传329/2000，两个sync0；2000请求计划两机互斥，早期145条逐行身份核查通过，不提前出质量结论。C依赖已真实修复：125独立venv实际torch2.8.0+cu128/TF4.56.2/triton3.4.0，13必要模块CPU import通过，CUDA未初始化，Torch要求全部满足；system/A环境抽查未改，完整环境仍继承system且pip check有无关冲突，不能当旧运行完全复刻。C STOP SHA不变、6失败保留，未重跑token桥接，不归因Torch。详细supervision1011_ablation/REPORT.md。

B独立新会话全1600行重新评分、原答案顺序和200分母核查通过，报告supervision1011_B_verification/REPORT.md；40证据SHA主控复核全通过。但仅历史评分usable-with-scope，运行身份/selector因果复用仍blocked，不能把不同αβγ说单因素，保护参数不得伪改4/128（128/128也缺exact-run绑定）。F取消、性能全暂停。


## Progress — 2026-10-30 19:03 北京时间监督

A原14卡队列已683/2000（113393/125290），均attempt1；独立supervision1041_A_review固定494条GovReport快照的重分词/官方评分/门控/缓存与40互斥分片核查无错误、全部400输入预检完成，完整A仍pending。严格full verifier已实际验证拒绝partial；主控732稳定文件SHA相符，session.log因CLI末尾追加不匹配已显式排除，不称全清单全过。已确认manifest candidate_mass描述与实际不收集回执不符，禁止mass结论；model/dependency历史绑定不足已采两机全部权重/metadata及部分依赖当前SHA补证，但不追认内存加载。没有质量输出缺陷证据，不中断在飞队列。

C单Gov32新wrapper准备完，72旧源与实际输入身份一致；独审supervision1041_C_review/ADMISSION.json通过且真实pc40完整CPU preflight exit0。主控部署到pc40同绝对新目录，双锁/空闲/无compute进程门禁后唯一父1909628/子1909629点火。19:02:40实见17.6GB/42%、五分片加载完成、start.json与约16.9GB读计数。启动SSH20秒超时但远端job持续且无重复启动。只Full一请求/零重试，不恢复旧127/139Far，旧6失败STOP保留。新结果尚pending独验，原始收回后继续本轮验收。无性能测量。


## Progress — 2026-10-30T19:09:44.336126+08:00 本轮收尾

报告/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T104152/ROUND_REPORT_ZH.md。旧V1/Qasper/corrected终态保持；A最新815/2000原14卡接力，全量pending；固定494独审无评分/输入错误，candidate-mass元数据限制保留。C在pc40GPU0已实跑Gov32单Full512token并完成不同会话独验，22绑定相符，全部TLI/KV回执也精确相同；旧STOP不删，不恢复127/139Far。Paged新隔离CPU guard独验完成172绑定，模型数值/长CUDA/GLM仍未闭合，不替换ring。F取消/性能暂停保持，无COMPLETE。


## Progress — 2026-10-30T19:23:28.705583+08:00 本轮监督进行中

19:23阶段更新：A 1620/2000，sync0、无FAILED。pc40 GPU0已恢复空闲（1MiB/0%/无compute/无未释放lease），C五条不同会话准入后38绑定通过；主控唯一启动parent1918954/child1918955，已实见17.6GB/43%、五权重分片加载完成及start.json/16.5GB读计数。旧STOP不删，结果pending。Paged CPU分解待独验。 旧V1逐记录3750/11250/128done、corrected100/400保持独验/报告完成，3897 raw与4722+958绑定连续性通过；本轮不重读5大权重、不新生成旧任务。证据ai_supervision/round_20261030T111152；F取消、性能暂停、整体无COMPLETE。


## Progress — 2026-10-30T19:39:42.638175+08:00 本轮收尾

本轮报告/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T111152/ROUND_REPORT_ZH.md。旧V1/Qasper/corrected终态保持。A最新1823/2000（30/40done、10worker）继续，完整pending。C其余五条真实各512token/全部exit0，与旧Full完整回执精确相同；不同会话独验/home/wangyuanshuo02/two-level-attention/agent_doc/results/nfa_ablation_20261030/supervision1112_C_review/RESULT_REPORT.md，主控180绑定通过，旧STOP/失败保留、127/139Far不恢复。Paged五脚本CPU独验数值一致，原TASK哈希漂移合同invalid保留；新版本/home/wangyuanshuo02/two-level-attention/agent_doc/results/nfa_paged_cache_20261030/supervision1112_rebind_1933/REPORT.md限定CPU usable-with-scope，230manifest绑定通过，不替换ring、完整模型因果/长CUDA/GLM未闭合。F取消/性能暂停/原1800秒保持，无COMPLETE。


## Progress — 2026-10-30T20:09:46.934307+08:00 19:42监督收尾

本轮A从1842到2000/2000、40done，全attempt1，原作业自然结束；不同会话完整输入/评分/门控缓存/逐cell配对B10000及CI已验收usable-with-scope，主控6726稳定绑定通过。覆盖Hotpot四格/Gov六格，不扩大为全部A–E完成；Gov highfreq32相对full128下降8.9656pp保留，full128非dense，无mass/性能结论。C旧候选CPU独验通过但资源准入拒绝；125新候选293CPU/127preflight/226部署完成，目标环境独立准入仍pending，未GPU。Paged CPU独验限定通过，P-1缺陷已新目录修复37回归及独立61例通过，未整体GPU准入/未替换ring。旧V1/Qasper/corrected完成范围维持，3897raw/4722+958绑定通过。完整报告/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T114205/ROUND_REPORT_ZH.md；NEXT_STEPS有精确入口。F取消/性能暂停/1800秒不变，无COMPLETE。


## Progress — 2026-10-30T20:39:54.730118+08:00 20:12监督收尾

本轮旧V1/Qasper/corrected范围继续完成；3897raw/4722artifact/958Qasper及A6726绑定全通过，A2000/40done。C实际首启动模型前CUDA初始化断言失败、0Full/Far，旧STOP保留；新隔离supervision1212_C_fix真实trace修复并获不同会话fix-v2准入，317文件/127输入/失败记账/真实锁生命周期独验，主控19绑定通过，未正式重试。Paged119完整包独审因全局TASK漂移invalid保留，版本化context-v2独审限定通过（471证据/119文件，root20绑定），本轮未GPU A/B，模型因果/长CUDA/GLM仍未闭合。F取消/性能暂停/原1800秒保持，无COMPLETE。报告/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T121204/ROUND_REPORT_ZH.md；NEXT_STEPS含确切新入口。收尾仅更新Progress，不改Plan/实验身份，独审原文档快照保留。


## Progress — 2026-10-30T21:10:52.091616+08:00 20:42监督收尾

旧V1/Qasper/corrected原完成范围保持，本轮3897raw/4722artifact/958Qasper与A6726绑定通过，A2000/40done。C真实首尝试缺accelerate失败保留；新隔离overlay-v3已独审并实际完成hotpotqa0000 Full+Far.025一配对，独立验收usable-with-scope，root135绑定通过，不扩127/139完成。Paged A真实生成并独验150绑定通过，各自历史精确而非backend等价；B已实跑439文件/28.1GB/432层状态，producer三门true，回传与完整独验仍pending，不能消费。collector3621630串行收尾初传3617951，不重复派GPU；GPU0/2空闲双锁释放。F取消/性能暂停/1800秒与STOP保持，无整体COMPLETE。报告/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T124220/ROUND_REPORT_ZH.md；NEXT_STEPS有精确新入口。


## Progress — 2026-10-30T21:38:58.447959+08:00 21:11监督收尾

旧V1/Qasper/corrected/A原完成范围保持，本轮3897raw/4722artifact/958Qasper/6726A绑定通过。Paged B439文件28.1GB/432状态完整独验，root657绑定；描述性比较216对/3360字段独立复算，root467绑定，首次共同QKV+mask精确而attention差异step1/layer1 maxabs6.103515625e-05，不是step5因果闭合/等价/默认替换。C隔离v3修复controller异常持久化及nonowner不得写STOP，经独审+root286绑定后21:34:24启动125GPU0唯一controller152128；21:37:40新3Full/3Far/3请求，自动接力无错误，目标126Full/138Far，raw pending。旧失败/累计attempt保留，不启动旧v1/v2，不重派。F取消/全部性能暂停/原1800秒保持，无COMPLETE。报告/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T131152/ROUND_REPORT_ZH.md；NEXT_STEPS有精确已有队列/回传/独验入口。

最终快照 2026-10-30T21:39:38.692311+08:00：远端2026-10-30T21:39:11.416687+08:00新5Full/4Far/4COMPLETE，ERROR数0，STOP=无；controller152128保持原队列。21:39:12本地最终rsync exit0，已回传5完整请求/5Full/5Far，共40文件，raw pending。未作新质量结论。


## 本轮收尾 2026-10-30T22:11:01.336640+08:00
C22:08已36Full/36Far/36完整、无ERROR/STOP，原controller152128继续，raw pending。Paged CPU/native GPU不同会话独验完成，root419/612绑定通过，12tensor各自复现原输出且layer1三个元素差异保留，不解释step5因果。C工具v3独立19反例通过、限定usable-with-scope；root通用绑定检查不适配其schema返回0项，未声称该绑定消费通过，下轮须按实际VALIDATION结构核验。F取消/性能暂停，整体未完成无COMPLETE。报告 /home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T134212/ROUND_REPORT_ZH.md。本地文件不会自动推送当前聊天。


## Progress — 2026-10-30T22:39:55.604283+08:00 22:11监督收尾

旧V1/Qasper/corrected/A完成范围保持，3897raw/4722artifact/958Qasper/6726A本轮连续性通过。C从41到67完整，67Full/68Far，无ERROR/STOP，原125GPU0 controller152128继续，raw pending。C44条/4000来源映射限定独验完成，partial正确拒绝；最新67快照已回传。Paged新CPU及GPU内部trace已不同会话限定验收，38tensor/1683648元素、两层PV/control复算通过，root143绑定；layer1三概率差/三输出差，非nativeIR/step5因果/模型等价。B五格2000请求最小补缺计划及98+2CPU已独审，root155绑定，历史身份仍不能补齐；新B需身份封套+独立GPU smoke，不需重复用户授权。F取消/全部性能暂停，整体未完成无COMPLETE。报告/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T141217/ROUND_REPORT_ZH.md；NEXT_STEPS含精确已有队列、实际manifest与下一步。仅追加Progress，原失败/Plan保留。本地文件不会自动推送当前聊天。

## Progress — 2026-10-30T23:10:37.209746+08:00 22:42监督收尾

旧V1/Qasper/corrected/A完成范围维持，3897raw/4722artifact/958Qasper/6726A绑定本轮全通过。C从73至90完整，23:09远端90Full/94Far/90COMPLETE，无ERROR/STOP，原125GPU0 controller152128继续；固定回传89/92/89机械与conditional正确拒绝partial，质量pending。Paged两层CPU阶段/候选算术已不同会话限定独验，root226绑定通过，64head/65480有效概率/8192输出/2048块及327400scalar复算；不支持nativeIR/step5/长CUDA/GLM或替换ring。B冻结400真实输入及完整模型/import身份，v4独审+3151跨host绑定后首启动初始化断言失败，0语义/模型，旧run保留；新隔离v5独审447检查/root33绑定后23:07:50唯一启动PID227561，实际25语义tensor+10四tokenmodel请求完成，结果待独立验收，正式0/2000。F取消、全部性能暂停、1800秒/STOP不变，整体无COMPLETE。报告/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T144206/ROUND_REPORT_ZH.md；NEXT_STEPS含精确入口。本地文件不会自动推送当前聊天。

## Progress — 2026-10-30T23:39:57.599254+08:00 23:12监督收尾

V1/Qasper/corrected32768/A原完成范围维持，3750/11250/128done及三机终态实核，3897raw/4722artifact/958Qasper/6726A绑定通过。C本轮92到100完整，末次23:38:28=100Full/105Far/100COMPLETE，ERROR0/无STOP，原125GPU0 controller152128继续；固定99/103/99机械/conditional正确拒绝partial，不评分。B v5 smoke实际独验561语义/1540身份及400重分词/2276依赖检查后，23:19:27唯一正式点火125GPU1 PID232849，23:38:30真实204/2000；固定回传202（Hotpot200+Gov2），已审机械exit2、0内容错误，完整科学验收pending。首完整计算三验实际40秒超30秒已披露，旧v4失败保留。

Paged新CPU末端投影独验3720检查/root40绑定；40捕获lineage独验757/root56绑定；实际125GPU2 step5/layer2无计时2×2在两原对角精确门通过后完成8tensor，exit0、自然释放。不同owner全raw/前后1387环境+108bundle独验usable-with-scope，root146绑定通过，报告本轮independent_review/MIXED_RESULT_REPORT.md。固定backend换历史KV1122/1123差异、固定历史换backend2/0，限定局部历史整体干预，不是step1单因果/全模型正确性，不替换ring。首次root远端封套SyntaxError未执行，修正版仅一次启动，失败保留。

F取消、全部性能暂停、1800秒/STOP不变，整体尚缺C/B全量及独验、Paged完整正确性，不写COMPLETE。报告/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T151204/ROUND_REPORT_ZH.md；NEXT_STEPS更新精确入口。本地文件不会自动推送当前聊天。


## Progress — 2026-10-31T00:03:51.478788+08:00 23:42监督收尾

本轮V1/Qasper/corrected32768/A原完成范围保持：3750/11250/128done三机终态，A950+1050/40done；3897raw/4722artifact/958Qasper/6726A绑定通过。C初101到107完整，00:02实时108Full/113Far/107完整、ERROR0/无STOP，原125GPU0 controller152128继续；固定105/110/105机械及conditional正确拒partial，338当前远端输入/源码/依赖绑定通过。B初207到221/2000、原125GPU1 PID232849双锁继续，固定210机械0内容错误但缺1790；解码/EOS/n=1缓存补充工具独立17边界及真实210拒partial通过，root225绑定，正式质量仍pending。

Paged本轮2×8局部真实32tensor和长guard真实六CUDA案例均完成不同owner限定独验，root164/172唯一绑定通过。2×8 v1协议CPU遗留noGPU矛盾invalid保留，v2仅文案修正，首短作业未采到活显存限制保留；long_guard实采活父子/412MiB/cases，3138检查通过，FAST1两长边界写前拒绝全页字节未改、FAST0三长度及FAST1阈下允许。CUDA选中1020/CPU1024差异保留，不称selector等价。不是实际首次ring fallback异常、FAST1长模型/GLM/整体模型支持；不替换ring。两作业自然exit0/GPU2释放，无重复GPU派单。

初spawn上下文失败、只读C probe dict处理失败与长guard CPU首版无driver失败均保留并修正隔离检查，不改生产源码或门限。F取消、全部性能暂停、原1800秒/STOP保持，整体尚未完成无COMPLETE。本轮报告/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T154200/ROUND_REPORT_ZH.md，NEXT_STEPS有精确合同/队列/工具/独验入口。仅追加Progress，不改Plan或guide；本地文件不会自动推送当前聊天。


## Progress — 2026-10-31T00:34:32.849777+08:00 00:12监督收尾

C本轮110到117完整，00:31:57实时117Full/126Far/117完整、ERROR0/无STOP，原controller152128继续；固定晚116/125/116的241臂不同owner内容零错但缺项/终态拒partial，未评分。C_seal_v2修复671历史依赖直绑，独审8软件/root6绑定通过，真实完整分支仍未运行。B229到244/2000、原PID232849双锁继续；固定229机械/解码0内容错误但缺1771，新增4493当前身份依赖不同owner复跑通过，仅interim，完整质量pending。A2000/40done既有独验与6726绑定保持。F取消、全部性能暂停，仅Hotpot/Gov。报告/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T161204/ROUND_REPORT_ZH.md；NEXT_STEPS有准确工具/队列入口。未重启/重派，整体无COMPLETE。


## Progress — 2026-10-31T01:09:34.355951+08:00 00:41监督收尾

本轮V1/Qasper/corrected/A既有完成范围保持，3897raw/4722artifact/958Qasper/6726A绑定通过。C低阈值从120到126完整，126Full/138Far/126COMPLETE及QUEUE_DONE/EXIT0，902文件收割；完整机械/4000映射/1471唯一来源臂语义已过，官方评分与最终科学独验尚pending。路径别名封装失败已保留并用原工具canonical路径重建，未改数据/生产/门限。B252→269/2000原125GPU1继续，固定253机械/解码0内容错但拒partial。Paged单次125GPU2四CUDA自然ring首调用完成六项独验，root305绑定通过；未复现异常，不称修复/完整模型/长Paged支持。F取消/全部性能暂停/原1800秒不变，无整体COMPLETE。报告/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T164152/ROUND_REPORT_ZH.md。


最终更正 2026-10-31T01:09:34.411438+08:00：独审会话结束后root核验PID4059109已不存在，C_SCORES_CPU.json未生成，不能沿用HANDOFF的“CPU仍活”快照。本轮C完整数据/封存/语义及桥接完成，但官方评分/CI/最终科学验收未完成，C_RESULT_VALIDATION_PENDING.json明确pending；禁止科学结论。下一轮先核无活checker后在独立会话恢复CPU评分（不能直接执行已disabled的finalize_c.py，需审绑定与真实结果），不重派GPU。原因未确认，不伪称持续运行。


## Progress — 2026-10-31T01:31:20.961322+08:00 本轮监督收尾

本轮报告/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T171152/ROUND_REPORT_ZH.md。V1/Qasper/corrected32768/A原完成范围保持，本轮3897raw/4722artifact/958Qasper/6726A绑定通过。C126/138/126完整及1471来源官方评分/4000映射/20组逐任务CI/144000层gate-cache已不同会话六项验收，原始与CPU包root各4579绑定通过；中文表/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T171152/C_REPORT_ZH.md。旧失败/首次CPU788检查点保留；验证脚本新v2只移出不变词表长度查询，官方scorer/统计/生产源未改，上轮退出原因未知。

B本轮273→285/2000，原125GPU1 PID232849继续；固定282快照机械/解码0内容错但exit2拒partial。Paged隔离两文件静态校验前移修复获CPU限定独验/root1744绑定，94producer+192独审检查通过；未GPU/未集成owner源码/未替换ring，完整模型与长FAST1仍缺。所有会话完成，无遗留CPU检查。F取消/全部性能暂停/原1800秒与STOP不变，整体未完成无COMPLETE。本地文件不会自动推送当前聊天。


## Progress — 2026-10-31T01:58:02.620879+08:00 本轮监督收尾

本轮报告/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T174152/ROUND_REPORT_ZH.md。V1/Qasper/corrected32768/A原完成范围维持，3750/11250/128done三机实核，3897raw/4722artifact/958Qasper/6726A绑定通过；C双验收独立各4579绑定保持、不重评分。B本轮295→305/2000，原125GPU1 PID232849双锁/显存/日志/IO继续；固定296独立机械/解码0内容错误、exit2拒partial，末收割302新增6未扩验收范围。新完整统计及远端Python3.12 AST回执工具获限定独验，root3809绑定通过；结束后仍须400输入/权重/source/import终态及全量科学独验。原队列固定10job/非阻塞锁/单PID准入无安全空卡移交，未重复派单。

Paged新隔离三文件同步异常整请求失效保护完成CPU独验/root1654绑定；producer200故障/288正常step/16隔离，独审96组2209检查。不是回滚/异步CUDA保护，未GPU/未集成dirty owner/未替换ring，完整预训练模型/长FAST1仍缺。首版pytest依赖及B检查AST版本/decorator失败保留，生产源/阈值未改。F取消、所有性能暂停、原1800秒/STOP不变，整体未完成无COMPLETE。本地文件不会自动推送当前聊天。


## Progress — 2026-10-31 02:16 B监督执行

原125GPU1 PID232849继续，02:13:58实时319/2000，原双FLOCK/GPU显存与利用率/日志IO有实核，无STOP/complete；固定319快照Hotpot200+Gov119，旧末302条全字节同。既有已审机械/官方逐条评分与解码工具均实际exit2、0内容错误、缺1681正确拒partial；不同owner独验pending，未出质量汇总。证据本轮round_20261031T021254/nfa_ablation/B_REVIEW_ENTRY.json、REPORT.md。未重启/抢尾/改source/lease；F取消、全部性能暂停、仅Hotpot/Gov。A/C/E不重评分，完整B质量仍pending。


## Progress — 2026-10-31 02:23 B末次收割

原125GPU1 PID232849从319推进326/2000，02:23:19 GPU1 20552MiB/38%、双FLOCK/日志Gov118→125/IO写增量均实核，无STOP/complete。末固定326请求/357文件，rsync exit0；初319字节全保持，新增7仅收割未内容验，不能扩张319检查范围。证据round_20261031T021254/nfa_ablation/B_quality_health_closeout.json、harvest_closeout.json、B_CLOSEOUT_CONTINUITY.json。未重跑统计/新科学验收，未改原队列；本owner所有检查/收割已退出，无本地遗留进程，远端B继续。F取消、全部性能暂停保持。


## Progress — 2026-10-31 02:54 B监督收尾

原125GPU1 PID232849本轮341→349/2000，末02:53:59 GPU1 19212MiB/33%、双lease FLOCK及cell锁、log/IO实核，无STOP/complete。初341机械/解码已不同owner重跑，均exit2、零内容错误/缺1659，1122绑定通过；末349/380文件收割exit0，初341字节全保持，新增8仅收割不扩独验。完整质量pending。证据round_20261031T024232/nfa_ablation/HANDOFF.md及independent_review/B_REPORT.md。

只读精确源码核查B_QUEUE_HANDOFF_DESIGN.md：10job内存列表无撤销/移交，非阻塞范围锁加单PID合同不允许无扰动抢尾；未来只能在自然终态/明确故障/另授权安全移交后新run静态唯一分片，并更新准入与验收。本轮未实现/未派GPU/未改活source/plan/STOP/lease，无本owner本地遗留进程。A/C/E不重评分、F取消、所有性能暂停保持。

## Progress — 2026-10-31 09:20 用户授权并行加速，B 6 卡铺满

用户新指令「继续推进跑完全部的 ablation，有 bug 及时 fix」+「关注 GPU_resources.md 尽量并行铺满卡」= 另授权安全移交。本机 collector 已死、无 cron 监督，先恢复。

**执行**：
1. 实测 B 单卡 9h38m 跑 1012/2000（~34s/req），剩余 988 req 单卡需 9.3h；125 除 GPU1 外 7 卡空闲、113 全 8 卡空闲，资源严重浪费。
2. 写 `parallel_runner.py`（新文件，不改原 B 任何文件）：复刻原 quality_runner 的 timing_scope AST 修改 + bound_atomic + prepared frozen input 校验，单 cell/dataset/range 参数化。**同 cell 同 dataset identity 逐字节 MATCH**（`2d52db39d26d/gov` orig=para=252496615c...），smoke 通过。
3. kill 原 GPU1 PID 232849（已跑 10h06m，完成 1012 req 全保留，当前 cell 已落盘 12 req 保留）。
4. 铺 6 个并行 run 到 125 GPU1/2/3/4/5/6：`06da hotpot` ✅200、`a5f4 hotpot` ✅200 已完成；`06da gov`（GPU3 0-200 + GPU2 100-200）、`a5f4 gov`（GPU5 0-200 + GPU1 100-200）、`2d52 gov`（GPU6 0-200 + GPU4 100-200，输出到原 B 目录自动 skip 已完成）。
5. 输出目录：`06da`/`a5f4` → `/home/hadoop-aipnlp/tli-nfa-B-parallel/quality`；`2d52 gov` → 原 B 目录。全部带 frozen input 校验 + rows lock + identity skip。
6. 本机 cron 每 10 分钟监督（job 7aab7579），全部完成后合并 + 最终科学验收。

当前总进度 ~1482/2000，剩余 518 req 全 gov，6 卡并行预计 ~16 分钟收官。F 取消、性能暂停保持。113 8 卡未用（B 环境在 125，跨机复制无必要）。

## Progress — 2026-10-31 12:10 B 组收官与最终独立科学验收

125 最终扩至 8 卡静态互斥分片；所有 run 自然退出并释放 GPU，无 Traceback/RuntimeError。重叠分片按 identity/input 校验自动 skip，合并后回传本机 `agent_doc/results/nfa_ablation_20261030/B_final/quality/`。覆盖 5 cell × 2 dataset × 200 row = **2000/2000**；缺失、重复、额外 row、identity 错配和配对 input/answers 错配均为 0。监督 cron `7aab7579` 已删除。

独立只读 reviewer 使用官方 `sample_score` 全量重评分，2000/2000 最大绝对误差 **0**；同 row percentile paired bootstrap（B=10000、seed=20261031）及独立标量复核全部通过。状态 **usable-with-scope**，产物为 `B_final_review/REPORT.md`、`paired_statistics.json`、`completeness_checks.json`。

相对生产默认 `2d2a466b4e9a`（far=minmax、near=avg、partition/boundary=True）：HotpotQA 四个消融臂均未检出差异；GovReport 中分区 MinMax/MinMax `2d52db39d26d` 为 **−0.3788 pp，95% CI [−0.7462, −0.0232]**，无边界页 `a5f4cb3f7cf2` 为 **−0.3252 pp，95% CI [−0.6534, −0.0042]**，其余两臂未检出差异。结果在本数据与协议范围内支持保留生产默认的 near=avg 和 boundary；区间为未校正多重比较的探索性 CI，不据此宣称统计最优、等效或无损。

至此本轮用户要求的待跑质量消融全部收官：A/C/E 既有范围验收保持，B 完成全量生成与独立验收；F 仍取消，D 及全部性能测试仍暂停。独立验收未重跑 GPU，也未覆盖 token→prediction 解码、门控/cache 或完整实现无 bug 证明。
