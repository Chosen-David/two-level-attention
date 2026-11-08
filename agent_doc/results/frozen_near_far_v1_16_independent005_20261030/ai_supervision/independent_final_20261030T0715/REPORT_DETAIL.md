# 独立 verify_experiment_result：V1 完整验收

结论：**usable-with-scope**。本轮实际完成3750配对/11250三臂/128done的独立CPU全量检查、真实六类artifact（含五个模型分片）封存与重读、六项验收和最终报告正文审读。有限检查不保证绝对无bug；没有独立GPU重生成，没有性能、无损或等价结论。

## 身份、权限与范围

独立Codex会话 `01a18ba9-1ca8-7cc1-a41d-89819b72673e`，用户明确授权并续用同一独立验收会话，与生产主控分离。PROJECT_ROOT=/home/wangyuanshuo02/two-level-attention；WORKFLOW_ROOT=/home/wangyuanshuo02/agent-research-workflows。07:44续轮截止08:08UTC。实际exec login=false，检查CUDA_VISIBLE_DEVICES为空、Python -B，仅CPU；未派GPU、未动生产进程/锁/源码、guide、TASK、NEXT_STEPS或COMPLETE。只写本独验目录及获授权V1/FINAL_REPORT_ZH.md。Qasper专项另有独立会话负责，此处只验V1的Qasper200请求、B5000口径，不替代专项B10000。

使用gpu-experiment-dispatch及其项目治理、复用、验收、执行和收割规范；原GUIDE、任务TL-FROZEN-V1-16-20261030及六项validation_plan均已读取。旧run_contract指向dispatch_plan，不能作为结果manifest。本轮使用现成prepare_spec/seal_artifacts项目适配生成统一精确scope的真实结果合同及manifest，原计划criteria逐字保留，未放宽阈值、未用N/A跳过检查。全部适用项为pass。

## 本轮实际执行与复用

- 原07:39截止时的pending报告、清单和记录已保存到history_20261030T0739；不把上轮被终止的recheck当成功。07:44续轮逐项重读上轮4167份证据，errors=[]，完整捕获对应生产文件SHA无变化。官方评分、缓存、closeout及源码覆盖检查因此复用，CONTINUATION_REUSE_CHECK.json与REUSED_CHECK_BINDINGS.json保留证据。
- FULL_RECHECK_CONTINUE真实运行07:43:55.011634–07:50:17.841222UTC，退出0。完成3750请求/11250输出、128manifest，running_files_skipped=[]、errors=[]。逐条重建tokenizer输入和截断/chat规则、原始answers、token decode、独立评分、stop规则、首token、严格0.05决策、缓存几何/可见性/节点容量和token守恒等。
- OFFICIAL已实际11250臂官方冻结sample_score交叉核验，无错误；独立recheck另有独立实现的QA/classification/retrieval/count/code评分，ROUGE使用共同参考库并明确这一依赖。
- CACHE已实际3750条 linked_order/attention_order/skip flags/stored_tokens/near-staging字节及TLI回执检查，errors=[]、changed=[]。所有3750 full-recheck raw SHA与捕获快照、缓存检查的生产SHA逐一一致；见FULL_RECHECK_SNAPSHOT_BINDING.json和SNAPSHOT_CACHE_BINDINGS.json。
- AUDIT_SCOPE确认3750/128manifest/128done、确定性rank映射/采样排列/身份/PID对应与无旧数据混入，源81文件及git对象绑定正确。FULL_RECHECK另核本地/共享部署两份源码。
- CLOSEOUT实际三组比较（Far−Full、Dual−Full、Dual−Far），48逐任务+3AVG16共51组B5000 percentile CI、16任务均值等权、seed42连续RNG顺序、sign counts、3pp工程点门槛、全部五份TSV、raw_index/predictions及门控/容量派生字段一致，errors=[]。
- CPU_SOURCE_RETRY实际53项通过，覆盖四态缓存/迁移/退化与错误条件、profile显式参考、tiny Qwen/TLI logits、SAMSum stop/min1及request guard。四份测试源另外与08ce44d git对象核对一致。首次CPU_SOURCE因共享盘pytest捕获临时文件失败、0项执行，改-s重跑；失败日志保留，没有掩盖。
- 历史3请求9臂smoke只在冻结核心不变的原范围复用，9项外部哈希重核并进入真实manifest。生产128次native parity是生产回执，不冒充本会话独立GPU复跑。
- FOCUS_FACTS直接核读musique/qasper/检索/计数/SAMSum全部对应请求。最终报告完整审读，51行比较表数值再次与独立closeout逐项比对；均分、触发、同token/同分、下降与限制相符。REPORT_PROSE_REVIEW.json绑定实际最终正文SHA。

## 真实封存与独立验收绑定

真实manifest：`tools/seal_v1_final_20261030T0744/artifact_manifest.json`。
同目录`result_contract.json`固定精确scope及manifest SHA；候选计划`tools/candidate_v1_20261030T075143685684Z/validation_plan.json`与原六项criteria一致，FINAL_PLAN_REVIEW.json记录适用性审读。manifest使用显式shared-storage-experiment-result/v1适配，允许真实canonical绝对路径和流式大文件；没有宣称通用项目内16MiB验证器或可信runtime provider已经安装。

六类引用数：code163、inputs30、config141、raw_data3879、outputs197、environment13；去重4410个实际文件。163 code含两份81冻结源码和封存器；检查脚本/日志/测试源也作为已绑定独验产物保留。原始数据含3750完整请求、128done及queue；128生产worker日志和最终导出均绑定。模型目录14文件含五个真实safetensors分片，sealer前后流式读取且VERIFY_ARTIFACT_SEAL再次全部重读通过。三机历史各13个模型/tokenizer磁盘哈希均一致；额外.gitattributes仅当前Git LFS元数据封存，不能追认历史未记录字段。

manifest本身保持sealed-pending-independent-review；封存工具只证明完整性。独立科学判定单独位于SCOPED_VALIDATION.json，精确绑定该manifest、validation_plan、完整4410文件哈希、六项证据和最终报告。这种分离避免用生产者pass或自写manifest身份替代独立验收。用户授权的真实会话构成独立身份来源，不以JSON自证安装了宿主provider。

## 六项结论

|项|判定|实际证据与范围|
|---|---|---|
|implementation|pass|冻结源码调用链审读、两处81源/git对象、53CPU测试、3750全量input/decode/score/gate/cache检查|
|reference_boundary|pass|四态/退化/异常、profile显式参考、tiny Qwen/TLI、stop/request guard；原smoke限定复用，不冒充GPU重生成|
|data_integrity|pass|3750/11250/128完整配对、输入/原始/导出SHA一致、三套检查快照连续性、六类真实artifact及模型封存后重读|
|numerical_sanity|pass|官方与独立评分、有限数/单位/决策/缓存不变量、51组CI/AVG16/sign/3pp/五TSV及派生统计|
|measurement_validity|pass|质量协议内相同输入/配置配对、任务等权和明确分母；Full=TLI gates-off，3pp非等价，无性能主张|
|reproducibility|pass|真实全量CPU重算及参考检查、命令环境种子、真实模型/输入/源码绑定并复验；GPU/QK/历史内存限制保留|

## 可支持的质量观察

Full/Far/Dual AVG16=49.462909/49.330056/49.325690分。Far−Full=−0.132854pp，95%CI[−0.235434,−0.032771]；Dual−Full=−0.137220pp，95%CI[−0.242975,−0.033832]；Dual−Far=−0.004366pp，95%CI[−0.013651,+0.004736]。全部逐任务及AVG16点估计在3pp工程容忍内，但Far/Dual对Full存在小幅下降，不能称无损或等价。

Qasper相对Full下降0.683604pp、11降/2升，Dual与Far200条同token同分；musique仅极小得分变化且Dual与Far200条同token；检索100分有满分上限，计数2分且196/200零分有低基线，两者都不能证明普遍安全。SAMSum Dual−Far仅2请求升分、195同token/198同分，区间下端为0。另保留multi_news Dual−Far −0.049934pp、CI[−0.105101,−0.008278]的下降，避免AVG16掩盖子任务。

Dual near-only319层、both5层、Near活跃105请求，Far活跃2258请求；Far决定与Far-only臂差异0。Near-only节点KV容量比例1.0，保留staging，不据此声称驻留KV节省。完整逐任务分数/CI/3pp/门控表在V1/FINAL_REPORT_ZH.md。

## 限制和后续消费

未独立GPU重生成；没有原始Q/K不能重算真实attention mass；当前和启动后磁盘hash不能认证历史内存加载；回执不证明每一步GPU张量；CPU有限边界不能保证绝对无bug。Full不是dense FullKV。B5000区间为探索性、未做多重比较校正；CI含0不证明等价。节点容量不含权重/临时gather/allocator，不是峰值显存；attention-token比例不是加速。无性能结论，不外推模型/阈值/输入/实现，不混旧AVG5/MK2或专项B10000。

SCOPED_VALIDATION.json是本轮独立可用性记录。EVIDENCE_SHA256.json封存本目录稳定证据并逐文件重读，排除活session/CLI输出和清单自身；哈希不替代语义审查。原PENDING_VALIDATION保留为历史并被本轮记录取代。任何代码/输入/配置/报告字节变化须重新核验相关绑定。未写COMPLETE，主控负责后续状态消费。

当前真实manifest SHA256：`1b396c215dd678875c56c31f49115e98c9bff1b77d12bff6946122f8f06b59af`。
最终报告 SHA256：`8cfcf98bfb61b7ab0558b288f2cae5569e7a7eb84f5af06d494cad51597af8e4`。
