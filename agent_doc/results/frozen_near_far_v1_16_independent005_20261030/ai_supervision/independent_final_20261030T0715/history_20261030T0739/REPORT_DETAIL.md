# 独立 verify_experiment_result — V1 全量收官

当前状态：pending。最终稳定证据以 EVIDENCE_SHA256.json 为准。

用户明确授权本会话作为独立验收者，与生产主控分离。PROJECT_ROOT=/home/wangyuanshuo02/two-level-attention；源码08ce44d6027a235080e7ff4127754af6b1ae91e0。仅CPU，exec login=false，执行检查时CUDA_VISIBLE_DEVICES空，Python -B。未修改生产源码、原始结果、guide、TASK、NEXT_STEPS或COMPLETE；未启动GPU/重复监督器，未干预生产进程或锁。Qasper专项由另一独立会话负责，本轮不重复。

已读取gpu-experiment-dispatch技能及项目文档治理、结果复用、结果验收、执行规范和收割章节，项目GUIDE、TL-FROZEN-V1-16-20261030、当前合同和验收计划。复用检索限本run指定的0644工具、0313统计工具验证、0415封存规范及历史smoke；不假称遍历全项目。adopt现成成套工具；adapt快照raw_index原始路径检查及FINISHED前置门禁，diff留档；defer全量结论直到真实全量验收。

## 已执行证据

- CPU_SOURCE首次pytest输出捕获遇共享盘临时文件FileNotFoundError，实际没有测试执行；保留失败日志。CPU_SOURCE_RETRY以-s禁用输出捕获后53项通过，覆盖四态缓存/迁移/退化边界、profile参考、tiny Qwen/TLI logits、V1停止规则、请求隔离。不能把CPU测试称作真实GPU重生成。
- SOURCE_REVIEW.md记录本轮对真实冻结生成、门控、缓存、TLI、统计与评分路径的源码审读。
- PRIOR_SMOKE_REUSE.json重新读取历史smoke9个绑定文件，全部哈希匹配；仅复用原3请求9臂/native parity记录范围，不对全3750授信。
- 07:27:46已观察3750请求文件、128done回执、队列128done。07:30:51 FINISHED.json及validation/summary/raw_index/scores.tsv仍不存在。文件数不是每条complete/三臂质量核验，不能擅写11250独立已验。
- 现成recheck/官方评分/cache/closeout B5000 TSV及六类artifact工具已备齐，等待用户要求的真实FINISHED及生产导出；尚未执行的项目不算通过。

## 六项验收（完整V1范围）

|项目|状态|证据与缺口|
|---|---|---|
|implementation|inconclusive|源码审读、53 CPU测试通过；全量实际记录对应检查未完成|
|reference_boundary|inconclusive|CPU参考与历史smoke限定复用；全量检查未完成|
|data_integrity|inconclusive|3750文件/128回执盘点；全量输入/原始SHA/导出/六类artifact封存未完成|
|numerical_sanity|inconclusive|未完成全量官方评分、独立评分、门控缓存及B5000 CI/TSV|
|measurement_validity|inconclusive|预定三臂同输入、任务等权AVG16和3pp工程门槛；最终结果尚未验证。无性能主张，不以N/A掩盖质量检查缺口|
|reproducibility|inconclusive|本轮CPU命令及历史smoke字节可追溯；全量独立交叉核查/真实manifest未完成|

## 固定口径与限制

3750配对/11250三臂/128done；Full为TLI gates-off，非dense FullKV。16任务AVG16等权，B5000配对percentile95%CI，seed42、冻结任务顺序与Far−Full/Dual−Full/Dual−Far顺序。逐任务及AVG16以delta>=−3pp判断工程点估计容忍；不称无损/等价，CI含0不证明等价。musique、qasper、passage_retrieval_en、passage_count、SAMSum均必须保留真实下降与完整分母。

未独立GPU重生成；未存Q/K，不能重算真实attention mass；启动后磁盘权重hash不能追认历史内存加载身份。缓存metadata回执不证明每一步GPU张量。Near-only保留staging KV，不能据此声称驻留内存减少；没有性能结论。旧AVG5、旧MK2与Qasper专项B10000不混入V1。

没有真实manifest/完整六项证据就不能标usable-with-scope。原run_contract指dispatch_plan且scope与计划不完全一致，不能把它冒充最终结果manifest。未写FINAL_REPORT_ZH.md或任何完成标记。

## 收官实际进展（以本段覆盖上文等待期间状态）

07:35:00观察到真实FINISHED与生产validation/analyze/export。PREPARE于07:36:27退出0，固定3750完整配对及生产产物，CAPTURE_BINDINGS/SNAPSHOT绑定。FULL_RECHECK按用户门禁后启动，直接只读完整生产目录；CACHE同样读取完整生产目录，其余检查读取固定副本。

CACHE实际3750条通过、errors=[]、changed=[]。SNAPSHOT_CACHE_BINDINGS逐一核对全部3750快照SHA与缓存检查前后稳定的生产SHA一致。AUDIT_SCOPE完整3750/128manifest/128done通过，源码与git对象核对及历史2296交集一致，无错误。CLOSEOUT实际完整B5000三比较、48逐任务+3AVG16 CI、任务等权、工程3pp及五份TSV、predictions/index/门控派生字段全部通过，errors=[]。OFFICIAL实际11250臂官方冻结评分交叉检查通过，errors=[]。这些是实际执行结果，未将生产者pass当独立验收。

完整独立tokenizer输入/独立评分/门控几何recheck仍未完成。六类真实artifact与模型全分片重hash封存、统一scope真实manifest及最终报告正文绑定尚未完成，因此整体仍pending；不发布质量结论、不写FINAL_REPORT_ZH.md、不称usable-with-scope。六项总体均inconclusive；numerical_sanity已有完整官方评分与统计/TSV证据，但独立完整recheck尚未闭合；data_integrity已有完整覆盖与快照/缓存SHA连续性，六类artifact仍缺。

复跑入口保留FULL_V1_EXECUTION.md；准备及closeout路径适配见diff，所有工具均来自0644套件，未开发新通用工具。后续从此真实3750快照和已有成功检查续验，先重核字节，完成全量recheck后再封存并逐项授信。不能以截止时间放宽标准。

最终截止处理：07:39:08仅对本会话CPU recheck精确PID2777596发SIGTERM，核对父PID2777392及命令后执行；run_logged实际记exit_code=-15。未产生cpu_checks.json，不算完成或失败的科学判决。父wrapper无重派逻辑并正常记录后退出，未留后台检查。其余CPU检查已退出0。DEADLINE_ACTION.json与FULL_RECHECK.command.json保留真实记录。最终结论pending，后续需重新完成全量recheck，再真实六类artifact封存/重读/六项记录和FINAL_REPORT_ZH正文绑定。
