# [TL-FROZEN-QASPER-MK2-20261030] Qasper + RULER MultiKey2 64K 双 gate 对照

Task-ID: TL-FROZEN-QASPER-MK2-20261030
Date: 2026-10-30

## Plan

用户要求补测Qasper与MultiKey2 64K，主问题为新方法Dual独立Near/Far .05相对于Far-only .05精度；Full为TLI gates-off辅助，非dense。K1=128/K2=1024，tail32，prefill尾256 query mean、strict<.05，各gate独立冻结。Near-only staging不省KV，Far/both物理驱逐，算子优化关闭。

Qasper200使用正在进行的currentV1同口径任务，完整后独立核验，不重复派发；历史full5仅带来源参考，不冒充当前代码。RULER当前数据100唯一ID、65536、YaRN2，原始prompt不额外chat、不截断，maxgen64，EOS+newline（首token不停止）沿现有RULER协议。单卡本机GPU0；GPU1保留算子优化，不中断全量V1。新代码隔离branch experiment/frozen-mk2-64k。

生产→verify_experiment_result（独立verify_pool_initial）→报告。先真实64K单例native-vs-node parity冒烟，再100条300生成。独立检查原始预测/哈希/分数/触发/缓存回执，10000次paired bootstrap seed42，报告Dual-Far和辅助差值。3pp为用户工程容忍界，不把CI含0或无触发当普遍无损。监督沿用已有30min AI周期，新增补测项，不重启E184。

## Progress

- 2026-10-30：已读取GUIDE重点AVG5+Qasper+MultiKey2 64K，未修改指南。result_store受既有symlink限制，改为明确路径只读核对；reuse_decision已记录。
- 历史Qasper200独立复核结果见新run/qasper_reuse_verification.md；当前V1结果等待全覆盖。
- 新adapter复用当前冻结cache/indexer，新增RULER入口/机械校验；独立预审发现停止规则差异，已对齐EOS+newline。尚未GPU派发。
- 结果目录：agent_doc/results/frozen_qasper_mk2_64k_20261030。

- 独立预审通过、56项回归＋独立4项CPU生成检查通过。source de2a1c1670b82bebcde6c0c3aa02fb8ee1431568 已提交隔离分支。本机GPU0 smoke PID2068330，启动8秒内进程/显存23.9GiB/100%util/模型加载日志三验证通过；GPU1未占用。

- 64K smoke独立验收通过（smoke_verification.md）：108层回执、native parity，Dual Near-only6层且6层迁移；不作单例质量结论。正式100条300生成已启动本机GPU0，parent2078085 / child2078087；三重验证见formal_ignition.json。代码de2a1c1已推origin+bee并ls-remote核实。结果写ruler64k/，max3次断点续跑，结束自动机械校验，待独立全量验收。现有PID1950631每1800s监督prompt已加入本任务及完成门，未重复新建监督器。


## Progress — 2026-10-30 本轮02:41:52 UTC监督

2026-10-30T03:08:50.731240+00:00：Qasper当前50/200，等待V1原队列不重复派发；MultiKey2@64K当前19/100配对、57/300臂，原parent2078085/child2078087在GPU0 attempt1正常。独立会话/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_qasper_mk2_64k_20261030/independent_review_20261030T0243/REPORT_DETAIL.md实际完成：全100输入重建、固定9条27臂972层CPU核验通过，236证据SHA全匹配；仅局部审计，全100仍pending，无独立GPU重生成。

证据：agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T024152/ROUND_REPORT_ZH.md。补采本机13模型文件与此前三机相符，但不追认历史内存权重。继续原300秒派单/1800秒监督；两任务全量机械+独验+最终报告均完成才COMPLETE。未产部分质量或性能结论；详见本轮closeout_requirements.md。


## Progress — 2026-10-30 本轮03:11:52 UTC监督

2026-10-30T03:38:29.429035+00:00：V1 1225/3750配对、3683/11250臂，37done/8running/83pending/0failed；Qasper current61/200；MK2 34/100配对、103/300臂，原GPU0 parent2078085/child2078087 attempt1继续。原队列自然接力，无重复派发或生产源/门限/guide修改。

本轮独立会话退出0，修复收官CI/TSV/门控导出漏检；21统计/导出+29缓存元数据合成测试和51CI逐次参考通过，4069文件SHA复核匹配。只接受工具验证，真实全量pending、未新增质量验收/独立GPU重生成。最终真实manifest/scope及MK2生产summary/report对照仍需补齐。有限重试修复只读巡检queue竞争；固定1125行快照收官拒绝退出2，不作质量结论。

证据：agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T031152/ROUND_REPORT_ZH.md；独验independent_review_20261030T0313/REPORT_DETAIL.md。下一步沿本轮closeout_requirements做完整机械+新版独验+两run最终报告，均完成才COMPLETE。本轮未写完成标记。


## Progress — 2026-10-30 本轮03:42:03 UTC监督

2026-10-30T04:08:39.802391+00:00：V1 1562/3750配对、4692/11250臂，49done/8running/71pending/0failed；Qasper current73/200；MK2 50/100配对、150/300臂，GPU0原parent2078085/child2078087 attempt1继续。原daemon自然接力，未重复派单/kill/偷lease/改生产源、门限或guide。

已done机械对账1459行、前后SHA稳定。独立MK2产物对照补验11CPU测试通过、785文件SHA匹配，仅工具与固定身份盘点，真实全量pending。同一独立会话定向补齐Qasper current200入口，报告/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_qasper_mk2_64k_20261030/independent_qasper_20261030T0355/REPORT_DETAIL.md；8个CPU测试方法通过，607份稳定文件主控SHA匹配，真实69/200预检查退出2、未tokenize/评分。全200/600真实分支及生产最终报告对照仍未运行，不能作为科学验收。成套复制qasper_current_check.py、qasper_record_checks.py、cache_receipts_check.py、specialty_statistics.py到专项run直接子目录后核RUN/SRC再用。

证据：agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T034203/ROUND_REPORT_ZH.md。最终真实manifest/统一scope与全量机械+独验+两份最终报告仍待完成；未独立GPU重生成，未写COMPLETE或部分质量结论。


## Progress — 2026-10-30 本轮04:12:04 UTC监督

2026-10-30T04:38:52.410405+00:00：V1 1890/3750配对、5679/11250臂，59done/8running/61pending/0failed；Qasper current83/200；MK2 65/100配对、197/300臂。 原daemon自然接力与本机attempt1继续；无重复派单/kill/偷lease/生产源、门限或guide修改。已done机械对账1720行、前后SHA稳定。

独立Codex会话01a18b03-7c7b-74c2-80cc-34d6d7e223c2实际退出0（collaboration后端no rollout found后改用单独exec）。报告：/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/independent_manifest_20261030T0415/REPORT_DETAIL.md。确认旧合同并非真实结果manifest；新增seal_artifacts.py/prepare_spec.py及FOLLOWUP_COMMANDS.md，分V1/Qasper/MK2精确scope封存六类实际artifact，显式处理共享盘别名，文件/目录变更拒绝，始终pending且不安装可信verifier。14项CPU合成测试通过，最终8项补验及prepare CLI检查通过；初始测试错误、共享盘ctime变化和FIFO不支持记录保留。主控69份稳定证据SHA复核全匹配。真实三scope preflight均退出2，未全量重评分/重hash模型/封存，不能当科学验收。Adopt工具与scope缺口修补；adapt为后续全量后在既有用户授权内执行，须真实独立六项记录及报告正文审读，不因工具提及“后续授权”新增确认流程；defer真实封存到完整输出齐。未独立GPU重生成。


证据：agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T041204/ROUND_REPORT_ZH.md。全量机械+真实manifest+独验+两份最终报告仍pending；未独立GPU重生成，未写COMPLETE或部分质量结论。


## Progress — 2026-10-30 本轮04:41:52 UTC监督

2026-10-30T05:08:03.134000+00:00：V1 2204/3750配对、6619/11250臂，71done/8running/49pending/0failed；Qasper current98/200；MK2 80/100配对、242/300臂。 原daemon自然接力与本机attempt1继续；无重复派单/kill/偷lease/生产源、门限或guide修改。已done机械对账2097行、前后SHA稳定。

独立Codex会话01a18b1e-f5c6-7a52-9cfe-452e186f1eeb退出0，固定V1 1960配对5880臂（含Qasper87）、MK2 69配对207臂，分别新增1559/60配对的CPU输入/评分/门控缓存回执审计通过；三机源/输入SSH检查通过。主控2597文件SHA复核一致。三次额外评分适配KeyError失败保留，按生产repobench别名及实际TREC all_classes修正后5880臂交叉检查通过，未改生产源码。仅限定技术审计，全量科学验收仍pending；未独立GPU重生成、无QK不能重算真实mass、启动后磁盘hash不能证明内存加载，均保留。报告：/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/independent_review_20261030T0444/REPORT_DETAIL.md。

证据：agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T044231/ROUND_REPORT_ZH.md。全量机械+真实manifest+独验+两份最终报告仍pending，未写COMPLETE。


## Progress — 2026-10-30 本轮05:11:52 UTC监督

2026-10-30T05:36:58.955149+00:00：V1 2531/3750配对、7603/11250臂，82done/8running/38pending/0failed；Qasper current124/200；MK2 95/100配对、285/300臂。 原daemon自然接力、MK2原attempt1继续；无重复派发/kill/偷lease/生产源或门限/guide修改。最新done机械对账2416行、前后SHA稳定。

独立Codex会话01a18b3a-7e68-7261-8bec-53161abeae2b实际退出0，固定V1 2296配对6888臂（含Qasper108）、MK2 84配对252臂，新增336/15配对；旧1960/69记录SHA全部不变。本轮全部固定记录CPU输入、decode/评分、门控缓存及三机源/输入检查通过，主控2996份封存文件SHA复核匹配。仅限定技术审计，全量科学验收pending；未独立GPU重生成、无QK不可重算真实mass、启动后磁盘hash不证明历史内存权重。报告：/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/independent_review_20261030T0514/REPORT_DETAIL.md。

证据：agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T051240/ROUND_REPORT_ZH.md。继续原300秒派单/1800秒监督及STOP；不重复派发、不抢lease、不触GPU1/E184。Qasper满200/600可先任务级独验；MK2满100/300执行冻结worktree validator及完整独验/配对统计；V1满3750/11250/128done后validate/analyze/export与全量独验。沿用既有closeout_requirements和independent_manifest_20261030T0415/FOLLOWUP_COMMANDS.md，真实六类artifact/manifest/统一scope与最终报告正文核验不可省略。两run均完成机械+独验+最终报告才COMPLETE。 本轮未写COMPLETE。


## Progress — 2026-10-30 本轮05:41:52 UTC监督

2026-10-30T05:59:31.998000+00:00：V1 2828/3750配对、8486/11250臂，92done/8running/28pending/0failed；Qasper current144/200；MK2 100/100配对、300/300臂。

独立Codex会话01a18b55-ec35-7bb0-89f8-8de07118f47a实际退出0，MK2完整100输入/300臂/10800请求层、源码/评分/门控缓存、B10000配对统计及修订后报告通过，54项CPU边界测试通过；六类真实artifact封存，模型5分片实际读取。主控601份独立证据及外部文件共817绑定SHA全部复核匹配，接受仅MK2 usable-with-scope。未独立GPU重生成、无QK不可重算真实mass、启动后磁盘hash不证明历史内存权重。

MK2 Full/Far/Dual=32/32/43分；主Dual−Far +11pp，95%CI[4,18]pp，下降2/提升13/同分85、同token49。Dual Near-only280层/88请求，Far与Both均0，不能推论未触发状态质量；3pp仅工程容忍，无性能/无损/等价结论。

已修正报告档位/实际token长度混写并通过独审；原attempt1完成，未重派/kill/改源或门限/guide。证据：/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T054203/ROUND_REPORT_ZH.md；独验：/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_qasper_mk2_64k_20261030/independent_review_20261030T0546/REPORT_DETAIL.md。

保持原V1三机300秒接力与1800秒监督/STOP，不重派、不抢lease、不触GPU1/E184。MK2已生成并限定独验完成，不重启原parent/child，不再生成100条。Qasper current满200/600后复用independent_qasper_20261030T0355成套工具，独立核验并出QASPER_CURRENT_REPORT.md；更新专项FINAL_REPORT_ZH.md时保留当前MK2已封存报告副本并重绑/复核新报告，原MK2验收只绑定当前字节。V1满3750/11250/128done后冻结源validate/analyze/export、全量独验、逐任务及AVG16 B5000配对CI/3pp/门控报告；专项B10000分开。真实artifact/统一scope和最终正文审读沿FOLLOWUP_COMMANDS，不混旧AVG5/旧Qasper，不扩V2。两run全部生产+机械+独验+最终报告完成才COMPLETE。


## Progress — 2026-10-30 本轮06:11:52 UTC监督

2026-10-30T06:38:23.041783+00:00：V1 3311/3750配对、9943/11250臂，109done/8running/11pending/0failed；Qasper current178/200；MK2 100/100配对、300/300臂。

独立Codex会话01a18b71-ee11-7080-9500-dae6ba3691f9实际退出0，固定V1 3036配对9108臂（含Qasper160）CPU输入、源码、官方评分交叉核验、门控缓存及三机源/输入检查通过；比旧2296新增740，旧交集SHA不变。主控3442份封存文件SHA复核匹配。只接受限定技术审计，全量科学验收pending。MK2本轮只核7个既有验收小文件连续性，不重新授信全artifact；未独立GPU重生成、无QK不能重算真实mass、启动后磁盘hash不证明历史内存权重。

本轮原daemon自然接力，done机械对账3199行前后SHA稳定；无重复GPU/监督器、kill/抢lease/生产源或门限/guide修改。证据：/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T061211/ROUND_REPORT_ZH.md。

保持原三机300秒接力与1800秒监督/STOP，不重派、不抢lease、不触GPU1/E184。Qasper current满200/600后使用independent_qasper_20261030T0355成套工具做任务级独验、QASPER_CURRENT_REPORT.md及真实scope/manifest封存；更新专项FINAL_REPORT_ZH.md前保留MK2已封存报告副本，之后重新绑定并独审正文。V1满3750/11250/128done后确认冻结源validate/analyze/export成功，再完整独验及逐任务+等权AVG16的B5000配对CI/3pp/门控报告；专项B10000单独处理。沿independent_manifest_20261030T0415/FOLLOWUP_COMMANDS.md核六类实际artifact和统一scope。MK2已完整生成并于0546限定独验，不再重启。两run生产+机械+独验+最终报告全部完成才COMPLETE；质量下降如实记录，不提高门限、不混AVG5/旧Qasper、不扩V2。


## Progress — 2026-10-30 本轮06:41:52 UTC监督

2026-10-30T07:09:12.749970+00:00：V1 3680/3750配对、11041/11250臂，121done/7running/0pending/0failed；Qasper current 200/200；旧MK2 100/100、300/300。

原三机接力继续；本轮逐样本、进程/GPU/log及done回执核验，无重复派GPU/kill/抢lease/生产源码或门限/guide修改。独立会话01a18b8c-fcc0-7603-9166-bd2360bc7539，具体验收与限制见/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/independent_closeout_20261030T0644/REPORT_DETAIL.md及/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T064152/consumed_independent_evidence.json。报告：/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030/ai_supervision/round_20261030T064152/ROUND_REPORT_ZH.md。

保持原三机300秒自动接力及既有1800秒监督/STOP，不重复派GPU、不抢lease、不触本机GPU1或E184。V1满3750/11250/128done后确认冻结源validate/analyze/export成功，再完整独验、真实artifact封存及逐任务+AVG16 B5000配对CI/3pp/门控报告。Qasper满200/600后优先完成专项B10000独验与QASPER_CURRENT_REPORT.md。旧MK2 original40960只作历史设置，不能进入original32768修正主表，不重启旧de2a1c1；纠错与统一配置重测由主会话TL-EVAL-BLUEPRINT管理，待修复独验后另派。专项报告改写前保存旧封存字节并重新绑定审读。两run全部生产、机械、独验、最终报告完成才COMPLETE；无独立GPU重生成须明确，无性能/无损结论。


本轮最后推进：Qasper已达200/600；主控运行/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_qasper_mk2_64k_20261030/mechanical_qasper_20261030T0709/qasper_current_check.py退出0，输出scoped CPU checks passed、128 covering manifests，静态快照/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_qasper_mk2_64k_20261030/mechanical_qasper_20261030T0709/snapshot_20261030T070911692926Z。此为主控机械调用，独立会话已先行结束，不可冒充独立全量验收；full_acceptance仍pending、未独立GPU重生成。下一步独立审该真实快照及B10000统计/报告正文/六类artifact封存，再发布QASPER_CURRENT_REPORT.md与专项最终报告。无需重复生成Qasper。
