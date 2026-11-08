# 收官 artifact manifest / scope 独立审查

**只交付封存工具与审计；真实 V1、Qasper、MK2 全量科学验收仍 pending。未发布 partial 均分/CI/3pp，未写 COMPLETE。**

本会话由用户指定为独立 verify_experiment_result，非生产主控，未派子 agent。PROJECT_ROOT=/home/wangyuanshuo02/two-level-attention。新增内容仅位于本 PROMPT 的 independent_manifest_20261030T0415 目录；未改生产源、guide、任务、原合同或原数据。exec 全部 login=false；实际 Python 均 -B、CPU、CUDA_VISIBLE_DEVICES=''。没有 GPU、SSH、派单、监督器、kill、安装或发布。

## 依据、检索和取舍

已读 gpu-experiment-dispatch 技能、文档治理/结果复用/结果独验规则及执行规范、作战手册相关段落；当前 GUIDE、任务索引及 V1/专项两详情；V1 NEXT_STEPS 与 round_20261030T031152/closeout_requirements；专项0343和0355报告、两 run 的合同/计划。当前明确授权优先，不启动 GUIDE 提及的其它 agent/监督器。

检索限定用户指定两 run，采用实际目录/具体文件读取；没有声称通用 result_store 完成全库搜索或授信。决定 verify_delta：复用现有 input/tokenize/score/gate/cache/bootstrap 接口，仅补封存/路径/scope。旧 partial 结果不授信新增样本，旧工具历史测试未在本轮重复运行。

CONTEXT_BINDINGS.json 固定关键文档、生产源码、既有独验接口；REVIEW_ENVIRONMENT.json 固定本轮环境、canonical路径、实际HEAD。未把文献、README、哈希或自写pass当科学验收。

## 确认的缺口

1. V1 run_contract 的 manifest_path 仍指 dispatch_plan.json，scope 字符串与原 validation_plan 不同。派单记录没有真实六类结果 artifacts，不能改名冒充。
2. 专项启动合同合并 Qasper200 和 MK2@64K100。MK2完成不能代替Qasper，Qasper200也不应等待其余V1。
3. 0343/0355的 CAPTURE_BINDINGS 是独验取证清单，并非生产完整结果合同。0355仍缺 QASPER_CURRENT_REPORT/专项FINAL_REPORT 的逐字段/正文独立比对。本工具只绑定报告字节，未冒称补完语义审读。
4. generic只接受项目内普通文件且每文件16MiB；V1共享盘别名、外部模型/输入及GB级权重不能直接通过。本轮没有创建假 verifier 或把generic拒绝改成通过。

处理：保留旧合同，提供明确 shared-storage-experiment-result/v1 文件完整性适配；这是项目级待验收manifest，不是 generic experiment-result/v1 已通过记录。未安装可信宿主provider、未自动注册消费者。

## 实际路径与生产约定

V1逻辑路径：
/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_near_far_v1_16_independent005_20261030

其canonical路径：
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/tla/results/frozen_near_far_v1_16_independent005_20261030

本PROMPT/交付目录也经同一别名落在上述canonical run的ai_supervision/independent_manifest_20261030T0415。专项run为本地普通目录，实际映射详见REVIEW_ENVIRONMENT.json。

实际只读HEAD：
- 本地V1对照源 /home/wangyuanshuo02/two-level-attention-frozen-main 和共享部署源 /mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/tla/frozen005_v1_16_source_20261030 均为08ce44d6027a235080e7ff4127754af6b1ae91e0。
- MK2源 /home/wangyuanshuo02/two-level-attention-frozen-mk2 为de2a1c1670b82bebcde6c0c3aa02fb8ee1431568。

核读实际 prefill_frozen_skip.py 的 prepare_ids/select_row_indices/shard_inputs/main；V1输入为 /home/wangyuanshuo02/datasets/LongBench/data/<dataset>.jsonl。selected_rows按dataset顺序展平，再global_index%128分片；31500 cap，任务对应chat/stop/maxlen。complete record位于 requests/<dataset>/<row_index:04d>.json，分片结束才写done。固定identity=628463dbc68fcb474cd9c77b48f553d832c23983795748689b9748f63575f7e2。

Qasper200覆盖128个rank manifest，但无需这些rank完成其它任务；只使用当前identity，不混full5/3e4a688。

核读 frozen_region_compare.py、validate_frozen_compare.py、score_ruler.py、yarn_receipt.py、run_local.py。MK2既有JSONL100唯一index，raw input直接tokenizer、不额外chat/截断；65536、YaRN2、max64、EOS+newline且首token不停；selected_indices实际0..99，requests/<index:04d>.json和GENERATED。identity=dd0e84bc766e7131d0ef12f935535693eaf28821a05fc09bc84af7b4dfea9214。

核读V1 validate/analyze/export：实际导出5TSV、predictions.jsonl、raw_index.json、REPORT.md。既有独验工具负责重评分/统计/导出语义；新增工具不重写它们。本轮未SSH验证远端实际import或历史加载，HEAD仅证明本轮可见源。

## 可执行工具

### seal_artifacts.py

stdlib-only；不导入torch/transformers/生产模块，不执行spec中的command。CLI固定真实run、源版本与identity。只在脚本目录内新建seal目录，拒绝覆盖已有目录。

|scope|完整性门|统计绑定|
|---|---|---|
|v1|3750唯一request/11250臂、128manifest/128done、queue全部done并核PID/identity|5000 seed42；Far−Full、Dual−Full、Dual−Far；AVG16等权|
|qasper|200唯一request/600臂、128 covering manifest；不要求其它V1 done|10000 seed42；Dual−Far、Far−Full、Dual−Full|
|mk2|100唯一index/300臂、GENERATED identity/samples=100|10000 seed42；Dual−Far、Far−Full、Dual−Full|

先结构完整门，partial立即拒绝，未进入模型大文件hash/统计。检查规范文件名、身份、三臂、非空生成、有限保存score只是结构门，不替代真实评分。固定protocol digest绑定冻结配置。

完整后封存六类实际文件：
- raw_data：scope全量requests、V1 queue/done、MK2 GENERATED及已有smoke原件。
- code：protocol.source_sha256逐项与实际文件/Git对象一致；V1同时核本地和共享部署源，拒绝sparse_attn新增未绑定Python模块；MK2另含run_local.py，封存器自身也绑定。
- inputs：scope真实JSONL及model目录全部普通文件；读取weight_map，全体真实safetensors分片必须齐全且无未索引分片，流式完整SHA256，不拿索引摘要冒充权重读取。
- config：实际rank/protocol、prompt/maxlen、model/tokenizer/generation配置、原合同/计划、新spec/计划。dispatch_plan仅作为provenance。
- environment：V1三机既有model/software identity receipts，MK2 formal_launch，以及调用者显式提交的实际环境/command evidence。字节绑定不认证历史环境充分性。
- outputs：V1 summary/validation/5TSV/predictions/raw_index/REPORT/FINAL_REPORT；MK2 summary/validation及指定FINAL_REPORT；Qasper要求QASPER_REPORT_FACTS和QASPER_CURRENT_REPORT。显式提交完整独验快照、检查脚本及额外产物；--review-directory递归绑定完成快照，拒绝活session、CLI答复、symlink和整个含合成测试的review目录。

每类非空；spec/plan/contract/manifest的scope相同，B固定不混。只有coverage count指标，无partial质量；independent_validation=null、full_acceptance=pending。退出0仅表示文件封存完成。

路径适配显式保存logical→canonical映射；canonical文件及中间组件不允许symlink，声明的根别名在结束时重核是否改指。只读普通文件，4MiB块流式SHA256，无generic16MiB限制且不伪称generic通过。单文件前后stat，记录inode/device/size/mtime/ctime，全体结束再hash，目录成员前后重核；变化拒绝，生产工具不忽略或自动重试。

--verify重新核manifest/contract/plan、scope/B、全部文件/别名/成员，只输出integrity rechecked/pending。逐文件检查不是原子快照，也不能抵抗恶意改写后还原。封存后的任何绑定变化使重核失败。

### prepare_spec.py

仅在本目录新增candidate_<scope>_<UTC>。原六项criteria逐字段保留，绑定父plan SHA，只统一候选scope；明确candidate projection。专项父计划包含两任务，最终可信reviewer需核criterion applicability，不能因分scope静默放松。未修改/激活原协议、未新增独立GPU任务或改N/A。

输出、环境、真实执行command及其证据必须显式输入；命令只记录不执行。--review-directory枚举已完成独验snapshot，不凭文件存在认证独立身份。后续科学通过须原有真实独验与报告审读。

## 本轮真实执行与回归

- PREFLIGHT_V1/QASPER/MK2.log：三条实际生产路径均退出2，拒绝不满足精确完整scope的文件集合；未tokenize/评分/发布partial统计，未产生真实完整manifest。
- CPU_TESTS.log：14个测试方法通过。合成3750/200/100完整结构，缺行/多行/缺臂/旧identity/重复ID/rank/PID/queue损坏拒绝；Qasper不依赖其它done；字节漂移/目录变化/别名重定向/symlink/非普通目录、JSON重复键/NaN、scope/B混用拒绝。
- 合成MK2端到端使用真实本地Git对象、tiny-byte模型分片、真实hash/file I/O，封存并重核始终pending；检查脏源和未索引权重拒绝、已有seal不覆盖。非真实GPU实验。
- CPU_TESTS_attempt1/attempt2.log保留失败：测试类run属性覆盖unittest.run已修复；共享挂载mkfifo返回ENOSYS，改用真实目录测试非普通文件拒绝，未声称实际FIFO用例通过；合成新文件ctime延迟更新使前后核验拒绝，同集后续稳定（FILESYSTEM_DIAGNOSTIC.log）。最终夹具setup有限等待稳定，本体production逻辑未放松。最终日志具体记录ctime_ns变化。
- 最后补充复验与prepare辅助入口日志见FINAL_TOOL_CHECKS.log；该日志补验最终代码与新helper接口，不替代上述14项。
- CONTEXT_CAPTURE/CONTEXT_BINDINGS/REVIEW_ENVIRONMENT是实际只读取证；未全量重hash生产模型、输入或生成输出。完整真实封存与--verify大文件分支仍待齐全后执行。

## 后续操作与独验接口

精确CLI见FOLLOWUP_COMMANDS.md；占位符必须替换为未来真实完整产物，不可将合成目录作为证据。后续执行旧独验脚本会写旧review目录，需对应授权或在新授权目录成套复制。本轮未越界执行。

V1必须原recheck --require-full + cache_receipts_check + closeout_check_v2/extra_checks，全量机械validate/analyze/export在前；旧recheck有reference_dependency，复制时保留。closeout退出0不能单独收官。

Qasper0355入口要和qasper_record_checks、cache_receipts_check、specialty_statistics成套，BASE必须专项直接子目录；MK2 check_independent_full和closeout_supplement成套，RUN=BASE.parent。已有0343完整入口核生产summary/validation/report事实块，但正文仍独立审读；0355 facts需与QASPER_CURRENT_REPORT/最终专项报告逐项核对。

封存前审候选scope/六项适用性及artifact完整性；封存后由宿主认证真实独立reviewer、开始/结束、非生产者身份，绑定同一manifest SHA、artifact hashes、plan SHA、scope，为六项给出verdict/reason/真实证据和limitations。可保存experiment-validation/v1，但共享存储必须由显式受信项目适配处理，不能说generic已通过。所有适用项真实通过后才usable-with-scope，无记录则pending。自写pass、测试通过、工具退出0均不能代替该记录。

Qasper可先单独封存/验收；两链全部生产+机械+独验+各自最终报告完成才COMPLETE。专项整体关闭还需Qasper与MK2两scope记录，不能只引用一个。

## 限制与交付证据

真实全量六项验收本轮均未完成，保持inconclusive/pending，未用N/A绕过。未独立GPU重生成；无Q/K不能复算真实mass；启动后模型/tokenizer/软件hash不能证明历史加载；metadata一致不保证每一步GPU张量正确。没有速度、显存、等价或无损结论；合法掉点不阻碍真实数据验收，缺证据则阻碍。

EVIDENCE_SHA256.json绑定稳定脚本、报告、context、执行/测试日志；保留失败日志，排除活session.log、CLI最终答复、清单自身及绑定测试代码可再生的合成工作目录。哈希证明字节，不授予科学通过。本轮结束仍无真实全量artifact_manifest、validation pass或COMPLETE。
