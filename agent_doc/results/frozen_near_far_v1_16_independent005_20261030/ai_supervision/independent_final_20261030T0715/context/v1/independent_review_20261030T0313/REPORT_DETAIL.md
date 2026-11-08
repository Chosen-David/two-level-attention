# 独立 verify_experiment_result：收官工具验证

**真实实验结论：partial pending。本轮没有全量科学验收，没有发布任何部分质量均分、CI或3pp判定。** 用户显式授权的本独立会话与生产主控分离；执行于2026-10-30约03:13–03:24 UTC，15分钟预算内。不是历史合同中 `/root/verify_pool_initial` 的身份，也没有把该字段当独立性证明。

PROJECT_ROOT=/home/wangyuanshuo02/two-level-attention；WORKFLOW_ROOT=/home/wangyuanshuo02/agent-research-workflows。唯一写入范围是本PROMPT所在目录。执行均login=false，Python均为PROJECT_ROOT/.venv/bin/python -B，CUDA_VISIBLE_DEVICES为空、PYTHONDONTWRITEBYTECODE=1。未使用GPU、SSH、派单、监督器、kill；未修改共享源码、原始生产数据、旧绑定证据、guide或任务文档。

## 读取与复用边界

使用gpu-experiment-dispatch技能，读取项目文档治理、结果复用、结果独验、执行规范，人工指南和任务入口；读取NEXT_STEPS、round_20261030T021152与024152的closeout_requirements、V1旧0213报告和recheck/closeout脚本，以及专项contract/validation_plan/0243报告及检查脚本。关键内容固定于context，来源与SHA在CONTEXT_BINDINGS.json。

本次裁决为verify_delta：旧401条V1、9条MK2的CPU审计只能作为既有范围背景，不复制其pass到新增样本。用户把本轮重点限定为full分支工具验证，因此没有重复大规模tokenization/评分，不新增真实样本质量验收。没有读入或混合旧full5。算法依据是冻结源码与已明确统计合同；未使用外部文献替代本地验证。

两个worktree实际HEAD分别为08ce44d6027a235080e7ff4127754af6b1ae91e0、de2a1c1670b82bebcde6c0c3aa02fb8ee1431568，tracked工作区为空；本次未核untracked或重hash全部权重。REVIEW_ENVIRONMENT.json记录身份、Python/OS和环境。

固定queue快照是30 done / 8 running / 90 pending，未齐128。此为一次只读状态取证，不是GPU监督；不将它换算成当前已验请求数，也不把用户提供的约967/3750当本轮独立重算。生产仍增长，快照不授信未来产物。专项仍沿旧报告的partial pending，本轮没有重新盘点或验收100/200条。

## 实质缺口与修复

重大缺口已先写FINDINGS.md，实际反例保留：

1. **原closeout接受空CI。** 原程序对`zip(entry['ci95_pp'], ci)`使用all，空数组真空通过；正常3750合成夹具把全部51个CI置空后，原脚本仍退出0。新增closeout_check_v2.py要求每个CI恰好两个有限且有序端点，实际同夹具退出2。
2. **原工具不读取五份TSV。** scores.tsv追加损坏行，原脚本仍退出0；新增工具退出2。scores、paired_differences、activation_storage、per_task_activation、per_request_scores各自损坏均被检出。
3. **派生字段未充分核验。** 新extra_checks.py从raw decisions独立累计gate_counts、near/far触发、逐任务n/触发/token完全相同数、Far选择差异，以及resident KV、attention token、near-only比例与层数。检查固定16任务顺序、summary协议/分母、精确比较集合、index键与路径/input/hash/bytes、predictions的answers/input/gate_counts。故意破坏门控汇总、KV比例、分母、index dataset、prediction answers均失败。
4. **原recheck full不是完整收官器。** 独立执行其AST提取的真实`if args.require_full`分支，正常合成数据通过；CI全部置空仍通过，证实缺口。注意：这仅执行该分支，预填其前置records/recomputed scores，不冒充完整recheck端到端执行；本轮没有运行真实tokenizer/input/scorer链。原recheck必须与新增统计/导出检查配合。
5. **缓存补充遗漏需继续携带。** 旧0213 final_snapshot_check内的linked_order/attention_order、缓存skip flags、stored_tokens节点和、near-only驻留字节、TLI receipt flags不在原recheck的全部检查范围。已提取为可复用cache_receipts_check.py，新增29项四状态合成元数据正反例通过。它不是独立Q/K或KV张量值校验；本轮没有对真实新样本运行它。
6. **结果manifest仍阻断收官。** CPU_CONTRACT.log实际核对validation_plan SHA通过，但run_contract.scope与validation_plan.scope不一致，manifest_path仍指dispatch_plan，不是experiment-result/v1，也没有code/inputs/config/raw_data/outputs/environment六类完整清单。本轮只取证，未改合同或伪造适配。必须由后续完整结果清单或明确项目适配绑定统一scope与全部证据。

这些发现是**验收工具存在漏检**，不是已经证明真实GPU生成、真实均分或真实CI损坏。未提出无证据的GPU重跑要求。

## 实际CPU测试及统计口径

实际命令均为`PROJECT_ROOT/.venv/bin/python -B 本目录/脚本名`，环境如上。test_full_synthetic.py退出0，CPU_SYNTHETIC.log及各子进程日志完整保留；test_cache_synthetic.py退出0，CPU_CACHE.log保留；check_contract_snapshot.py输出pending审计JSON，CPU_CONTRACT.log保留。

SYNTHETIC_TEST_RESULTS.json有21组测试，all_pass=true；这是测试预期满足，**包括故意证明旧工具错误接受坏输入的测试**，不表示旧工具合格。SYNTHETIC_CACHE_RESULTS.json有29个通过的元数据案例。

合成统计夹具为16任务共3750配对/11250臂、128 manifests/done；任务分母200/150/500，与协议一致。各任务效应不同、配对差非恒定，避免全零退化掩盖宏平均或随机流错误。夹具的一层缓存只是统计输入结构，不模拟Qwen真实36层生成。

从冻结analyze/export源码经AST仅加载函数和CPU导入，去掉生成模块atomic_json导入并提供本目录写函数；只向SYNTHETIC_NOT_QUALITY生成测试产物，未导入模型生成器。冻结生产统计作为被测对象，新增closeout为交叉重算。另用独立逐次5000次抽样参考，按三组对比连续seed42 RNG顺序，核对48个逐任务CI及3个AVG16 CI，共51个CI全部一致。AVG16由16任务均值等权组成，宏bootstrap为任务内配对重采样后等权平均；没有改成3750 pooled平均。

三组为Far−Full、Dual−Full、Dual−Far，和冻结V1顺序一致。配对差单位pp，percentile [2.5%,97.5%]，B=5000。正常full分支原/新程序都退出0；统计破坏夹具同时改变AVG16、degraded和CI下端点，新程序退出2。3pp采用点估计delta>=−3，−3边界及两侧测试通过；这是工程容忍，不是等价检验，不以CI含0断言无损。

新程序TSV采用冻结导出的精确文本字段/顺序比对，会拒绝人为改写但数值等价的格式；这种保守拒绝需重新规范导出，不能静默放宽。新工具仍是**收官补充检查器**，其输出full_acceptance始终pending，不会写COMPLETE，不自动认证最终报告或历史GPU身份。

## 专项静态复查

MK2旧check_independent.py的完整分支明确100条后才运行10000 bootstrap seed42，顺序Dual−Far、Far−Full、Dual−Full；逐次采样参考核对整组字典，包含delta/CI/degraded/improved/same_score/same_generated_ids，以及各臂均值和触发数。它不复用V1的5000次CI。本轮没有重做旧专项合成检查或真实100条分支。

新发现需后续补验：该脚本在副本上调用summarize并核对其新结果，没有freeze并逐项对照**生产最终summary/report**，因此不能仅凭脚本通过认证最终发布产物；3pp工程描述也需最终报告单独核对。原脚本RUN=BASE.parent，后续若复制到V1审查目录会指向错误run，必须显式适配目标，不能机械复制后运行。本轮不修改专项旧脚本。

Qasper必须完整200条，再按专项10000次统计合同单独核验和报告，保持其与V1 5000次口径区别；不得将旧full5或MK2样本混入。专项完整artifact/报告核验仍pending。

## 六项契约状态

|项|本轮通过范围|真实全量状态|
|---|---|---|
|implementation|静态调用链和旧漏检反例；新full统计/TSV检查正反例|inconclusive，未跑真实全量|
|reference_boundary|51 CI逐次抽样参考；29缓存元数据边界|inconclusive，未独立GPU或真实张量参考|
|data_integrity|合成3750/11250完整分支、导出字段反例、固定合同/queue取证|inconclusive，真实未齐；manifest结构失败|
|numerical_sanity|合成均值/AVG16/3组CI/sign/3pp、门控缓存汇总|inconclusive，真实全量统计未检查|
|measurement_validity|明确任务等权、配对和分母；未输出部分质量|inconclusive，正式完整样本未消费；无性能主张|
|reproducibility|CPU可重跑脚本、完整日志与哈希；冻结HEAD核对|inconclusive，无独立GPU重生成或历史加载身份认证|

没有用N/A绕过未完成检查。本轮可用范围仅为**收官工具的已测试CPU行为、发现的漏检及固定合同快照审计**。不能将合成数据当科学实证，不能据此验收未来样本或整个实验。

## 后续收官执行边界

全量静止后，新独立会话应将本轮closeout_check_v2.py与extra_checks.py成对复制绑定到新验收目录，保留旧证据；先对固定原始快照跑原recheck.py --require-full，补cache_receipts_check.py，再跑closeout_check_v2.py。各次output须位于新审查目录的子目录。全量source/input/config/environment/raw/manifest/done及分析导出须前后hash一致，旧recheck未独立绑定的最终文件由新增closeout绑定；仍需完整结果manifest适配和人工最终报告数值/措辞审查。

统计检查通过不代替输入重建、官方评分独立复算、严格gate/缓存几何与byte检查、全部128唯一done覆盖和来源验证。合法掉点保留；陈旧summary/TSV优先重新分析导出，不因为本轮工具缺口就重跑GPU。

未独立GPU重生成；生产native parity回执不等于本会话重生成。历史模型权重/tokenizer是在启动后补采，不能证明历史进程内存加载身份。本轮未重hash16GB权重。未存Q/K，无法复算真实attention mass；metadata一致不证明实际GPU每步mask/KV数值。没有延迟、加速、整体显存或普遍无损结论。

未写最终科学报告、FINISHED或COMPLETE。EVIDENCE_SHA256.json绑定本目录稳定脚本、固定上下文、合成夹具/检查结果/已结束CPU日志和本报告；不含活session.log、生产活日志、FINAL_RESPONSE.md或清单自身。哈希证明字节绑定，不证明语义正确或独立身份。
