# 独立 verify_experiment_result：MK2 最终产物对照补验工具

**真实生产全量仍 pending。本次交付是工具修补与有限身份盘点，不是 MK2100 / Qasper200 科学验收。** 未发布部分质量均值、CI、3pp判定，未写科学最终报告或完成标记。

用户授权的独立会话为 `independent_review_20261030T0343`，非生产主控，也不冒用合同里的历史 verifier_actor。PROJECT_ROOT=/home/wangyuanshuo02/two-level-attention；WORKFLOW_ROOT=/home/wangyuanshuo02/agent-research-workflows。所有新增文件仅在本 PROMPT 所在目录。exec 均 login=false，CPU Python 为 PROJECT_ROOT/.venv/bin/python -B，CUDA_VISIBLE_DEVICES 为空。未派 GPU、监督器、kill、SSH、修改生产源/原结果/guide/任务。工作从约03:43 UTC开始，18分钟预算内结束。

## 依据与复用决定

使用 gpu-experiment-dispatch 技能，读取文档治理、结果复用、结果独验契约和执行规范/作战手册适用段落；只读项目 GUIDE、任务索引及 TL-FROZEN-V1-16-20261030 / TL-FROZEN-QASPER-MK2-20261030 两任务详情。读取 V1 round_20261030T031152/closeout_requirements、专项 run_contract/validation_plan、0243检查脚本及报告。

提示中的 V1 `independent_review_20261030T0313/REPORT_DETAIL.md` 不在 run 直属目录；实际找到并读取的是 **ai_supervision/independent_review_20261030T0313/REPORT_DETAIL.md**。context 和 CONTEXT_BINDINGS.json 固定以上关键材料及冻结生产 validator 源码。

复用决定为 **verify_delta**：旧9条MK2审计及旧CPU边界测试仅作背景，不转授给新增样本；本轮针对已确认的最终文件漏检补测。未读取混合旧full5/3e4a688 Qasper结果，未重复大规模tokenization。代码依据是冻结实现及固定统计合同，没有用文献代替实验。只读核对MK2工作树HEAD为de2a1c1670b82bebcde6c0c3aa02fb8ee1431568，tracked状态空；V1 frozen-main HEAD为08ce44d6027a235080e7ff4127754af6b1ae91e0。本轮没有重新认证全部源码/权重；未来完整入口仍保留旧脚本逐源码/输入检查。

## 修补内容及准确边界

旧 check_independent.py 满100时调用副本 summarize，并比较独立统计与新生成副本。它没有读生产最终 summary/validation/report，无法检出这些最终文件陈旧或被改坏。此为验收工具缺口，不能据此断言真实GPU或质量数据错误。

新增 `closeout_supplement.py`：

- 消费成功的完整独立snapshot，先要求100个预定ID、300臂、输入重建和统计检查记录；旧9条snapshot实际被拒绝。
- 保存生产 manifest、全部原始请求、summary.json、validation.json、GENERATED.json、报告的原始字节及来源SHA；核对请求与独立snapshot逐字节一致。核对旧snapshot所有 CAPTURE_BINDINGS 的本地副本与原路径，结束时再次检查依赖与本轮捕获文件。无原子目录快照保证，但变化会拒绝，不能把逐文件读取称为同一瞬间采集。
- 独立逐次重算10000次seed42配对bootstrap，连续RNG顺序固定 Dual−Far、Far−Full、Dual−Full；完整字段集合核对，不用zip接受空CI。均值/CI为pp或0–100，数值允许1e-10绝对舍入容差，整数/布尔/键集合精确检查，拒绝NaN/Inf。
- 从逐层decisions累计gate四状态及Near/Far触发请求，检查保存gate_counts。核对升/降/同分、同generated_ids。它不独立复算Q/K；原始输入、decode/score、strict gate、cache几何依赖先运行完整独立脚本，不能单独拿补充器代替这些检查。
- **生产**summary逐字段对照独立参考与snapshot summary；**生产**validation逐字段对照完整独立validator结果；生产GENERATED要求identity及100样本。报告固定字节，核对其结构化事实块，绑定原始集合和validation哈希，包含主比较、完整summary、3pp点估计和限制。
- 报告需且仅需一个 `json acceptance-mk2` fenced块（格式见下文）。没有该块会拒绝，不能把任意Markdown关键词视为验收。自由文本中的数值/推论、图表及两任务之间的口径仍须独立人工/agent审读；机器成功始终返回 `full_acceptance=pending` 和 `report_prose_review=required`，不自动认证整篇报告。

新增 `check_independent_full.py` 是旧脚本在本新目录的副本，显式保留 **RUN=BASE.parent**，未改变原10000次统计/门限。新增前置100完整行及生产文件存在门，避免部分数据重tokenize；原完整CPU链结束、CAPTURE_BINDINGS落盘后自动调用补充器。旧脚本和原结果没有修改。

## 实际执行与日志

`CPU_TESTS.log`：最终 **11个unittest测试方法通过**，包含多个明确命名的反例子检查。合成夹具全部标在 SYNTHETIC_NOT_QUALITY，不是GPU质量数据。通过范围：

- 从冻结生产源AST仅提取真实summarize函数作为被测对象，非退化100条三臂合成记录对照独立逐次10000抽样，三组CI/均值/计数一致。
- 实际走生产文件读取/复制/对照路径的正例通过，仍pending；改坏生产summary的CI、validation臂数、报告bootstrap，及生产raw字节，各自实际拒绝。
- 空CI、NaN、缺/多字段、错均分、错触发、错gate、错degraded/同token数拒绝；改用5000次或seed43生成的统计CI也拒绝。
- 报告seed、B、主方向、raw绑定、3pp描述、限制缺失，以及缺失/重复事实块拒绝。
- −3pp边界及两侧按点估计判断；部分记录统计拒绝、非有限score、raw gate计数损坏和捕获后字节漂移拒绝。
- 真实旧9条snapshot拒绝，未为此重做tokenization。

`FULL_PREFLIGHT.log`：对真实当前生产运行新增完整入口，**退出2**，明确100完整请求不足，未执行tokenization或统计。`INVENTORY.log`：身份/行数盘点**退出2（预定pending）**。测试日志退出0不是生产实验通过。

固定 inventory 位于 `audit_20261030T034744805200Z`：MK2 **39/100**，Qasper current **64/200**。保存各完整请求原字节，核对manifest identity/预期源commit、三臂集合、唯一ID、选行集合归属以及读取稳定性。missing_ids逐项在RESULT.json中。该状态属于采集时刻，生产继续增长；只支持身份/数量审计，**没有新增39条或64条的输入/评分/cache验收**，没有部分质量结论。未另行盘点V1全部3750行，不能把用户给出的约1283当本会话实测。

## Qasper current200：明确未闭合项

现有 `--inventory` 是当前来源的只读准备入口，不是Qasper验收器；即使将来齐200也保持pending。64/200缺136行，不能进行科学结论。不能运行旧qasper_reuse_check.py来代替：它混读历史与current且会计算部分质量均值，并写专项父目录，不在本轮授权范围。

完整后仍需独立实现/运行以下专项适配，当前没有冒称已实现：

1. 固定current V1 manifest身份628463dbc68fcb474cd9c77b48f553d832c23983795748689b9748f63575f7e2、08ce44d源及selected_rows完整200；所有完成记录、input/config/实际评分依赖/model环境须绑定，不能使用3e4a688旧结果或仅依赖rank0盘点。
2. 按V1 Qasper prompt、31500 cap截断/chat及停止协议重建200输入；独立decode和Qasper F1重评分、answers对应、strict .05独立门控、三臂Far一致性、缓存/回执核验。V1全量recheck可提供机制参考，但它默认16任务和5000统计，不能直接宣称已适配专项。
3. 三臂完整200配对后，专项另算10000 seed42，顺序Dual−Far、Far−Full、Dual−Full；主Dual−Far，报告升降/同分/同token、Near/Far/Both触发。**不得把V1 5000 CI拷贝成专项10000结果，也不改变V1原统计。** 本补充器statistics可复用作独立参考，但没有因此完成Qasper输入/评分验收。
4. 对生产QASPER_CURRENT_REPORT.md及专项最终FINAL_REPORT_ZH.md进行实际数据绑定/数值/自由文本核对，3pp仅工程点估计门槛。MK2报告块schema明确dataset=niah_multikey_2，不能用它冒充Qasper报告验收。
5. 补齐符合experiment-result/v1或明确共享盘适配的真实artifact清单与六项独验记录。当前专项run_contract仍是计划/启动期材料，不是已完成结果manifest。

## 后续执行命令与报告接口

以下命令仅写本审查目录内的新snapshot/audit，生产读取保持只读。若在未来新独验目录运行，必须成对复制两脚本并重新核准RUN：新目录必须是专项run的直接子目录；不同位置必须显式适配。新授权和独立身份仍由宿主确认，不能把本报告转授为未来实验通过。

```bash
# CPU回归（合成程序测试）
CUDA_VISIBLE_DEVICES='' /home/wangyuanshuo02/two-level-attention/.venv/bin/python -B /home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_qasper_mk2_64k_20261030/independent_review_20261030T0343/test_supplement.py

# 只盘点，退出2始终表示pending，不算质量
CUDA_VISIBLE_DEVICES='' /home/wangyuanshuo02/two-level-attention/.venv/bin/python -B /home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_qasper_mk2_64k_20261030/independent_review_20261030T0343/closeout_supplement.py --inventory

# MK2齐100、生产机械summary/validation/GENERATED及报告已生成后：
CUDA_VISIBLE_DEVICES='' /home/wangyuanshuo02/two-level-attention/.venv/bin/python -B /home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_qasper_mk2_64k_20261030/independent_review_20261030T0343/check_independent_full.py --report /home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_qasper_mk2_64k_20261030/FINAL_REPORT_ZH.md

# 已有上述完整snapshot时，可仅重跑产物补充器，无需重复tokenize。
# 替换SNAPSHOT绝对路径为该次真正通过完整CPU检查的snapshot_...目录：
CUDA_VISIBLE_DEVICES='' /home/wangyuanshuo02/two-level-attention/.venv/bin/python -B /home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_qasper_mk2_64k_20261030/independent_review_20261030T0343/closeout_supplement.py --snapshot /ABSOLUTE/SNAPSHOT --report /home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_qasper_mk2_64k_20261030/FINAL_REPORT_ZH.md
```

生产报告正文中嵌入一个以 ` ```json acceptance-mk2 ` 开头、下一行为JSON的代码块。JSON必须等于该次补充器生成的 `audit_.../EXPECTED_REPORT_FACTS.json`（仅100真实数据通过后才生成；此轮只有合成例）。它绑定summary完整字段、原始集合hash、validation文件hash、bootstrap/seed/比较顺序、主比较、三个3pp点估计判断和限制。首次缺块会保留EXPECTED_REPORT_FACTS并拒绝；由生产报告作者核对并纳入后，重跑补充器，仍需独立审阅块外正文。此接口只规定报告绑定方式，不改生产统计算法、样本、gate或阈值。

## 六项状态与未验收范围

|项|本次已检范围|真实全量状态|
|---|---|---|
|implementation|旧漏检静态核对、新补充器及文件对照正反例|inconclusive；未跑真实100/200全链|
|reference_boundary|冻结summarize与独立逐次bootstrap、3pp边界|inconclusive；没有独立GPU参考|
|data_integrity|合成全量文件绑定、当前39/64身份盘点|inconclusive；缺61/136行及最终artifact合同|
|numerical_sanity|合成CI/计数/gate/报告错误拒绝|inconclusive；不发布真实部分统计|
|measurement_validity|保持配对/单位/固定统计顺序；未测性能|inconclusive；未完整验收真实样本|
|reproducibility|可运行CPU脚本、日志、稳定哈希|inconclusive；未独立GPU重生成|

六项均未用N/A绕过。真实Far/Both是否触发，本轮仅盘点身份未重新审计，不能沿用旧9条无触发推断全部当前样本，也不能把合成四状态覆盖当真实GPU触发。未独立GPU重生成；未保存Q/K，不能复算真实attention mass；metadata一致不保证每一步GPU张量正确。启动后补采模型/tokenizer哈希不能证明历史内存加载身份。本轮也未重hash全模型权重，没有速度/显存/等价/无损结论。

稳定证据由EVIDENCE_SHA256.json绑定，排除活session.log、最终CLI回复文件和清单自身。哈希验证的是字节，不授予科学验收。可使用范围仅为本次工具已测行为、缺口和固定身份盘点；实际生产全量仍pending。
