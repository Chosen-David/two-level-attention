# 独立 verify_experiment_result 详细验收

**整体结论：pending；不得消费为全量 usable-with-scope。** 本会话独立于生产主控，2026-10-30 02:13 UTC 起、20 分钟内完成。最新完整 CPU 快照在 02:20:16 UTC 检查结束：**401/3750 配对请求、1203/11250 臂**限定检查通过，errors=[]。三机对其中 LCC/repobench-p 共 264 臂实际预测的评分一致。没有发布部分 AVG16、3 pp、速度或总显存结论。

PROJECT_ROOT=`/home/wangyuanshuo02/two-level-attention`；run=`agent_doc/results/frozen_near_far_v1_16_independent005_20261030`。唯一写入范围为本报告所在 `ai_supervision/independent_review_20261030T0213/`（run 为用户既有共享盘符号链接）。所有 shell exec 均 login=false；Python 使用 -B。没有写 guide、任务、生产原始数据、冻结源码、队列，没有启动 GPU、代理、监督器，没有领取 lease 或操作生产进程。SSH 使用获授权三机地址/解释器、8416、sshpass 文件及 StrictHostKeyChecking=no；UserKnownHostsFile=/dev/null，未输出密码、未修改 known_hosts。

## 一、身份、读取范围与复用裁决

已读取用户指定 README、dispatch_plan、validation_plan、NEXT_STEPS、closeout_requirements、上次 REPORT_DETAIL 与原 recheck.py；相关字节保存于 `context/`。使用 code-reading 技能及项目文档、结果复用、独立验证契约；只读查看当前项目 GUIDE/TASK，当前用户的窄权限优先。

冻结源 `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/tla/frozen005_v1_16_source_20261030` 和 frozen-main worktree 的实际 HEAD 均为 `08ce44d6027a235080e7ff4127754af6b1ae91e0`，tracked status 为空（未检查 untracked），见 `git_identity.json`。协议身份均为 `628463dbc68fcb474cd9c77b48f553d832c23983795748689b9748f63575f7e2`。queue 的 `251569fd350a311b99bfd54f43aafd82345e06c9145c8a91ee7f276558f29a7d` 是 dispatch_plan 的独立哈希，已重算；它不同于实验 protocol digest 是两种对象，不是混协议证据。

重新核对 protocol 的 81 个源码文件、16 个本地数据集 SHA 与全 3750 行计划。三机只读独立补核在 02:17:39 UTC 各返回 97 项源码/数据 SHA 匹配、errors=[]；NumPy 2.0.2/2.4.6 对 seed=42 的所有 16 数据集 permutation 均等于 manifest.selected_rows。源码复制于 `source/`，远端证据见 `pc*_identity.json`、`remote_identity_probe.py`。这些是当前磁盘交叉核验，不能证明历史进程内存中加载的内容。

有界检索当前项目 frozen/near_far 结果目录，发现旧 full5、compute_audit、glm 和本 run。裁决为 **verify_delta**：沿用本 run 上次 145 请求作为比较对象，但重新核验，不复制旧 pass；旧 AVG5、smoke、GLM 结果均不混入新计数。最新 401 中包含旧 145，145 份字节均一致，新增 256 条；不是 401+145。旧 GPU/core 合成验收仅为背景，本次未重新验其全部历史 artifact，不用于本轮 reference_boundary 授信。证据：`context/prior_cpu_checks.json`、`additional_checks.json`、`final_snapshot_checks.json`。

实际使用安装的 model-with-knowledge 文件检索工具查询 paired/bootstrap/reproducibility/scoring/difflib，见 `knowledge_search.json`。词法命中的双线性风险等条目与本题评分/快照验收不适用，reject，不引用其定理、不冒充文献支持；knowledge_refs=[]。本轮依据为实际冻结源、数据和执行输出。

## 二、实际执行与覆盖时间边界

直接执行原 `independent_review_20261030T0142/recheck.py`，没有修改它或复制改写后冒充原脚本：

```bash
python -B ai_supervision/independent_review_20261030T0142/recheck.py \
  --run . \
  --source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/tla/frozen005_v1_16_source_20261030 \
  --output ai_supervision/independent_review_20261030T0213/cpu_current
```

cwd 为 run。首次退出 0，02:15:18 UTC，350 请求/1050 臂、16 manifests；日志 `recheck.log`。第二次同命令只将输出改为 `cpu_final_snapshot`，退出 0，02:20:16 UTC，401 请求/1203 臂、16 manifests；日志 `recheck_final_snapshot.log`。两个目录均保留逐文件原始字节快照，没有覆盖、未使用 --no-snapshot，**没有执行真实全量 --require-full**。本轮两次原 recheck 无失败重试。

|任务|最新 CPU 已验配对数|全量分母|
|---|---:|---:|
|narrativeqa|32|200|
|qasper|32|200|
|multifieldqa_en|16|150|
|hotpotqa|16|200|
|2wikimqa|32|200|
|musique|16|200|
|gov_report|32|200|
|qmsum|30|200|
|multi_news|16|200|
|trec|25|200|
|triviaqa|11|200|
|samsum|22|200|
|passage_count|11|200|
|passage_retrieval_en|22|200|
|lcc|44|500|
|repobench-p|44|500|

所有 16 任务已有实际样本仍不等于全量。快照是在运行期间逐文件读取，不是整个目录原子快照；各完成记录字节绑定稳定，新的产出不属于已验范围。首/末核对已纳入文件的 SHA 均未变化，见 `additional_checks.json`、`final_snapshot_checks.json`。没有将仍在写入的 running 记录视为完成。

**done 唯一覆盖：** `supplement.py` 于 02:15:55 UTC 固定 queue、manifest、done 和对应 raw：8 个 done rank 0–7，每片 30 行，240 唯一键，identity/PID/计划归属/三臂完整/路径一致全部通过；每片有一个生产 native_full_parity=true 记录。rank 来源按生产 `shard_inputs` 的输入串联次序模 128 独立重建，不以文件总数代替键覆盖。

随后 `refresh_coverage.py` 于 02:19:34 UTC 读到 **11 份 done 回执（0–10）及 330 个唯一完整请求**；这些回执的 identity/PID 和 raw 覆盖一致，330 行全部包含于最新 401 CPU 快照。但该次 queue 仍为 8 done/8 running/112 pending，因此保留 `queue done 8/9/10` 与 `queue done set` 四项错误，**这次整体 coverage 不能标 all-pass**。02:20:58 的 final_queue_snapshot 仍为 8/8/112。源码 remote_frozen_queue.py:160–186、241–275 显示 daemon 在每 300 秒 tick 才 reconcile 回执，故这是可由对账滞后解释的当前状态差异，不据此断言重复生成或丢数据；本次未操作进程或强制同步。后续需等生产自然对账后重新固定快照验收。原失败/不一致证据保留于 `coverage_refresh/`。

## 三、源码调用链、评分与门控缓存

以下均对应上述 commit，文件字节在 `source/`：

- `prefill_frozen_skip.py:193–198,236–263,288–321`：按 16 任务 selected_rows 串联分片，身份绑定配置/输入/源；三臂完成才写 complete，整个 shard 完成才写 done。`65–79` 是模板、31500 cap 的首尾截断、再分词和指定 no-chat5；`35–47,133–140` 是 greedy、任务 EOS、SAMSum newline/min1。已从原数据和 tokenizer 重建实际输入 token SHA、长度、答案，检查输出 token 解码、终止规则、三臂首 token。
- `prefill_frozen_skip.py:305–306` 把 repobench-p 映射 repobench，进入 `eval.py:264–272 sample_score`，再按 `dataset2metric` 调用 `metrics.py:109–116 code_sim_score`。代码任务取首个不含反引号、#、// 的行，与各答案计算 difflib 相似度，round(100*ratio)/100 后取最大。不是 Levenshtein；daemon main 显式设 TLI_SCORER_BACKEND=difflib（remote_frozen_queue.py:243）。SSH 新解释器的环境字段为 null，不能冒充实际 worker 环境；生产源码设定与实际保存评分一致共同构成限定证据。
- `generate:102–154 → capture_profiles:83–97 → frozen_region_gate.profile_regions/decide_regions`：prefill 后逐层冻结，近远各自严格 `<0.05`，不是两者相加的预算。qwen3_attn_patch.py:297–312 在 post-RoPE 后按 request cache 路由，decode 先走 region_decode 并返回，避免通用 cache 再追加。
- `frozen_region_cache.py:114–133,135–202`：far/both 真正删除相应节点；near-only 保留不可见 staging，经 SWA→near→far 正常迁移；gather 仅取 visible。`frozen_region_tli.py:28–82` 使用实际 TLI compute_score/compute_mask，最终 mask 投影回可见位置。full 是 **TLI gate-off，不是 dense FullKV**。

原 recheck 已检查所有已验臂逐层 profile 有限性/跨臂一致、独立 gate、节点几何、容量字节、可见性、token 守恒和 TLI receipts。补充 `final_snapshot_check.py` 再检查原脚本未逐项覆盖的 linked_order、attention_order、state skip flags、receipt skip flags、stored_tokens 节点和、near-only 冻结前后驻留 KV 字节相等：errors=[]。

401×3×36=43308 个层状态：keep 40423、far-only 2833、near-only 51、both 1；near-only far 增长 3 条层记录。这些是日志内部一致性/迁移记录检查，**实际 Q/K 张量未保存，未重新计算真实 attention mass，未重跑 GPU KV 数值或生成**。没有根据节点计数推出加速或总显存节约。

## 四、Python/difflib 实际评分差异核验

第一批实时快照 66 个代码请求（LCC34、repobench32）198 臂，三机逐项 ratio、score、stored 均一致。随后为避免仅覆盖旧样本，将最新已验快照的 **LCC44 + repobench-p44，共 264 臂**传入三机指定 Python，再次全部一致（`code_score_final/comparison.json`）。每机输出包含每个样本、每个答案的未舍入 ratio、最终 score 及 stored 对照，执行退出码均 0。

`remote_score_probe.py` 只 AST 提取实际冻结 metrics.py 中 difflib `_fuzz_ratio` 和 `code_sim_score` 函数执行，不导入生成器或 GPU；输入 prediction/answers 来自绑定 raw，另行按相同明确规则计算未舍入 ratio。该跨版本核对属于同一真实评分函数的独立调用；原 recheck 另有独立实现，不把此调用称为从头实现第二套算法。

|环境|Python|difflib SHA-256|
|---|---|---|
|pc02、pc40|3.9.9|5ecbbaaa838dbb2fdafc41280a281eeb7deeae05d889c31ebbf0d18d4cf66c3d|
|pc31|3.11.12|0c6afc23568d55b3e9ac914f9c5361e3033e778aa5b58d3cc82835fc5c638679|

完整 difflib 源码已保留在远端 JSON。`difflib_39_311.diff` 显示唯一变化是文档中 writeline()→writelines()；去除 docstring 后 AST 完全相同（additional_checks）。因此，**在本轮实际 264 臂代码评分中未造成不一致，文件 SHA 不同源于该文档差异**。这不证明所有 Python/runtime 或跨 GPU 生成数值路径等价。ROUGE 仍调用留存的 Rouge 1.0.1 参考库，属于独立调用同库，非新写的 ROUGE 实现。

## 五、六项契约检查

|检查|本轮实际证据|全量 verdict|
|---|---|---|
|implementation|冻结 HEAD/81文件、调用链及401请求记录重算；补充缓存顺序/flags通过；export 快照绑定缺口仍在|inconclusive：限定静态/CPU支持，未覆盖全量和全部动态依赖|
|reference_boundary|独立输入/评分公式重建；264代码臂三机交叉检查；done分片原生对拍回执可追到对应行|inconclusive：未新增真实GPU或四状态KV值边界复跑；生产 parity=true不是本会话GPU复现|
|data_integrity|401唯一完成配对/1203臂通过；旧145字节不变；8队列done=240唯一行；11回执=330唯一行均已CPU验|inconclusive：3750/11250/128未齐，新增3回执与queue尚未对齐|
|numerical_sanity|评分有限/重算、gate/cache/字节守恒、264代码臂ratio和分数对照通过|inconclusive：真实mass未从Q/K重算，全量汇总/导出/CI未验|
|measurement_validity|质量配对同输入/生成限制、各任务分母明确；源码采用任务等权而非3750 pooled mean|inconclusive：全量逐任务与AVG16及3pp判定待做；无性能主张，不以N/A代替质量验收|
|reproducibility|两次原recheck运行成功；三机实际源码/数据/permutation/代码评分核验成功|inconclusive：无独立GPU重生成；权重启动前身份与环境实现依赖仍不完整|

详细机器记录见 `verification.json`；它是本会话的独立审核记录，不冒充已经通过通用 experiment-validation/v1 接口或由生产者自证独立身份。当前实际 source/input/config/environment/raw 绑定见 CPU artifact_hashes、远端收据和本目录总哈希清单。

## 六、修复重测裁决与收官脚本缺口

1. **未发现这批已验预测需要因评分差异而重新 GPU 生成的证据。** Python/difflib 跨版本疑点在已验264臂范围关闭；最终全部代码任务仍须再核。队列未对齐目前归为 pending，不改队列、不删回执。若自然 reconcile 后仍不一致，应先诊断再补验；不能改状态 JSON 假装成功。
2. **adopt：export 陈旧 summary 风险未关闭。** `export_frozen_skip.py:30–33,63–79` 只比较完成状态/协议身份，然后重读 raw；协议哈希没有绑定生成内容。此次重新阅读确认缺口，未重新运行上次合成反例，也未证明真实 export 已遭污染。必须全量静止后固定 raw/manifest/done/source/input/config/environment，validate→analyze→export→独验前后 SHA 不变，且核对派生值来自同一快照。若检出陈旧汇总，重算分析/导出并复验；若是生成路径缺陷才版本化修复并重测受影响生成。
3. **adopt：旧 recheck.py --require-full 不足以单独收官。** 它验均分/AVG16及部分导出绑定，但未独立复算配对 CI、degraded/improved、逐任务与AVG16 3pp，也未充分绑定所有读取阶段；不要将退出0当作全量科学验收。已在本目录准备 `closeout_check.py`：先拒绝非128done/3750/11250唯一覆盖，完整后才重算三组配对差/5000次分层bootstrap/CI、3pp点估计判定，核raw_index与predictions并前后哈希。当前实跑 `closeout_current/` 返回2、pending_or_failed，未输出任何部分质量数值。缺失done/运行中记录是预期未齐证据，不是生产损坏证明。**完整分支尚未用真实全量执行，脚本存在不是验证完成；它仍需原 recheck 的全量输入/评分/gate/cache验证，并需最终TSV/报告与完整artifact适配检查。**
4. **adopt：partial validator 与合同结构缺口仍在。** `validate_frozen_skip.py:38` 在partial处理前要求全部manifest；本次未运行生产validator，避免越权写validation.json。`run_contract.manifest_path` 仍指派单计划，scope文本也不等同validation_plan，不是完整 experiment-result/v1 artifact manifest。生产主控需给真实清单或明确项目适配，将统一scope、原始哈希和六项检查绑定；本报告不能自行补授权或宣布通用合同通过。
5. **adapt：权重/环境后采证据只能限定复现范围。** 上次报告的三机13模型文件补采一致，不能追认启动前身份；本轮未重哈希16GB权重。已补成真正独立SSH源/输入/评分核查，解决上次SSH失败的部分缺口，仍不证明实际worker的历史内存权重、CUDA数学路径、所有依赖实现完全相同。NumPy选行排列一致不自动证明最终bootstrap实现/随机流已跨版本等价，最终CI须核固定分析环境。

最终仍需128唯一manifest/done、3750完整配对/11250臂、全量独验、同快照分析导出、全部16任务和AVG16的配对差/CI与3pp判定、最终报告及明确限制。合法质量下降应保留，不因不满足3pp就删除数据；CI含0不证明等价。未写 FINISHED、COMPLETE 或 FINAL_REPORT_ZH.md。

## 七、证据交付与限制

`EVIDENCE_SHA256.json` 对本目录全部稳定有效证据、脚本、两个CPU快照、远端输出、上下文和本详细报告逐项绑定，并列出最终CPU检查覆盖的外部原始字节哈希。清单不自包含，排除仍变化的 session.log、CLI -o FINAL_RESPONSE.md。`FINAL_RESPONSE.md` 只承载最终简答，不覆盖本文件。哈希只证明内容绑定，不证明语义正确。

本轮未启动模型生成、GPU、代理或监督器；未复现实际Q/K attention mass、所有GPU/cache值路径或最终全量CI；未宣称无bug、无损、加速、总显存改善或整个实验通过。整体保持 **pending**。
