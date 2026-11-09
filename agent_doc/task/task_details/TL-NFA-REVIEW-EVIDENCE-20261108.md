# TL-NFA-REVIEW-EVIDENCE-20261108

## 目标与串行顺序

在 `/tmp/nfa-evidence-release-20261108` 隔离 worktree 完成 NFA 审稿证据补测：M1 → Q0 → A1a → P0a/P0b。Q0 已独立验收；用户 2026-11-09 最新指令将 fused-LSE 标定器修复与三长度真实 prefill 测量作为优先子链，完成后先行安全推送，其余任务继续。

生产性能口径为 tail32、page64；FlashInfer 同样使用 page64。page16 仅限 M1 等明确标注的正确性证据。正式性能必须独占锁卡、同输入 ABBA、独立进程、3 warmup、512 tokens、5 固定输入×3 重复，并保存完整 token 与 30 秒旁路监控。

性能自查不变量：tail32 应优于 full 维索引，skip 应优于 no-skip，NFA 应优于 FlashInfer；若不满足，先排查实现、配置语义、锁卡、GPU hygiene、warmup 和 ABBA，不充分排查不得直接接受。

## Progress

### M1（完成）

- `agent_doc/results/nfa_selection_trace_20261108/` 已导出 RULER 32K `niah_single_1` sample0 首 eager decode step、层 0/17/35、page16 的真实 L1/L2/Gather 轨迹。
- 独立校验 `NFA_SELECTION_TRACE_VALIDATION_PASS`，范围仅限首步三层，不外推。
- 本地提交 `72df36220a91393f6fbb0a4b1c91703ec4a6860d`；相对 `c4edd7f5` 的 production paths diff 为空。尚未推送。

### Q0（完成，2026-11-09 收尾）

- 证据包：`agent_doc/results/nfa_q0_execution_20261108/`。
- dense/seed-on/strict-skip 各 500 条；HotpotQA 200、GovReport 200、RULER 64K MultiKey-2 100。
- strict LongBench 的原前缀、旧分片、resume 和 20 个 A/B 分片经合并器检查：400 条、41 producers、无重无漏。
- 三臂完整汇总：1500 quality rows、36000 gate rows；独立验收 `status=PASS`，输入/producer/source/token/gate/raw SHA/官方评分全部复核。
- 质量均值：HotpotQA 59.4162/61.5140/61.6377；GovReport 33.5915/33.5633/33.4856；RULER 0/73/76（dense/seed/strict）。dense RULER 0 分来自真实重复问题输出，不是 scorer 漏算；只作本输入观测。
- token 完全一致数：dense-seed 174/500、dense-strict 172/500、seed-strict 310/500；不得写成三臂 bitexact。
- strict RULER 100/100 首步回执有效；NFA 每请求 36 层 decisions 完整。
- 慢质量 worker 由 py-spy/cProfile/同输入对照定位为每请求 Graph capture 的 Triton JIT；共享 wall 不进入性能结论。
- raw 已压缩为 `archives/q0_raw_1500.tar.gz` 并通过 tar 校验；`data_manifest.json` 登记 raw+summary 2508 文件及 SHA。

### A1a（执行中：84d5 eager replay 修复已闭环，正式三新格生成/补格中）

- 四格：full/tail × no-skip/natural-skip，使用 Q0 同一 500 冻结诊断输入。
- tail/natural 在配置与输入身份复核后复用 Q0 strict-skip；其余三格补生成。
- no-skip 必须显式 `gate_epsilon=0`；natural-skip 保存真实 keep/near/far/both。full 是 NFA 全维索引，不是 dense FullKV。
- **已定位并修复 full 维选择链 bug**：full-L1 的 Q 为 128 维，但页统计仍错误使用 tail32；L2 kernel 也无条件复用 L1 维索引，导致 full+sub reshape 失败、tail+full 的 fine 维度错误。修复分离 `l1_dims` 与 `fine_dims`，不改打分、Top-K、gate 或预算；tail/tail 路径不新增 full `arange`。
- 隔离源码 `/tmp/nfa-a1a-source-20261108` 本地提交 `f8ec4b522997bad36d36892c2d1c7cb3eeed5fb1`（`Fix full-dimensional NFA selector indexing`），仅修改 selector 与对应测试，尚未推送。
- 新真实 eager decode 回归覆盖 full+sub（L1=128/L2=32）、tail+tail（32/32）、tail+full（32/128）的页统计、Q/K 维索引与反事实最终选择；修复前新增用例 2 failed/1 passed，修复后定向 5 passed，完整 `test_nfa_current_query_selector.py` 为 **43 passed**。
- 六个 dirty-source 调试 smoke 均通过：full/natural、full/no-skip、tail/no-skip 的 RULER 与 LongBench；RULER 均生成 128 tokens 且首步 strict 回执有效，LongBench smoke 均为 HotpotQA sample0、score=1.0、5 tokens。两个 no-skip 臂在两数据集均为 36/36 `keep`，page64/epsilon/selector 配置正确。
- tail/natural 修复后同输入对拍 Q0 strict RULER sample0：token、prediction、36 层 gate 均一致，支持继续复用 Q0 tail/natural。调试 smoke 的 source diff 非空，不得混入正式 raw。
- clean `f8ec4b52` committed-source full/natural 与 full/no-skip RULER smoke 均完成：各 128 tokens、首步回执有效，`source_diff_sha256` 为空 diff SHA，显式记录 `l2_subspace=same`、page64；no-skip 运行时四态硬门通过。
- 正式生成仍需逐机确定性分片，保存逐样本 token、官方分数、manifest；最终汇总目标为 2000 quality rows、72000 gate rows，并由独立验收器复核。
- **正式三新格已派发**：`formal/dispatch_plan.json` 固定 pc33×2、pc40×2、pc02×4、pc31×2 共 10 个 worker、60 个 job；每 worker 对每臂负责 40 条 LongBench + 10 条 RULER，按冻结 target 确定性互斥分片，三新臂各覆盖 500 条、无重无漏。四机源码均为 clean f8，模型/tokenizer/RULER 数据 SHA 与 committed smoke 完全一致。
- `formal/dispatch_receipt.json` 登记十条父链精确 PID 和首轮进程/显存/log 三重验证；所有链存活，pc33 已有多条 `strict_valid=True` 记录。当前均为共享卡质量运行，wall 严禁进入性能结论。
- 首轮 pc40×2、pc02×4 的 full/natural LongBench job 在 import 阶段返回 1：缺 `jieba`，未生成记录、未进入模型生成。两机已安装并验证 `jieba`/`rouge`；这些分片待当前同卡链结束后幂等补跑，不与在跑 RULER 重叠。其余 RULER 与 pc33/pc31 LongBench 正常产出。
- **真实长生成进一步定位并修复 full eager replay bug**：full 非融合路径把独立 `paged_write` 录进 eager call-plan，但旧 replay 只刷新融合写入及当前 Q，没有刷新该独立 launch 的 K/V；从第二个 decode step 起会把首步旧 K/V 写入当前 slot。逐步真实轨迹显示第 1 步 36 层 Q/gather/lens/attention 全部逐位一致，第 2 步 layer0 Q/gather/lens 仍一致而 attention 已分叉，排除仅由 L1/L2 打分或 Top-K 选错解释。
- 修复提交 `84d5a2eb1f6987cdc807f547abbacab018109247`（`Fix eager replay decode KV refresh`）：在 `_replay_eager_selector_call_plan()` 内将独立 `paged_write` 的参数替换为当前 K/V/slot；不改 tail 融合路径、Graph capture/replay、选择算法或 gate。三条维度链定向回归修复前 full-same/tail-full 均在 step1 K/V 断言失败、tail-same 通过；修复后定向 3 passed，完整 selector 回归 **46 passed**。
- 真实对拍已闭环：GovReport sample0 修复后 eager 与正式 full Graph 的 **464 tokens、prediction、decisions、profiles、receipts、gate counts、score 全部一致**，仅 wall 字段不同；修复前 eager 为 512 tokens、score=0 的重复输出。clean 84d5 RULER sample0 恢复正确答案 `6073160`，128 tokens、strict_valid=True，旧 eager raw 为重复字符/数字串。机器可读回执为 `diagnostics/eager_replay_fix_validation.json`，状态 PASS。
- **旧 raw 隔离与补格**：16 个已完成的 clean f8 full eager RULER producer（full/natural 10 个、pc40/pc02 的 full/no-skip 6 个，共 160 条）已保留原始字节、逐目录树哈希并移至 `formal/quarantine/f8_eager_replay_invalid/`，不得进入正式质量汇总。四机源码已 fast-forward 到 clean 84d5 并 compile/import 冒烟；本机及 pc40/pc02/pc31 的完整 selector 回归均 **46 passed**（远端 pytest 隔离安装在 `/tmp/nfa-a1a-pytest-deps`，不修改生产环境）；当前已加载 f8 的 Graph LongBench 进程未被打断。
- 十条 `recovery_worker.sh` 等待链已登记精确父 PID；它们只在各自原父链自然退出后运行。pc33/pc31 补 full/natural RULER，pc40/pc02 另补 full/no-skip RULER 和缺 `jieba` 的 full/natural LongBench；点火核验确认恢复链均处于 waiting、原 Graph 任务及显存占用未受扰动。恢复任务运行前硬门要求 HEAD=84d5 且 diff 为空。
- 三远端 raw/logs/quarantine 已增量收割回本机证据树。四机原始 quarantine 回执分别保存为 `receipts.<host>.jsonl`，避免同名汇总互相覆盖；`validate_quarantine_harvest.py` 已对 16 个目录、160 条记录逐文件重算树哈希并核验 manifest/worker_done/identity，结果 `validation.json status=PASS`，合并回执 SHA256=`2a18b312447e96a71f805ce04f504fc30506c21866e42d7dd71975b7b87b7ccb`。
- pc31/gpu0 的 full/natural LongBench 已完整完成 40/40，随后原链自然启动 full/no-skip RULER；其 manifest 为 clean 84d5 eager，进程/显存/log 三重核验通过，已落盘样本均为 128 tokens、strict_valid=True、36/36 prefill keep 且 decode near/far/both 全 0。最终数量仍以完整汇总为准。
- 最终 `validate_a1a.py` 新增两道独立硬门：正式 eager RULER 只接受 clean 84d5，任何 f8 eager 即使误放回 parts 也拒绝；两个 no-skip 臂除 36 层 prefill decisions 外，还必须逐请求 decode `skip_policy_counts` 的 near/far/both 全 0。正/反例检查通过。
- 2026-11-09 新一轮收割后，pc31/gpu0 原链与恢复链均已自然完成；其 full/natural、full/no-skip、tail/no-skip 的 LongBench+RULER 六 producer 全部独立复核 PASS（300 条记录、输入/官方分数/token/36 层 gate/strict 与 no-skip decode 回执均通过）。pc33/gpu1 的 full/natural LongBench、full/no-skip LongBench+RULER 三 producer 也全部 PASS；其中修复版 full/no-skip RULER 为 clean 84d5 eager、10/10、90 分、360/360 keep，decode near/far/both 全 0。pc33/gpu1 原链随后自然进入 tail/no-skip，未重复派发恢复链。
- Graph 路径不执行该 eager call-plan，且 GovReport sample0 完整对拍一致；但单样本不自动证明全部 Graph raw，正式复用仍须按输入、manifest、源码谱系及最终独立验收判定。A1a 目前仍为 pending，2000 quality/72000 gates 尚未闭环，共享 wall 不进入性能证据。

### Prefill fused-LSE 轻量标定器（完成，2026-11-09）

- 实现提交 `e8710c38`/`ed011215`：CUDA 支持形状使用单趟 online-softmax/LSE，同时维护 total/far/near 三组状态；不物化完整 scores/probabilities，原生 GQA 并支持生产非连续 Q/K strides。不支持维度或非 2 幂 GQA group 保留参考实现回退。
- 测量与验收提交 `c16639ac`/`976bd6ed`/`6d513caf`/`6fd22304`：真实 Qwen3-8B、RULER `niah_multikey_1` sample0，32K/64K/128K，dense/fused 各一次 warmup，三轮 ABBA；`torch.inference_mode()`、锁卡及 30 秒 hygiene 全部生效。测量源码 HEAD=`6fd22304`、空 diff SHA；后续 `dc2a929b` 仅新增回退测试，生产文件相对测量 HEAD 无差异。
- 正确性：相关回归 **57 passed**；每次 calibrated prefill 均为 36/36 fused kernel calls，warmup 36 层 profile/gate 与旧 chunked-softmax 完全一致，最大误差分别 `1.788e-7`/`2.384e-7`/`4.172e-7`，三长度全部 next-token identity。
- 配对增量中位数（增量/dense prefill）：32K **66.62 ms（0.928%）**，64K **115.71 ms（0.546%）**，128K **251.88 ms（0.358%）**；device 时间中位数分别 65.51/109.57/199.76 ms，占 dense prefill 0.913%/0.517%/0.284%。64K 首轮与 128K 第三轮 wall 有主机侧波动，保留原始三轮，不用 device 时间替代 wall 主结论。
- 证据包 `agent_doc/results/nfa_lse_calibrator_20261109/`：三份 v2 JSON、hygiene、ABBA 日志、逐文件 SHA manifest 与独立重算验收 `validation.json status=PASS`。逐层 host wait 包含前序 dense SDPA 等待，仅作诊断，不能表述为标定器开销。
- 边界：当前是**单趟 fused-LSE calibrator + 独立原生 dense SDPA**，没有复用主 SDPA 内部 LSE，也不声称主 prefill 已合并为同一 attention kernel；结果限当前模型、输入与几何，usable-with-scope。

### P0a/P0b（待执行）

- P0a：32K/64K/128K × B1/B4 × eager/Graph × NFA/FA2，共 24 方法单元，正式 5×3 覆盖。
- P0b：full/tail × no-skip/natural-skip 的 e2e skip benefit；双臂 page64，NFA vs FlashInfer 同样 page64。
- 质量任务共享卡不得冒充性能锁卡；Triton JIT 诊断与共享 wall 不进入正式 TPOT。

## 证据与边界

- 当前只完成 M1/Q0，不代表 A1a 或性能已验收。
- 原始记录保持只读归档；新配置写新目录，不与旧口径混池。
- 推送前检查远端 main 是否前进，安全整合他人工作，不覆盖、不强推。
