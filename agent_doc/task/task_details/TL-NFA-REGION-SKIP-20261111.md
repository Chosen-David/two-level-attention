# TL-NFA-REGION-SKIP-20261111

## 目标与验收口径

将旧版仅缩小部分注意力工作的 near/far gate 改为选择器真实区域跳过，并在修复后全量重测 NFA eager 与 CUDA Graph 性能。

- skip far：该层只执行 near 的页统计、L1 score/Top-K、candidate、L2 fine-score/Top-K；skip near 同理，只执行 far 链。
- both-skip：该层不进入页统计、L1/L2、candidate 或 ranking kernel，只保留显式 decode KV 写入与 protected-only union/attention。
- 单侧 skip 必须同时缩小实际 Top-K 处理宽度、partial tile 和 candidate 宽度，不能只将分数置为 `-inf` 或依赖空 CTA 早退。
- 以 nsys kernel 实例缺失证明 skipped region 入口未执行，以 `gridX/gridY/gridZ` 与 candidate 宽度证明实际缩容；eager/Graph 输出逐 token 一致。
- 正式矩阵为 Qwen3-8B × RULER `niah_multikey_2` sample2，32K/64K/128K native 输入与 256K/512K 性能扩展输入，B1/B2/B4，tail=32/32、full=128/128，skip/no-skip，eager/Graph。
- 32K 强制四态为验收覆盖手段；正式矩阵使用自然 prefill gate。no-skip 显式 epsilon=0，skip 为 epsilon=0.10。
- 算子与 clean-E2E 分进程测量；5 个 nsys profiled decode steps 与 32 个无 profiler decode steps 不相减推导 host overhead。FlashInfer eager operator profile 不存在，不伪造该臂。

## 实现

- `nfa_layer_batch_selector.py` 使用 `REGION=0/1/2` 编译期区域裁剪、独立 near/far L1 wrapper、紧凑 candidate 物理槽与原 full-pack global owner slot；mixed batch 保持通用 full-pack。
- far compact Top-K 只排序 far 前缀，再与唯一一份 canonical `-inf` 后缀归并；保留 score 降序、同分新 token 优先、NaN/±0/±inf、quota 补洞及最终逻辑 token 升序语义。
- region2 增量页统计补算最新页和 near→far 入域页，不重算 skipped near 区。
- both-skip 通过 `region_mode=3` 走 protected-only 路径；eager launch-plan 键包含 region，防止不同 wrapper/grid 错用快路。
- Graph key 为 `(batch_bucket, visible_bucket, skip_signature)`；capture 使用显式 frozen-signature override，replay 签名 miss 必须 eager fallback，重新 capture 后才能 replay 新图。

## 已完成验证

- 本地 32K B1 强制 no-skip/near-only/far-only/both-skip × eager/Graph 的 kernel 实例审计全部通过；both-skip 的 active page-stats/L1/candidate/fine-score 实例均为 0。
- Graph L1 score `gridX=5→2/4`，L1 partial `gridY=2→1`，candidate expand `gridY=9→3/7`，L2 partial `gridY=5→2/4`；eager 同样观察到 L1/L2 缩容。
- 四种强制模式的 clean eager/Graph token identity 全部通过。Graph 每格 capture=1、replay=8、fallback=0；eager 的逐步 fallback 计数是无图执行的正常口径。
- 最终综合聚焦回归：`tests/test_nfa_current_query_selector.py` + `tests/test_multi_request_graph_runner.py` + `tests/test_nfa_region_kernel_audit.py` + 三个 CUDA 专项文件共 **85 passed、54 subtests passed**。另通过 `git diff --check` 与生产文件 `py_compile`。
- 本地功能验收目录：`~/ak_verify_results/nfa_region_skip_acceptance_20261111/`。采集时本机存在不可见宿主 GPU 占用，因此本地耗时只作功能证据，不作为干净性能结论。

## 正式复测进度

- 基础提交：`84bd19ee947c2498db7c988a9fa2ca766caf1267`；生产与验收补丁在 pc31 隔离 worktree `/tmp/nfa-region-skip-20261111` 运行，不修改远端原工作区。
- 补丁：`/tmp/nfa_region_skip_20261111.patch`，当前 SHA256=`de0746fc8381d691fff6149c45cf6ef2cd1a063befcd4d2102c278c856bdca02`（已包含后续 eager host 修复；早期区域实现补丁哈希保留在下文来源记录中）。
- 原始证据：`~/ak_verify_results/nfa_region_skip_full_20261111/`；脚本 `/tmp/run_nfa_region_skip_full_20261111.sh`。
- 第一代 orchestrator PID=2128927 在 256K B2 Graph profile 的 nsys 汇总处 fail-closed 停止：实际 kernel 为每步 36 次（generic=6、far-only=30），旧审计却按两个请求重复期望 72 次。根因是审计器误按请求计数，而 selector 实际按层对整个 batch 发射一次；mixed layer 应统一计为一次 generic。生产 selector 未因此修改。
- 已修正审计器并新增 `tests/test_nfa_region_kernel_audit.py` 四项测试，本地 4 passed，pc31 无 pytest 环境下以直接 B2 调用冒烟 PASS。失败代完整保留在 `~/ak_verify_results/nfa_region_skip_full_20261111_pre_batched_audit_fix/`；由于 harness SHA 改变，旧回执不冒充新来源。新一代 orchestrator PID=2176792 已从 32K 强制四态重新运行。
- 新一代 16/16 强制四态回执再次全部 PASS，source diff SHA256=`0ab60bfca394f1c2f1c52cbf9ba31f06cf3aa56bbb05b3ac8da5dffd826ba53f`，harness SHA256=`8bfa80e38d88995fa8902fd00dd34ec86acdb726a757eda6ff7a1fe9b0299aef`。Graph 四态 operator/TPOT 分别为 no-skip 3.935706/10.315936、near-only 2.986498/9.292064、far-only 3.456438/9.776368、both-skip 0.467480/6.932288 ms；eager 分别为 4.040815/26.908608、2.953656/26.568528、3.514951/28.612832、0.477958/23.251504 ms。单次 eager TPOT 有波动，不与失败代做性能优劣判断。
- 计划为 16 个 32K 强制四态测量回执与 192 个正式 NFA 测量回执，共 208 个；每组完成后由同一驱动 `--summarize-only` 生成汇总。
- 首个 pc31 `acceptance_no-skip_graph` 已完成且 `gpu_hygiene=clean`：NFA 算子总耗时 3.9395544 ms/步，clean TPOT 10.3454080 ms；后续单元仍在运行，尚未形成全量结论。
- 第一代 pc31 四组强制 Graph 回执已通过当时的审计器预验，区域实例、grid 缩容、nsys 哈希、输入身份、GPU hygiene、Graph capture/fallback 和 profile/clean token 前缀均 PASS；这些数值保留作失败代对照，最终交付以新 harness 代为准：

  | 强制模式 | L1 ms/步 | L2 ms/步 | TopK ms/步 | 总算子 ms/步 | clean TPOT ms |
  |---|---:|---:|---:|---:|---:|
  | no-skip | 0.834372 | 2.088566 | 1.104480 | 3.939554 | 10.345408 |
  | near-only | 0.656285 | 1.312308 | 0.911172 | 2.989308 | 9.285760 |
  | far-only | 0.760127 | 1.679479 | 0.996087 | 3.462734 | 9.805328 |
  | both-skip | 0.000000 | 0.121543 | 0.000000 | 0.468570 | 6.938112 |

  both-skip 的 L2 仅为 selection union，选中 attention 另计；L1/L2 粗细分及 TopK 实例均真实消失。以上仍为 pending-independent-verification，正式自然 gate 全矩阵尚未完成。
- 第一代 16 份强制四态回执已预验 PASS，四种模式的 Graph/eager 完整 clean token 均逐 token 一致；eager 结果如下（最终数值仍以新 harness 代重跑为准）：

  | 强制模式 | 总算子 ms/步 | clean TPOT ms |
  |---|---:|---:|
  | no-skip | 4.031333 | 26.702720 |
  | near-only | 2.955325 | 25.967664 |
  | far-only | 3.515828 | 26.680064 |
  | both-skip | 0.477530 | 25.520753 |

  eager 主机调度占完整 TPOT 的大部分，因此 both-skip 的算子消除不会等比例转化为 TPOT；不把 Graph 与 eager 的绝对 TPOT 混作同一执行口径。
- 全量内部审计器已部署为 `/tmp/audit_nfa_region_skip_full_20261111.py`，将在 `COMPLETE` 后核对 208 份测量回执与 104 对 nsys report/sqlite，并检查同配置 eager/Graph 完整 clean token identity。
- 全部新结果保持 **pending-independent-verification**。完成前不提交、不推送；完成后需核对八组正式 summary、`COMPLETE`、token identity、Graph capture/fallback、GPU hygiene、源码/输入/nsys 身份及原始回执 manifest。
- 为缩短单卡串行等待，正式矩阵改为确定性跨机分片：pc31 GPU0 只跑 Graph，pc02 GPU0/1/2/3 分别跑 `tail_skip/tail_noskip/full_skip/full_noskip` eager；每格只归属一机一卡，最终合并原始回执，不把不同主机的 eager/Graph 绝对 TPOT 混作同一执行口径。双机 HEAD 均为 `84bd19ee`，source diff SHA256=`0ab60bfca394f1c2f1c52cbf9ba31f06cf3aa56bbb05b3ac8da5dffd826ba53f`，harness SHA256=`8bfa80e38d88995fa8902fd00dd34ec86acdb726a757eda6ff7a1fe9b0299aef`，五长度输入 SHA256 与关键生产文件逐字一致。
- pc02 eager 分片脚本 `/tmp/run_nfa_region_skip_eager_shard_20261111.sh`，SHA256=`66de2d92784b8472a424f8531d965f2d447d9a47366e262b3332f90b1b81c2b9`。32K B1 `tail_skip` 双臂冒烟通过且 hygiene=clean：算子 3.822910 ms/步、clean TPOT 47.557344 ms；该值只证明 pc02 路径可用，不与 pc31 eager 数值直接比较。正式父 PID 为 3653667/3653668/3653669/3653670，点火后四卡均约 119.9 GiB、100% 利用且日志正常。
- pc31 原父 orchestrator PID=2176792 已精确停止，正在运行的 `tail_skip_graph` 子进程 PID=2182675 保留并转为 orphan，未中断测量。负载均衡后 pc31 只完成该 Graph 组，脚本 `/tmp/run_nfa_region_skip_graph_only_20261111.sh`，SHA256=`cd5d0b0acd7c6e11d3e3f2edd170522e8f66d2dbdad194e78454f197b63e5c26`，新父 PID=2203483，完成标志为 `TAIL_SKIP_GRAPH_COMPLETE`。
- 其余三组 Graph 由 pc02 在同卡 eager 正常完成后接续：GPU1/2/3 分别对应 `tail_noskip/full_skip/full_noskip`，链脚本 `/tmp/run_nfa_region_skip_graph_shard_20261111.sh`，SHA256=`5cc00382aae2e00c79cea5b75b8bd1485440fb46e9e7827bec7db2a9c07a13a9`，父 PID=3656780/3656781/3656782。若对应 eager 未写出 `COMPLETE`，链会 fail-closed，不启动 Graph；GPU0 的 `tail_skip` eager 完成后不再跑 Graph，避免与 pc31 重复。
- 24 份复用的 FlashInfer Graph 原始回执已提前重新验证通过：page=64、wrapper、输入 SHA256、nsys report/sqlite 哈希、GPU hygiene、capture/fallback 和 profile/clean token identity 全部 PASS。最终仍只声明 FlashInfer Graph 参考臂，不伪造 eager operator arm。
- `tail_skip_graph` 已在 pc31 完成 12/12 回执、summary 与 `TAIL_SKIP_GRAPH_COMPLETE`。为利用释放的 GPU0，`tail_noskip_graph` 随后唯一转移至 pc31，PID=2285024，脚本 SHA256=`8770daa784dbaa91fbe91093730d0a07d3d2a1288e4cfe7108f1560dc5171658`；pc02 原等待链 PID=3656780 在未启动测量前精确取消，其 eager 父进程不受影响。pc02 最终只接续 `full_skip_graph` 与 `full_noskip_graph`。
- 正式测量中出现的 eager B2/B4 clean TPOT 反向开销暂保留为负结果：skip 的 nsys 算子时间下降，但完整 TPOT 未同步受益。最终报告不得用选择器算子加速替代 clean TPOT 结论，也不将跨主机 Graph/eager 绝对耗时直接比较。
- pc02 四组 eager 均已完成 12/12 回执与 `COMPLETE`。为利用空闲 GPU0/1，旧 `full_skip_graph`/`full_noskip_graph` 父链 PID=3656781/3656782 在确认两组均 0/0 回执后按精确 PID 停止，四卡显存均释放至 1 MiB；新脚本 `/tmp/run_nfa_region_skip_graph_split_20261111.sh`，SHA256=`f09fbb373594573bb31e867ab51268f35b5249b11eea23c4fc023c1ded2d57f2`，父 PID=3876832/3876833。`full_skip` 由 GPU2 跑 512K/256K/128K、GPU0 跑 64K/32K；`full_noskip` 由 GPU3 跑大长度、GPU1 跑小长度。四个分片长度集合互斥，driver 回执按 GPU 区分，两臂都完成后才统一 summarize 并写 `COMPLETE`；点火后四卡均 100% 利用且对应 nsys 进程、日志正常。
- 旧自然 skip eager 的反向 TPOT 已归因为 host launch-plan 抖动，而不是区域 kernel 变慢：`_nfa_launch_geometry` 是全局单对象，但 plan 按层保存；相邻层 region 不同时，下一步会继承相邻层的 geometry，旧 `is` 判断令本层 plan 反复 miss。32K B1 有少量 region 边界，256K B2 更多，而 512K B1 全部 36 层统一 skip near，与旧反向开销的分布一致。
- host 修复分两阶段：完整入口把 `call_plan[0] is geometry` 改为按值匹配；公共快路在除 `geometry[31]`（region）外全部一致时，重新读取当前冻结签名并校验本层 region，再复用本层 plan。其余 layout revision、visible bucket、Q layout、capture 状态、`middle_only` 和输出地址守卫不变。该快路只在非 capture 路径生效，Graph replay 不执行 Python selector。
- 新增多层交替 region 回归测试：旧实现第二 step 会重建 plan；只做值比较后仍对两层各进入一次完整 `_layer_batched_select_impl`；最终修复后第二 step 的完整入口调用为 0，且两个 plan 对象均保持身份不变。最终综合聚焦回归 73 passed，最终 host 修复的三项定向测试 3 passed。
- 本机 GPU1 诊断中，32K B1 tail natural-skip eager TPOT 从第一阶段值比较的 29.214607 ms 降至最终快路的 27.230832 ms；同机 no-skip 对照为 27.360960 ms。该结果只证明根因与修复方向，不替代正式 pc02 回执，也不跨机比较绝对 TPOT。
- 旧 `tail_skip_eager`/`full_skip_eager` 两组 12/12 回执保留在 pc02 原 root 作为 host 缺陷失败代，不进入最终 208 份回执。新隔离 worktree `/tmp/nfa-region-skip-eager-fastpath-20261111` 仅增加非 capture host 修复，最终 selector SHA256=`06dd5a9c02254e2187f2527960f4013707f3fcb1783dad1b04090484964a2e20`，受控 source diff SHA256=`53eb98b3cc2f87ab868ceed71e3b4f311af927b23c5e45097d11508f9e9026fc`；Graph、no-skip eager 与统一 region 的强制四态继续保留原 source diff SHA256=`0ab60bfca394f1c2f1c52cbf9ba31f06cf3aa56bbb05b3ac8da5dffd826ba53f`，最终审计按组精确核验两种来源。
- 两组定向 eager 重测脚本为 `/tmp/run_nfa_region_skip_eager_fastpath_20261111.sh`，SHA256=`00dcfb885de6aafc9eacd5c1046478653c737230ef6d08bf30b7c337f9f09d10`，输出 root 为 `~/ak_verify_results/nfa_region_skip_eager_fastpath_20261111`。pc02 GPU0 的 `tail_skip_eager` 父 PID=3911996、GPU1 的 `full_skip_eager` 父 PID=3911997 已点火；启动前 fail-closed 核验 donor 小长度 Graph 六对回执，点火后 GPU0/1 均约 119.9 GiB、100% 利用，GPU2/3 继续大长度 Graph。最终 harvest 只从旧 root 收 Graph 与 no-skip eager，从新 root 收两组修复后的 natural-skip eager。

## 最终全量结果（内部审计 PASS，独立验收 pending）

- pc31 的 `tail_skip_graph`/`tail_noskip_graph`、pc02 的 `full_skip_graph`/`full_noskip_graph` 与新 root 的 `tail_skip_eager`/`full_skip_eager` 均已完成 12 份 operator + 12 份 clean、summary 与完成标志；旧 root 的 `tail_noskip_eager`/`full_noskip_eager` 也各为 12/12。加上 16 份强制四态回执，最终正好纳入 208 份 NFA 原始回执和 104 对 nsys report/sqlite。
- 最终审计 `/tmp/audit_nfa_region_skip_full_20261111.py` SHA256=`784aa972f4fcc2c009216814139ee0cf44be5485f60537711d2757bc11e55c75`，harvest SHA256=`046715b896a9350a3c6ad00aa5e839e8350ad0abb466f094ec7cd5b67692c6af`。审计 PASS：208 份 NFA 回执、104 对 nsys、24 个正式 paired cells、8 个 acceptance cells；输入 SHA256、GPU hygiene、Graph capture/fallback、区域 kernel 实例、强制 grid 缩容、profile/clean token 前缀、48 组同配置 Graph/eager 完整 clean token identity、两种 source identity 全部通过。
- 另重新核验 24 份未受 selector 修改影响的 FlashInfer Graph 参考回执，总核验回执数 232；page=64、wrapper、输入身份、nsys 哈希、hygiene、capture/fallback 与 token identity 均 PASS。不声明 FlashInfer eager operator arm。
- 结果：[`agent_doc/results/nfa_region_skip_performance_20261111.json`](../../results/nfa_region_skip_performance_20261111.json)，SHA256=`df0dbea733befe5d61161b565afa4e1be79566173652c0b4ae79315b9f27beff`；原始回执 manifest：[`agent_doc/results/nfa_region_skip_raw_receipt_manifest_20261111.json`](../../results/nfa_region_skip_raw_receipt_manifest_20261111.json)，SHA256=`2f03eb883d078883874ecfbc72f7de83bdcbd3ce31196b492cf6a233c93d402b`。`validation_status=pending-independent-verification`，内部 PASS 不冒充独立验收。
- 审计器收尾发现两处审计代码问题，均未修改任何原始回执：分组名带 `_graph/_eager` 后缀，不能用 `endswith('_noskip')` 判定 epsilon；eager 单 CTA 快路将 candidate 融合进 L1 Top-K，双臂 `candidate_ms=0` 时 0/0 加速比记为 `null`，不伪造 1×或非标准 Infinity，candidate 成本仍含在 `l1_topk_ms`。

### Graph 正式矩阵

下表均为 clean decode TPOT 比值；`skip` 为 no-skip/skip，`tail/full` 为 full-skip/tail-skip，大于 1 表示前者更慢、后者更快。NFA/FI 为 FlashInfer/NFA tail-skip。

| 长度 | B | tail skip | full skip | tail/full | NFA/FI TPOT |
|---:|---:|---:|---:|---:|---:|
| 512K | 1 | 1.0520 | 1.0264 | 1.3124 | 2.9712 |
| 256K | 2 | 1.0515 | 1.0867 | 1.3687 | 2.7452 |
| 256K | 1 | 1.0416 | 1.0159 | 1.2842 | 1.7737 |
| 128K | 4 | 1.0312 | 1.0796 | 1.4893 | 2.3504 |
| 128K | 2 | 1.0306 | 1.0572 | 1.3577 | 1.6087 |
| 128K | 1 | 1.0293 | 1.0109 | 1.2532 | 1.2926 |
| 64K | 4 | 1.0355 | 1.0939 | 1.4550 | 1.4712 |
| 64K | 2 | 1.0384 | 1.0660 | 1.3299 | 1.1900 |
| 64K | 1 | 1.0306 | 1.0073 | 1.2436 | 1.0544 |
| 32K | 4 | 1.0256 | 1.0591 | 1.5039 | 1.1147 |
| 32K | 2 | 1.0229 | 1.0428 | 1.3459 | 0.9883 |
| 32K | 1 | 1.0201 | 1.0079 | 1.2380 | 0.9616 |

- Graph 的 24/24 个自然 skip/no-skip clean TPOT 比值均大于 1，12/12 个 tail/full 比值均大于 1。512K B1 的 tail-skip/no-skip=10.711680/11.268816 ms，full-skip/no-skip=14.057488/14.428272 ms。
- NFA/FI clean TPOT 在 10/12 格大于 1；32K B2/B1 为 0.9883/0.9616，负结果保留。包含 NFA 选择器的 attention-path 对比在 64K B1、32K B2/B1 也小于 1（0.7193/0.7094/0.4601）；这些短上下文格不能用长上下文收益掩盖。所有对照均已通过相同输入、page=64、源码、Graph 和 hygiene 审计，状态仍为 pending-independent-verification。

### 修复后的 eager 正式矩阵

| 长度 | B | tail skip | full skip | tail/full |
|---:|---:|---:|---:|---:|
| 512K | 1 | 1.0063 | 1.0071 | 1.0143 |
| 256K | 2 | 0.9916 | 1.0143 | 1.0031 |
| 256K | 1 | 0.9938 | 1.0515 | 0.9903 |
| 128K | 4 | 1.0049 | 1.0183 | 1.0068 |
| 128K | 2 | 0.9807 | 1.0006 | 0.9866 |
| 128K | 1 | 1.0173 | 1.0186 | 1.0144 |
| 64K | 4 | 1.0202 | 1.0231 | 1.0164 |
| 64K | 2 | 0.9872 | 1.0255 | 0.9899 |
| 64K | 1 | 0.9805 | 1.0234 | 1.0043 |
| 32K | 4 | 0.9824 | 0.9843 | 1.0069 |
| 32K | 2 | 0.9926 | 1.0009 | 1.0154 |
| 32K | 1 | 1.0059 | 0.9704 | 1.0353 |

- eager 仍由 host 调度主导，算子消除不会等比例转化为 TPOT；tail 有 7 格、full 有 2 格 skip/no-skip 小于 1，全部保留，不宣称 eager 全格正收益。host 修复消除了旧实现的异常大幅反向开销，例如 256K B2 tail/full 从旧 50.360401/52.775167 ms 降至 32.823105/32.924112 ms；新 32K B1 tail-skip/no-skip=31.088912/31.272144 ms，full-skip/no-skip=32.185007/31.232832 ms，full 格仍为负结果。
- 最终综合聚焦回归为 **85 passed、54 subtests passed**，`git diff --check` 与生产/审计 Python 静态编译通过。性能结论严格区分区域 kernel 消失、grid 缩容、Graph clean TPOT、eager clean TPOT 与旧 host 缺陷失败代。

## 三长度单层等效四态算子（2026-11-12，内部审计 PASS，独立验收 pending）

- 用户要求的口径是强制全 36 层处于同一模式，再以“整模型每个 decode step 的阶段总耗时 ÷ 36”得到**单层等效算术均值**，不是自然 gate 混合层 trace，也不是对某个具体 layer index 的单独归因。32K 复用上述 pc31 已审计强制四态回执；64K/128K 在 pc33 clean `b97af7ef909957e69b194e10ce4c97f5b1089be2` 上补测 16 份 operator 回执，不新增 clean TPOT 回执。
- 输入统一为 Qwen3-8B × RULER `niah_multikey_2` sample2、native 32K/64K/128K、B1、tail=32/32、epsilon=0.10，3 warmup + 5 nsys profiled decode steps。Graph 每格 capture=1、replay=8、fallback=0，eager capture=0。32K 与 64K/128K 的测量主机和源码代次已在每份记录中显式保存；优先使用同长度四态比值，不把跨主机绝对值解释为受控长度扩展。
- 下表单位均为 **µs/层/decode step**。`near-only` 表示 skip far、只算 near；`far-only` 表示 skip near、只算 far。TopK 为 L1/L2 ranking 合计且不与 L1/L2 再相加；NFA total = L1 + L2 + selected attention。eager 的单 CTA 快路把 candidate 融进 L1 TopK，因此其独立 candidate 为 0 并不代表未执行 candidate。

### CUDA Graph

| 长度 | 强制模式 | L1 | L2 | TopK 合计 | selected attention | NFA total | total/no-skip 加速比 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 32K | no-skip | 23.180206 | 57.953339 | 30.620572 | 28.191628 | 109.325172 | 1.0000 |
| 32K | near-only | 18.235744 | 36.504556 | 25.344894 | 28.217972 | 82.958272 | 1.3178 |
| 32K | far-only | 21.112817 | 46.654611 | 27.670883 | 28.244750 | 96.012178 | 1.1387 |
| 32K | both-skip | 0.000000 | 3.376922 | 0.000000 | 9.608639 | 12.985561 | 8.4190 |
| 64K | no-skip | 24.356306 | 71.436761 | 44.767006 | 28.521106 | 124.314172 | 1.0000 |
| 64K | near-only | 19.642172 | 48.261444 | 37.188528 | 28.491606 | 96.395222 | 1.2896 |
| 64K | far-only | 20.864756 | 59.810667 | 41.115839 | 28.505500 | 109.180922 | 1.1386 |
| 64K | both-skip | 0.000000 | 3.393439 | 0.000000 | 9.660467 | 13.053906 | 9.5231 |
| 128K | no-skip | 26.194206 | 71.779172 | 46.084356 | 28.474717 | 126.448094 | 1.0000 |
| 128K | near-only | 18.100661 | 48.378244 | 37.203994 | 28.485050 | 94.963956 | 1.3315 |
| 128K | far-only | 22.293744 | 59.664711 | 41.894694 | 28.432228 | 110.390683 | 1.1455 |
| 128K | both-skip | 0.000000 | 3.401256 | 0.000000 | 9.638800 | 13.040056 | 9.6969 |

### eager

| 长度 | 强制模式 | L1 | L2 | TopK 合计 | selected attention | NFA total | total/no-skip 加速比 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 32K | no-skip | 24.974106 | 58.818444 | 39.016861 | 28.452311 | 112.244861 | 1.0000 |
| 32K | near-only | 16.024600 | 37.559767 | 27.480089 | 28.461622 | 82.045989 | 1.3681 |
| 32K | far-only | 21.320089 | 47.871289 | 34.010633 | 28.446150 | 97.637528 | 1.1496 |
| 32K | both-skip | 0.000000 | 3.475194 | 0.000000 | 9.801422 | 13.276617 | 8.4543 |
| 64K | no-skip | 32.526667 | 71.983017 | 59.474122 | 28.562911 | 133.072594 | 1.0000 |
| 64K | near-only | 16.186339 | 49.307022 | 39.251878 | 28.640817 | 94.134178 | 1.4136 |
| 64K | far-only | 22.015722 | 60.650733 | 47.455778 | 28.553822 | 111.220278 | 1.1965 |
| 64K | both-skip | 0.000000 | 3.484111 | 0.000000 | 9.743678 | 13.227789 | 10.0601 |
| 128K | no-skip | 55.460350 | 72.243150 | 82.041100 | 28.589617 | 156.293117 | 1.0000 |
| 128K | near-only | 15.709906 | 49.220100 | 38.771844 | 28.545711 | 93.475717 | 1.6720 |
| 128K | far-only | 30.091728 | 60.663633 | 55.230450 | 28.579906 | 119.335267 | 1.3097 |
| 128K | both-skip | 0.000000 | 3.487122 | 0.000000 | 9.692878 | 13.180000 | 11.8584 |

- both-skip 的 L1、L2 score、candidate 与全部 TopK 都是真实 0；表中非零 L2 仅为 protected-only `selection_union`，另有 selected attention，所以整层 NFA total 不为 0。相应零分母加速比在 JSON 中记为 `null`，不伪造 Infinity 或 1×。单侧 skip 的 selected attention 基本不变，围绕 1 的微小比值不作收益主张。
- 24 份 JSON + 24 对 nsys report/sqlite 共 72 个文件的 SHA256、字节数、输入身份、源码身份、GPU hygiene、Graph capture/replay/fallback、四态层签名和区域 kernel 实例全部通过审计。Graph 三长度的 `_l1_topk_partial_kernel`、`_candidate_expand_batched_kernel`、`_nfa_filtered_topk_partial_kernel`、`_nfa_fine_scores_paged_kernel` 在两种单侧 skip 下均有真实 `gridY` 缩容；eager 不伪造单 CTA 快路的同类 grid 结论。
- 汇总与 fail-closed 审计脚本：[`agent_doc/results/build_nfa_region_four_state_per_layer_20261112.py`](../../results/build_nfa_region_four_state_per_layer_20261112.py)。结果：[`agent_doc/results/nfa_region_four_state_per_layer_20261112.json`](../../results/nfa_region_four_state_per_layer_20261112.json)，SHA256=`f297815d568f770d09acb0037e057b7c1def48ca6928b0bbe149542756c01f1d`；manifest：[`agent_doc/results/nfa_region_four_state_per_layer_raw_manifest_20261112.json`](../../results/nfa_region_four_state_per_layer_raw_manifest_20261112.json)，SHA256=`829d50a39ce2f6af12f76da442f1780419ead54259a296c7f3d5bc69c3b63d63`。另以独立只读小程序复核 72/72 文件哈希、24 个唯一配置格和 264 项单层换算，均 PASS；这不替代不同 owner 的正式独立验收。双卡测量结束后均回落至 1 MiB、0%。

## 三长度单层等效纯 L1/L2 打分 kernel（2026-11-12，内部审计 PASS，独立验收 pending）

- 上节的 `operator_summary.l1_score_ms/l2_score_ms` 是**较宽阶段桶**：L1 桶还含页统计/Q 准备，L2 桶还含 grouped Q 准备。本节按用户口径只提取真正的打分 kernel，不使用阶段桶代替纯 kernel 时间，也不改写上一节结果或其哈希。
- L1 仅包含 no-skip 的 `_nfa_l1_scores_kernel`、near-only 的 `_nfa_l1_near_scores_kernel`、far-only 的 `_nfa_l1_far_scores_kernel`；L2 仅包含 `_nfa_fine_scores_paged_kernel`。排除 Q 准备、页统计、L1/L2 Top-K、candidate、selection union、selected attention 和所有非选择器模型工作。单位仍为 **µs/层/decode step**，由对应 kernel 的整模型 `mean_ms_per_step × 1000 ÷ 36` 得到。

| 长度 | 强制模式 | Graph L1 | Graph L2 | eager L1 | eager L2 |
|---:|---|---:|---:|---:|---:|
| 32K | no-skip | 4.026506 | 25.909461 | 4.199344 | 25.760567 |
| 32K | near-only | 3.462578 | 7.408722 | 3.553250 | 7.580300 |
| 32K | far-only | 3.911667 | 16.084522 | 3.994156 | 16.317022 |
| 32K | both-skip | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 64K | no-skip | 4.204100 | 25.916489 | 4.314139 | 25.599944 |
| 64K | near-only | 3.533156 | 7.374233 | 3.615289 | 7.564461 |
| 64K | far-only | 3.948983 | 16.101917 | 4.016856 | 16.141083 |
| 64K | both-skip | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 128K | no-skip | 4.753806 | 26.284489 | 4.845178 | 25.852722 |
| 128K | near-only | 3.485883 | 7.371578 | 3.588094 | 7.510067 |
| 128K | far-only | 4.368367 | 16.186367 | 4.433617 | 16.140500 |
| 128K | both-skip | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

- `near-only` 即 skip far。Graph 纯 L1 的 no-skip/near-only 加速比在 32K/64K/128K 分别为 **1.1629/1.1899/1.3637×**，纯 L2 分别为 **3.4972/3.5145/3.5657×**；skip near 的 Graph 纯 L2 加速比约为 **1.611/1.610/1.624×**。L2 收益显著且与 candidate 缩容一致；L1 虽观察到 `gridX=5/9/17→2`，但短 kernel 固定成本、Q 读取和写回等使其时延没有同比下降，不能包装成大幅加速，也不能据此反推 skip 未生效。
- both-skip 的四种 L1/L2 score kernel 均无实例，因此两项严格为 0，零分母加速比记为 `null`。单侧 skip 的 L1 `gridX` 与 L2 `gridY` 缩容断言全部通过。32K 仍复用 pc31 回执，64K/128K 使用 pc33 回执；只解释同长度模式比值，不把跨主机绝对值当作受控长度扩展。nsys/CUPTI 观察者效应保留，本表不是 clean TPOT 或 host-overhead 结论。
- 汇总与 fail-closed 审计脚本：[`agent_doc/results/build_nfa_region_pure_score_per_layer_20261112.py`](../../results/build_nfa_region_pure_score_per_layer_20261112.py)，SHA256=`56d0ffd76b4abd5025405ddbab0a1d99483fe00b5d5a26fb3fcba0dc9db0cbc7`。结果：[`agent_doc/results/nfa_region_pure_score_per_layer_20261112.json`](../../results/nfa_region_pure_score_per_layer_20261112.json)，SHA256=`1808fd38634e2e7e65e4aa40d663c780cb42dfcaecd7930f9f82e0898f20f738`；继续引用上述原始 manifest，SHA256=`829d50a39ce2f6af12f76da442f1780419ead54259a296c7f3d5bc69c3b63d63`。构建器重新核验 24 份回执、24 对 nsys report/sqlite、源码/输入身份、GPU hygiene、kernel 实例和 grid 缩容，全部 PASS；另以独立只读程序复核 24 个唯一配置格、表中全部数值、both-skip 零值以及旧结果/manifest 哈希，均 PASS。这不替代不同 owner 的正式独立验收，状态保持 `pending-independent-verification`。
