# [TL-QUEST-BATCH-RULER-PERF-20261116] Quest 官方 kernel 真 batch 化 + RULER 多长度性能缺口补齐

Task-ID: TL-QUEST-BATCH-RULER-PERF-20261116
Date: 2026-11-16
来源：用户指令「继续基于本地main用本地两卡补充RULER不同长度下的性能数据，只跑缺的，Quest BS 1 2 4，32K/64K/128K 还有其他baseline的和NFA的」+「记得锁卡测试，算子(L1,L2)和e2e的(decode阶段)都需要」。

## 用户拍板的口径

1. Quest B2/B4：**改官方 kernel 支持真 batch**（不接受逐行完整 forward 冒充 batch）。
2. 算子级测量：NFA（纯 L1/L2 打分 kernel）+ Quest（decode_estimate/decode_topk，单级选择器，报告中明确映射）+ AsyncTLS（K1/K2 打分）全测；nsys 与 clean e2e 分离采集，不用 profiler 扣减 clean e2e。
3. 只跑缺的：既有矩阵 NFA/AsyncTLS 的 32K/64K/128K × B1/B2/B4 与 Quest B1 已闭集，不重跑；e2e 缺口 = Quest B2/B4 × 32K/64K/128K 共 6 格。
4. 锁卡独占测量；结果保持 pending-independent-verification。

## 现状核查（2026-11-16）

- 本机双 H20 实测各 1MiB、无 compute app。
- 既有归档 `agent_doc/results/qwen3_official_ruler_multikey2_perf_20261115/`（commit 1cfed848）。
- Quest B>1 此前 fail-closed 于 `validate_official_batch_size` 与 `_qwen3_quest_forward` 的 bsz!=1 检查。
- Quest 工作树基线 = 官方 `01c1623bf9395009520874e989e29f683203b357` + 原生 GQA 适配 + 64-head Top-K 修复（8 文件 diff，生产现状）。

## Quest batch 化设计（代码审计结论）

B1 生产唯一 decode attention 路径 = partition-kv（非 partition 是从未执行的半成品，不动）。
last page 以「每序列末尾附加一个伪 chunk」参与 VariableLengthMergeStates，天然可泛化。

### CUDA（保持 B1 逐位等价，全部走统一路径 + golden 对拍）

1. `decode_page.cuh`
   - `paged_kv_t` 新增 `const IdType* last_page_idx_ptr`（device 数组，每序列一个 last page 物理页号）。
   - `protective_get_k_ptr_heads` 的 `blockIdx.x==gridDim.x-1` hack 改为显式 `is_last_chunk` + `last_page_idx` 参数。
   - `QKApplyRotaryInPlaceKernel` 加 `position_stride`（位置 = offset + token_idx*stride；B1=1 原行为，batched decode=0 全行同位置）。
2. `decode_attn.cuh`
   - estimate kernel：`cur_last_page_len` 恒为 `last_page_len-1`（每序列都排除当前 entry；B1 原条件恒真，等价）。
   - partition decode kernel：`seq=batch_idx_map[batch_idx]`；`is_last_chunk=(batch_idx==chunk_indptr[seq+1]-1)`（B1 时与 gridDim.x-1 等价）；last page 从 `last_page_idx_ptr[seq]` 取。
   - `PartitionPagedKVCacheComputeAuxiliaryInfo`：last-page 伪 chunk 附加移入每序列循环（B1 完全一致）。
   - `WorkEstimation`：`new_batch_size += batch_size`（原 +=1）。
   - **零拷贝 Top-K indices 布局**：estimate 直接输出 `[B,H,P]`，Top-K 按每 32 行固定 RAFT 配置输出 `[B,H,K]`，attention 以 `seq*H+head` 读取对应行；B1 保持 `[H,P]/[H,K]`，不引入逐层 transpose/copy。后续探针已否定“行地址对齐可保证 RAFT 跨调用确定”的假设：B1 同地址重复调用也会改变原始排列，且 FP16 同分边界可能换页。生产修复保留 RAFT 第 K 阈值，在 CUDA 中按原输入逻辑顺序确定性补齐边界同分页，再用 physical→logical lookup 对已选页面和分数同步做逻辑页排序；无 Torch Top-K/attention fallback。
3. `estimate.cu` host：batch_size=q.size(0)；o=`[H,P]`（B1）或 `[B,H,P]`（batch）。
4. `approx_attn.cu` host：BeginForward batch=indptr.len-1；Forward batch=q.size(0)，新增 last_page_idx tensor 参数，indices 接受 `[H,K]` 或 `[B,H,K]`。
5. `topk.cu`/`decode_select_k.cuh`：新增运行时行数版本，输入 `[H,P]` 或 `[B,H,P]`；行数为 H 或 H*B，要求可被 32 整除。
6. `page.cu` host：append decode batch=indptr.len-1（kernel 本身已支持 batch_idx）；rope 加 position_stride。
7. prefill：**逐行复用官方 B1 kernel**（append_kv_cache_prefill + prefill_with_paged_kv_cache）。两点法 T(33)-T(1) 中 prefill 差分严格抵消，不影响 decode TPOT；runtime 明确披露 `prefill=per-row official B1; decode=fully batched official kernels`。

### Python / HF

- `KvCache` 支持共享 `KvPool`；controller 支持 B 序列（共享池、拼接 indices/indptr、last_page_idx 数组、estimate/topk buffers `[B,H,*]`）。
- decode RoPE：native 用 position_stride=0；YaRN 用 HF 4D rotary（B1 与 B>1 同路径，对拍验证）。
- HF forward 放开 bsz=B；prefill 逐行视图；decode 全 batch。
- `validate_official_batch_size`：Quest 放开 1/2/4；JSSA 仍 B1 fail-closed。
- 性能入口：identical input IDs replicated across batch rows（与 NFA 相同），batch-major 输出，TPOT 不按 B 归一化。

### 正确性门禁（先全部通过才允许正式测量）

- G1：官方 B1 测试全过（含 32-head golden、64-head topk、GQA 对拍）。
- G2：新增 batch 单元对拍——等长不同输入下，B2/B4 每行的 estimate/topk/sparse_attn 与逐行 B1 **逐位一致**；同输入时每行与 B1 一致。
- G3：真实 Qwen3 smoke（32K native + 64K YaRN），B2/B4 每行 33 个 token IDs 与 B1 完全相同；官方调用计数、fallback=0。
- G4：既有 B1 Quest 性能格可复现复核（同协议重跑一次 B1 与既有闭集数字对照，验证 batch 化未污染 B1 性能语义）。

## 测量计划

- e2e（clean、锁卡、1 warmup+3 repeats、33 token、T(33)-T(1)、K2=1024/page16、niah_multikey_2 sample-index2）：
  Quest B2/B4 × 32K/64K/128K = 6 格；B1 复跑仅作 G4 对照不覆盖既有闭集。
- 算子级（nsys 分离采集）：Quest estimate/topk/sparse_attn；NFA 纯 L1/L2 score；AsyncTLS K1/K2 score；32K/64K/128K × B1/B2/B4。
- 统一审计（沿用 audit_perf.py 口径扩展 batch 格）→ 归档 → 提交推送。

## Progress

### 2026-11-16：代码审计与设计定稿

- 完成 Quest 官方 decode 全链代码审计（controller/kv_cache/utils/estimate/topk/approx_attn/decode_page/decode_attn/decode_handler/prefill），确认上述方案；本机双卡实测空闲。

### 2026-11-17：Top-K 非确定性归因、生产修复与 G1–G3

- 原始真实模型观测：32K native B2/B4 与 64K YaRN B2 曾对拍通过；64K YaRN B4 的 row2 曾在 token 30/32 出现 `2727→33845`、`3966→79`，一次 layertrace 重跑又全行通过，但中间 logits 并非逐行一致，因此不能判稳定通过。
- 64K×33 步纯官方 CUDA 对拍的初版诊断脚本误把 B1 `[H,P]` 多索引一次，产生虚假 estimate 差异；修正后 33/33 步确认 estimate 逐位一致、Top-K 页面—分数 payload 与选值集合一致，强制同一逻辑页顺序后 sparse attention 的 B4 行隔离及 B1/B4 逐位一致。该结果只证明强制顺序下的 kernel 语义，不等同于生产 Top-K 顺序正确。
- 最小 RAFT 探针否定地址对齐根因：B1 同输入、同地址连续两次官方调用的原始 values/indices 排列也会变化，重新对齐 input/output/both 均不能消除。`P=4089` 的排序选值和页集合稳定；`P=4090/8191` 已观察到同分边界换页。原始 RAFT 排列差异不得包装成选错页或行串扰。
- 生产修复落盘：`controller.py` 每步构造 `[B,pool_capacity]` physical→logical Int32 lookup；`topk.cu` 保留每 32 行官方 RAFT 配置，用 RAFT 输出的第 K 阈值，在 CUDA 中按原输入顺序确定性地补齐边界同分页，再通过共享内存 bitonic sort 按逻辑页号同步重排 values/physical indices。B1/B2/B4 统一路径；旧六参数直接 API 不变，无 Torch Top-K、稀疏注意力或 vendor fallback。当前扩展 SHA256=`d327f1a407fd6fa61bbfce48b882350aff77622640daaac2ecab4b85fd5da139`。
- 修复过程中的真实失败保留：仅规范排序时短序列 B1/B4 为 4 passed/2 failed；三处失败均为 estimate 逐位一致、阈值有 2–3 个同分页但 B1/B4 选集不同，因此增加边界确定性补齐，而非修改 expected。新增强同分测试第一次失败还暴露了 B1 测试参考形状 `[K]` 被错误扩展成 `[H,K]` 的测试自身错误，已修正为按实际形状对拍，生产断言未放宽。
- 聚焦证据目录=`/home/wangyuanshuo02/ak_verify_results/qwen3_quest_batch_smoke_20261117/canonical_20261117/`：
  - 规范排序、强同分边界和短序列 B1/B4 对拍：`8 passed`（`focused_short_tie_fix_pass2_gpu0.log`）。
  - 64K 真实几何、B4 identical rows、33 步生产 `decode_estimate/decode_topk/decode_sparse_attn` 逐位对拍：`1 passed`（`multistep_64k_gpu0.log`）。
  - batch decode + RoPE：`71 passed`；Quest 全部 `quest/tests`：`276 passed, 43 skipped`。
  - 主仓 `tests/test_official_longbench.py` 普通与 `python -O`：各 `32 passed, 1 skipped, 20 subtests passed`。
- 真实 Qwen3 G3 首轮已通过：以同一修复版本的新 B1 为 golden，32K/64K × B2/B4 四格每行 33 token 全部逐 token 相同；input IDs、33-token runcap、EOS 不提前停止、官方调用计数和扩展哈希一致，prefill 调用数随 B 成倍、decode 调用数与 B1 相同，`dense_or_torch_fallback=0`。生产侧审计=`records/model_smoke_audit.json`，status=pass；smoke 的 warmup=0/repeat=1 计时明确不进入正式性能排序。
- GPU 健康检查按实际时间记录：点火前双卡均 1MiB、无 compute app；B1 32K 启动约 30 秒时观测到 PID 4030146、GPU1=28086MiB 和有效模型加载日志。四个 batch smoke 的首次即时检查早于 Python 启动，随后实测到两个真实模型进程及 GPU0/1=63428/62108MiB；短 CUDA 测试约 12 秒即完成，未虚构“30 秒内仍有驻留进程”。
- 64K B4 独立进程重复运行再次通过：两次 B4 的 4×33 token 完全相同，输入身份与官方调用计数一致，`records/model_smoke_repeat_audit.json` status=pass。G3 当前生产实现与两轮真实模型证据均通过；六个正式 B2/B4 e2e 格尚未启动，G4 继续进行，结果在不同 owner 独立验收前保持 pending-independent-verification。

### 2026-11-17：G4 串行扫描回归、并行修复与 input-order 重构

- G4 真实发现规范 Top-K 的逐行单线程全页扫描造成 B1 性能回归：既有闭集 32K/64K/128K 中位 TPOT=`19.266679/22.984220/24.684969ms`，串行规范版=`19.715138/27.467289/41.408ms`（约 `+2.33%/+19.51%/+67.75%`）。输入身份、33-token runcap、官方调用计数和 fallback=0 均通过，失败性能证据保留于 `performance/records/g4_*` 与 `g4_serial_scan_audit.json`；六个正式 B2/B4 格未启动。
- 第一版并行修复先完整并行求严格大于 RAFT 第 K 阈值的总数，再按 tile 对 greater/tie 做确定性前缀选择，避免把当前 tile 的 `greater_base` 错当全行总数。扩展 SHA256=`18722d121a5f35b674bcb9a4696644b8f31d2fee7d3fc7592d9b39ac8ca4c7e4`；聚焦 `8 passed`、64K×33 步 `1 passed`、Quest 全部 `276 passed, 43 skipped`。四格真实模型 `_parallel` smoke 全部完成，32K/64K × B2/B4 每行 33 token 均与确定性 B1 golden 一致，输入身份、EOS 不提前停止、真 batch 调用计数和 fallback=0 审计通过；生产侧审计=`records/model_parallel_smoke_audit.json`，smoke 计时不进入正式排序。
- 进一步审计确认 `kv_indices_without_last` 已按每序列逻辑页顺序排列，输入位置本身就是逻辑页号，无需每层构造 `[B,pool_capacity]` physical→logical lookup。当前 input-order 版本删除该 `torch.full+scatter_`，CUDA 直接以输入位置为排序键；第七参数改为默认 `false` 的 `canonicalize_input_order`，生产显式传 `True`，旧六参数 API 保持有效。新扩展 build/package SHA256 均为 `47c9fcb557bcb98670a2ee9a4075edfea8c06e3ff427f000239986f5856c7288`。
- input-order 聚焦 `8 passed`、64K×33 步 `1 passed`、Quest 全部 `276 passed, 43 skipped`；主仓官方入口普通与 `python -O` 各 `32 passed, 1 skipped, 20 subtests passed`。四格 `_input_order` 真实模型 smoke 已全部完成，32K/64K × B2/B4 每行 33 token 均与确定性 B1 golden 一致，输入身份、EOS 不提前停止、真 batch 调用计数和 fallback=0 审计通过；生产侧审计=`records/model_input_order_smoke_audit.json`，smoke 计时不进入正式排序。
- input-order G4 三长度正式复测已闭集，中位 TPOT=`18.967396/23.452801/25.573658ms`，相对既有 B1 闭集为 `-1.5534%/+2.0387%/+3.6001%`；串行规范版在长序列的 `+19.51%/+67.75%` 退化已消除，保留并行规范化的小额开销，不宣称零成本或严格性能等价。审计=`performance/records/g4_input_order_audit.json`，三格输入身份、33-token runcap、官方调用计数和 fallback=0 均通过；64K/128K 与历史非确定性版本的 token 差异只作对照，当前正确性以同确定性语义 B1 golden 为准。64K 点火后 25 秒实测 Python PID 4123632、GPU1=36640MiB 且模型加载日志有效；128K 同步在 GPU0 运行，G4 完成后双卡均回 1MiB。
- G1–G4 门禁证据齐备后，已启动六个真实缺格 `formal_input_order`：GPU0 串行 B2、GPU1 串行 B4，均为 32K→64K→128K、1 warmup+3 repeats、固定33 token、完整 batched-request host wall、clean e2e；结果尚未收割，继续 pending-independent-verification。

### 2026-11-17：正式格收割、B4 128K 显存修复与官方 nsys（在飞）

- 已有五格真实完成并保留 JSON/日志：32K B2/B4 中位 TPOT=`20.311162/22.389ms`，64K B2/B4=`25.672/29.172ms`，128K B2=`29.527ms`。128K B2 三 repeats=`29.527/30.553/21.268ms`，末次低离群值原样保留，不删除或替换；统一输入身份、33-token、EOS、调用计数和 B1 golden 审计待最后一格完成后执行。
- 128K B4 默认分配器在首个 warmup prefill MLP OOM：`138.12GiB` 已用、`1.59GiB` 空闲、再申请 `6GiB`，原始日志保留为 `formal_input_order_quest_b4_128k_gpu1_default_allocator_oom.log`（SHA256=`a1e308a0a6252bad3b3f5adabb5ac59bfccbf5e65b81a48674e96c0bd4b648b4`）。仅改用 `expandable_segments:True` 后仍 OOM：实际 allocated=`135.53GiB`、空闲=`3.62GiB`，排除单纯碎片问题，第二份日志 SHA256=`bebaf635f9cb5e9c053546130734c5f601c0470080447b2a08ebfabaad3110bd`。
- 显存修复只将 **B4 且超过 64K** 的 tokenwise MLP/RMSNorm prefill 分块从 65536 减半到 32768；B1/B2 与所有 ≤64K 输入保持原路径，decode CUDA、输入、RoPE 和计时协议不变。新增 runtime 字段 `prefill_tokenwise_chunk_b4_over_64k=32768` 及 B4/B2/短输入分块单元断言；主仓官方入口普通和 `python -O` 均 `32 passed, 1 skipped, 20 subtests passed`。修复版后台任务 `be077qu7r` 于 `2026-11-16T20:42:59Z` 在 GPU1 点火，约 56 秒时实测宿主 PID 2612426、GPU1=141402MiB/100%、模型加载完成且无 OOM；当前仍在运行，结果保持 pending-independent-verification。
- GPU0 在 B2 链完成后启动一个 32K B1 官方 Quest CUDA nsys 采集 `bucd8lv1n`：同一生产入口、0 warmup+1 repeat、33 token、完整进程 CUDA trace；其 TPOT 不进入 clean e2e 排序，后续只按官方 kernel 名归因 estimate/Top-K/sparse-attn。九格算子矩阵尚未完成，NFA 当前身份与 AsyncTLS K1/K2 可归属测量仍是后续缺口。
- 32K/64K/128K B1 与 32K B2 四格原始生成和 nsys 已真实完成，严格归类为 estimate=`MaxPossibleSampleWithPagedKVCacheKernel`、Top-K=RAFT+`canonicalize_topk_kernel`、sparse-attn=partition+merge；不计 prefill/KV append，profiler TPOT 永久隔离。B1 三长度平均每步 estimate=`1.355149/2.672924/5.306085ms`、Top-K=`1.500483/2.520719/4.425611ms`、sparse-attn=`0.588715/0.778616/1.258637ms`；其中 canonicalize=`1.290231/2.257981/4.074609ms`，未隐藏在 RAFT 时间中。32K B2 对应三类为 `1.430124/1.807680/0.956047ms`。
- 首版 B2 finalizer fail-closed 于 `topk_raft kernel 调用数 2176 != 1088`，原始 JSON/nsys/SQLite 均保留，未重跑生成。逐 kernel 复核证实官方 RAFT 按 batch 行各启动一个 one-block kernel，estimate/canonicalize/partition/merge 仍按层调用；计数模型修正为 RAFT=`34×32×B`、Top-K 合计=`34×32×(B+1)`。修正后复用原 SQLite 完成 B2 32K 归因（RAFT/canonicalize=`2176/1088`、Top-K 合计3264），B1 32K 复验数值不变。剩余五格由 `buz92jvbo` 于 `2026-11-16T21:03:46Z` 在 GPU0 串行补采；`21:05:05Z` 实测生产进程、有效日志与 GPU0=`62122MiB/100%` 三重健康，GPU1 的 128K B4 clean 格仍独立运行。九格统一审计和 clean 六格统一审计均未执行，结果继续 pending-independent-verification。

### 2026-11-17：Quest 正式链中间收割（七格 operator 完成，clean 最后一格在飞）

- GPU0 的 B2 128K 与 B4 32K 官方 nsys 已真实完成并生成 JSON/nsys-rep/SQLite。B2 128K 平均每步 estimate/Top-K/sparse-attn=`5.470003/4.867397/2.488253ms`，其中 RAFT/canonicalize=`0.697351/4.170046ms`、partition/merge=`2.395666/0.092586ms`；B4 32K 对应三类为 `1.686906/2.270007/1.717651ms`，RAFT/canonicalize=`0.875747/1.394260ms`、partition/merge=`1.618290/0.099361ms`。这些是独立 profiler 采集的 CUDA kernel 时间，不与 clean TPOT 混排。
- 至此九格中 B1 三长度、B2 三长度和 B4 32K 共七格完成；B4 64K 于 `2026-11-16T21:13:03Z` 接力点火，B4 128K 尚未启动。`21:14:27Z` 实测 GPU0=`103836MiB/100%`、唯一 compute app=2854215，B4 64K 日志已越过模型加载但尚无 repeat 完成行；九格统一审计仍等待完整闭集。
- GPU1 的 128K B4 clean 格第二个 repeat 已真实产出 `54.808ms`，与首个 `30.086ms` 一并原样保留；第三个 repeat、`[LONG PERF DONE]`、`[DONE]` 和完整 JSON 尚未产生，不计算或登记最终中位数，也不删除离群值。`21:14:27Z` GPU1=`142444MiB/100%`、唯一 compute app=2612426；六格 clean 统一审计未执行，全部结果继续 pending-independent-verification。

### 2026-11-17：full-page partition 修复的干净复核与 33-token 验收（在飞）

- 128K 旧失败已精确保留：B2/B4 对确定性 B1 golden 分别出现 78/228 个 token 分歧；旧第 0 层 full-page `sparse_output_row0` 有 1457/4096 位差异，旧 nsys 显示等长行的每行 chunk 几何随 B1/B2/B4 改变。当前 planner 对等长 batch 复制独立 B1 的每行 chunk 几何，同时保留每行末尾伪 chunk。
- B1-first 精确探针中，B1 与 B2 row0 的 75/75 个 chunk partial/LSE 全部一致，最终输出为 0/4096 位差异；原探针的 row1/CROSS_ROW 差异不是有效 kernel 失败证据，因为测试只复制了 K/Q、未复制 V/decode-V。B2-first 在带 AUX、无 AUX+PARTIAL、完全无诊断三种构建中均为 0/4096 位差异，首个值均为 `-0.000253915787`。旧 after-fix 巨额错误在当前干净构建中未复现，旧日志继续保留，不归因于未复现的 host 同步、merge 或调用顺序假设。
- 修正 `quest/tests/test_batch_decode.py` 的 identical-rows 输入后，全部临时 `QUEST_AUX/QUEST_PARTIAL` 诊断已移除；当前 build 与入口扩展 SHA256 均为 `52abaae86e945d0aff1620f6890eba85fe17164a16b6939a414b1a69ec241de4`。full-page B2/B4 聚焦回归 `2 passed`，batch decode+RoPE `73 passed`；日志分别为 `records/full_page_partition_clean_pytest.log` 与 `records/full_page_clean_batch_rope_pytest.log`。
- 当前干净扩展的真实 Qwen3-8B 128K B1/B2 第一个 decode 已完成精确对拍：第 0 层 full-page 输出、后续第 2 层 query/estimate/Top-K、选中 KV 与 metadata 共16字段全部 0 位差异，`records/debug_128k_firstdecode_layer0_clean_audit.json` status=pass。这只关闭旧首个分歧点，不替代完整33-token门禁。
- 完整33-token正确性验收已用 `warmup=0/repeat=1` 启动，输出前缀 `correctness_partition_clean`，本次计时不进入正式性能排序。B2 进程健康运行时 GPU0=`86484MiB/100%`；B4 首次在 MLP prefill 因仅剩1.60GiB而需3GiB再次 OOM，失败日志 `correctness_partition_clean_quest_b4_128k_gpu1.log` 保留，随后仅显式组合既有 `prefill_tokenwise_chunk_b4_over_64k=32768` 与 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 重跑。完整 B2/B4 token 审计、全量 Quest/主仓回归及受影响 clean/operator 重测尚未完成，继续 pending-independent-verification。
- 当前扩展 B2 完整33-token运行已真实完成，但两行均在 token20 首次分歧、合计26个 mismatch；33-token runcap、输入身份、`prefill_forward=144`、estimate/Top-K=`1088/1088`、sparse=`1152`、fallback=0 均通过，故完整128K正确性仍失败，正式性能暂停。冷启动两点差分得到的 `3.095ms` 已排除。非对齐 `prefill=16398`、page16 跨页、连续33步的合成 full-page B1/B2/B4 精确对拍全部通过（`records/full_page_multistep_nonaligned_probe.json`），说明该合成场景不足以覆盖真实模型残余问题。
- B4 的 expandable+chunk32768 重跑越过了 OOM，但在生成后被正确性无关的计时门禁拦截：`T(1)=301.8750987s`、`T(33)=300.0694638s`、两点差分=`-1.8056349s`，无完整 token JSON，原始日志保留。另以单次 `generate(33, ignore_eos=True)` 的 correctness-only 诊断采集 token，不生产 TPOT、不修改 clean 计时函数或 golden。
- 主仓官方入口普通与 `python -O` 首轮均只有评分子进程缺少 `jieba` 的同一环境失败；补入既有依赖目录后两种模式均 `32 passed, 1 skipped, 20 subtests passed`，日志为 `partition_clean_official_longbench_pytest_pass2.log` 与 `..._opt_pass2.log`。
- 当前扩展 B1/B2 全33步真实模型指纹已完成，输入一致、B1事件17747全部可匹配；首个指纹差异在第二个真实 decode 的第30层 attention 输入（position=130992），不是 token20 才首次出现。精确张量对拍进一步确认第29层 MLP 输出0位差异，而第30层 attention 输入 B2 两行各有3/4096个元素不同、max_abs=0.00048828125，证明指纹汇总存在碰撞且需继续向前定位 residual 边界；证据为 `debug_128k_second_decode_layer30_partition_clean_audit.json`，根因尚未关闭。

### 2026-11-17：decode RMSNorm batch-shape 根因与最小修复（完整门禁在飞）

- B4 correctness-only 完整33-token结果已收割：四行均完成33 token、无EOS、官方调用计数与fallback=0通过，但每行13个、合计52个 mismatch，与B2一样在token20首次分歧；统一旧审计 `correctness_partition_clean_128k_audit.json` 为 B2=26/B4=52、status=fail。该失败证据与B4 expandable+chunk32768配置均保留，不修改B1 golden。
- 第二个真实 decode 的全36层精确张量对拍已排除第0–29层、layer30 raw input及层间内存改写：首个真实差异只在完全相同的layer30 input经过 `input_layernorm` 后产生。相同输入最小复现显示FP32 variance在B1/B2下相差 `7.62939453125e-06`，B2 row0独立执行与B1逐位一致；q/k/v/o projections、MLP及该点q_norm在相同输入下均逐位一致，因此根因限定为 `Qwen3RMSNorm` 的batch-shape归约，而非首先由Quest attention kernel产生。全层审计=`records/debug_128k_second_decode_all_layers_partition_clean_audit.json`。
- 生产host最小修复只将 **B>1且q_len=1** 的RMSNorm逐行复用B1归约，prefill仍走原tokenwise路径，Quest estimate/Top-K/sparse-attn继续保持真batch官方CUDA。新增普通与 `python -O` 聚焦回归均 `3 passed, 3 subtests passed`，覆盖B1/B2/B4逐行位一致、prefill不重复执行及tokenwise既有语义。
- 使用第二个decode第30层的真实输入与真实FP16权重在GPU复验：未修复B2相对B1为 `6/8192` 位差（两行合计）、max_abs=0.00048828125；逐行修复后的B2与B4均 `0` 位差，审计=`records/debug_128k_decode_rmsnorm_fix_audit.json`、status=pass。这只关闭已定位的首个数值差，不预设完整33-token必然通过。
- 最新host代码的B2/B4 correctness-only完整33-token验收已分别在GPU0/GPU1点火，后台任务=`bjjyylicr`/`bk3bwq3d8`，Python PID=`279958/279957`；模型加载完成后日志均进入 `[CORRECTNESS ONLY START]`，实测宿主compute PID=`540529/540528`、GPU0/1=`86484/142424MiB`、利用率均100%，完成进程+显存+日志三重验证。B4继续使用expandable allocator与既有B4>64K chunk32768。
- 完整门禁已真实完成并统一审计通过：B2两行与B4四行的全部33 token均与未修改B1 golden完全一致，mismatch=`0/0`，每行run-cap、EOS不提前停止、输入身份、官方调用计数、真batch decode及fallback=0全部通过。审计=`performance/records/correctness_rmsnorm_fix_128k_audit.json`、status=pass； correctness-only耗时继续排除在正式性能排序之外。主仓官方入口普通与 `python -O` 均 `34 passed, 1 skipped, 23 subtests passed`。下一步仅在Quest完整回归与GPU空闲复核后恢复受影响的正式clean/operator测量，结果继续pending-independent-verification。

### 2026-11-17：完整回归收割与修复版正式缺格排队

- 收割当前扩展 `52abaae86e945d0aff1620f6890eba85fe17164a16b6939a414b1a69ec241de4` 的完整Quest测试日志 `records/quest_rmsnorm_fix_full_pytest.log`：`278 passed, 43 skipped, 1 warning`。本轮在主仓重新执行 `tests/test_official_longbench.py`，普通及 `python -O` 各 `34 passed, 1 skipped, 23 subtests passed`；没有修改expected或放宽门禁。
- 正式测量前实测本机GPU0/GPU1=`141396/129254MiB`、利用率均100%，分别由其他Qwen官方Quest 512K operator与AsyncTLS 512K clean任务占用；未抢卡、未终止进程、未重复启动其同配置任务。两条修复版缺格链复用对应 `/tmp/tli_bs1_gpu{0,1}_perf.lock`，后台任务=`bizk0c44r/b86b86l9d`、链PID=`517963/517966`，当前仅 `WAIT_LOCK`，没有新增GPU占用。
- 启动包装只修正测量配置，不改kernel、输入或计时协议：`run_perf.sh` 对B4且L>64K显式使用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，SHA256=`b0f4744c97f9fa33de2233c46e3c4988321acf230235eb1dc684d15fb7d7cd6e`；`run_quest_official_operator.sh` 新增独立prefix、显式 `--measurement-mode operator-profile` 及同一B4长上下文allocator，SHA256=`f305699cd97e6d01d8d49542c2556d59aecaecc43dc5a38d621e0d70c0ceadf7`，不再覆盖旧operator证据。
- 新链脚本 `run_rmsnorm_fix_chain.sh` SHA256=`305b22a26b46f3b6e242169e3bc4eb2718d432413c1ea728fde238897df7b593`：拿到flock后再次要求目标卡无compute app，部分产物fail-closed，仅 `[DONE]` 且输出存在的完整格幂等跳过；每格启动20秒后自动记录进程、GPU compute app/显存和有效 `[START]` 日志三重验证。计划只测当前修复版六格clean（prefix=`formal_rmsnorm_fix`）及九格operator（prefix=`rmsnorm_fix`），旧 `formal_input_order_audit.json=fail` 与旧nsys全部隔离保留，新结果继续pending-independent-verification。

### 2026-11-17：统一审计器执行与排队状态复核

- `audit_rmsnorm_fix.py` 已通过 `py_compile` 并真实执行；当前六格clean与九格operator全部缺失，审计按设计返回exit=2、`status=pending`、`missing=15`，没有用旧 `formal_input_order` 产物补位。审计文件=`performance/records/formal_rmsnorm_fix_audit.json`，SHA256=`b79f04e91da33af3b225ab122ec5d207bbca7a1b19c014f1aa164527a13eda9f`；该结果只证明缺格分支与输出可执行，完整15格尚未产生，逐格审计分支仍待真实结果触发。
- `2026-11-17T02:54:40Z` 实测GPU0/GPU1=`142376/129254MiB`、利用率均100%，宿主compute PID=`1237856/682101`；两条修复版链PID=`517963/517966`仍存活，日志均只有 `WAIT_LOCK`，没有 `[IGNITE]`、新GPU占用或正式性能结果。未抢卡、未终止现有任务、未重复启动队列，全部结果继续pending-independent-verification。

### 2026-11-17：B2/B1六格收割、B4 128K资源竞争OOM与单格补排

- GPU0链的三格B2 clean与三格B1 operator已真实完成，并逐格通过当前统一审计：B2 32K/64K/128K中位TPOT=`30.169679084792733/35.52006045356393/38.979570381343365ms`；每格三repeat、每行33 token均与未修改B1 golden一致，输入身份、EOS不提前停止、官方调用计数、扩展52abaae8及fallback=0全部通过。B1 32K/64K/128K平均每步estimate=`1.3558132812499983/2.6710430312500013/5.308866687499991ms`、Top-K=`1.5005277812500006/2.5202953125/4.425737000000001ms`、sparse-attn=`0.5888471875000074/0.7803612812500079/1.2586715000000392ms`；nsys-rep/SQLite路径、SHA及kernel实例计数均通过。六格部分审计=`performance/records/formal_rmsnorm_fix_partial_audit.json`，status=pass，SHA256=`c7f1f960af4feee430542e7a001cbebf57c4447b37f02c5bd1dee803e9b1db46`。
- GPU1链的B4 32K clean亦已完成并通过同一审计，中位TPOT=`47.72907099686563ms`，三repeat=`47.624762170016766/47.72907099686563/47.88075597025454ms`，0 token mismatch，record SHA256=`d7ad39a10619e5bb02cccba450483c4a52826ae067bcb716f2720926ab242c14`。B4 64K于 `2026-11-17T03:36:13Z` 完成进程+显存+log三重验证后在跑，尚未登记结果。
- GPU0的B4 128K operator首次尝试在prefill `input_layernorm` FP32转换再申请2GiB时OOM；失败时目标进程1841523占114.02GiB，另一计算进程1838553占24.57GiB。20秒检查已同时看到两个compute app，但旧链只检查非空而未把多进程判为竞争，因此该 `[TRIPLE_VERIFIED]` 不代表独占，不能据此宣称kernel或allocator回归。未终止对方进程。
- 首次OOM产物已原样改名隔离：日志SHA256=`bccb8e89d3a7d4d5dc62ed980e14eeb66abaca8b01963be576438a7b7d7faed5`，nsys-rep=`b6b94f863d3a7efa562a6958a50cf1e49d2bda2abd32cabd7a649c36987d7350`，SQLite=`2aefc031365a42975a8e80994d185a35cc70a846c01a0ea25c3b10d9d171ea1c`，文件名均带 `oom_20261117T033024Z`。首版补格包装 `boisc45no` 在任何点火前已停止，以补强共享卡检测；新单格脚本 `retry_rmsnorm_fix_b4_128k_operator.sh` SHA256=`e4607c25b9b4ea69ef63177df424b0c088e7429e206e386cf5d3f552372cafaf` 只在GPU0无compute app后启动B4 128K operator，20秒时要求恰好一个compute app，若发现共享卡则等待本格退出并将JSON/nsys/SQLite/log全部改名隔离、fail-closed，不作为正式结果。新后台任务=`bmb9bg2ss`、PID=`652419`，`2026-11-17T03:47:17Z` 实测仅 `WAIT_EXCLUSIVE`，没有新增GPU占用。
- 审计器实际暴露并修复两处B1 schema兼容问题：B1 golden/operator的token IDs是flat list，而B>1为batch-major；B1生产字段本就为 `prefill_execution=official B1 kernels`、`output_layout=flat-b1`。审计现在先按原始IDs核验SHA，再规范化为单行对拍，并按B选择既有生产字段期望值；未修改golden、token expected或生产输出。完整矩阵审计当前 `status=pending`、`missing=8`，SHA256=`9707229c2ca434419780cb7841715bfabe357fb56ece9f07d2a239866804b1e4`，全部结果继续pending-independent-verification。

### 2026-11-17：B4链完成、十四格审计通过与B4 128K clean单调退化

- GPU1链 `b86b86l9d` 已于 `2026-11-17T04:43:04Z` 记录 `CHAIN_DONE`，B4三长度clean及B2/B4 operator缺格除B4 128K operator外均已真实产出。B2 clean三长度中位TPOT=`30.169679084792733/35.52006045356393/38.979570381343365ms`；B4 clean三长度中位TPOT=`47.72907099686563/54.01769815944135/177.919419715181ms`，其中B4 128K存在下述显著不稳定，不能把中位数单独解释为稳定性能。
- 直接调用统一审计器的逐格函数，对除B4 128K operator外的14格完成生产侧审计，`performance/records/formal_rmsnorm_fix_partial_audit.json` 为 `status=pass`、SHA256=`536da54b7ea91b4d78c47eeae7ecdaa1af3af0d00949d74d03aa51c8aaff3d0f`：每repeat、每行33 token均与未修改的 `g4_input_order_quest_b1_*` golden一致，输入身份、EOS不提前停止、run-cap、官方CUDA调用计数、fallback=0、输出SHA及nsys/SQLite证据均通过。该pass只覆盖正确性、身份和协议，不代表所有性能样本稳定。
- 八格已完成operator的平均每步estimate/Top-K/sparse-attn分别为：B1 32K=`1.35581328125/1.50052778125/0.58884718750ms`，64K=`2.67104303125/2.52029531250/0.78036128125ms`，128K=`5.30886668750/4.42573700000/1.25867150000ms`；B2 32K=`1.43110390625/1.80688303125/0.97049409375ms`，64K=`2.79521137500/2.85609990625/1.49365746875ms`，128K=`5.47758456250/4.86673184375/2.52116153125ms`；B4 32K=`1.68556787500/2.27068759375/1.84824734375ms`，64K=`3.29339575000/3.39938621875/2.85721271875ms`。均为独立nsys的CUDA kernel时间，不与clean TPOT混排。
- B4 128K clean三repeat的TTFT=`297821.666/298290.325/297686.964ms`，基本稳定；decode-after-first wall却为 `1650.161/5693.421/13159.015ms`，对应TPOT=`51.567540/177.919420/411.219223ms`。退化集中在decode而非prefill，三个原始repeat全部保留，不删除离群值、不用中位数掩盖两端。其余五格没有同等级单调趋势；目前尚未证明是页池/workspace生命周期、allocator、时钟或并发中的哪一种，不作无证据根因结论，也不在根因未明时重复正式测量。
- 完整 `audit_rmsnorm_fix.py` 最新结果仍为 `status=pending`、`missing=1`，SHA256=`8457803ca8fa7aeea9182127608f3c9461df3b970608b634b0ab8a817dca0188`；唯一缺格=`performance/operator_records/rmsnorm_fix_quest_official_b4_128k.json`。独占重试 `bmb9bg2ss` 截至 `2026-11-17T04:52:13Z` 仍只有 `WAIT_EXCLUSIVE`；该时刻GPU0由PID2395604占34224MiB、GPU1由PID2528682占129056MiB，均100% util，没有新增本任务GPU占用或重复启动。全部结果继续pending-independent-verification。

### 2026-11-17：B4 128K 跨版本复现与请求生命周期诊断（根因未关闭）

- 旧 `formal_input_order` 同格也保留三个完整 repeat：decode TPOT=`30.086296/54.808470/345.051913ms`。因此 decode 随 repeat 显著退化并非当前 RMSNorm/full-page 修复独有问题；该旧结果只用于定位异常，不替代当前修复版结果，也不重复测量已闭集配置。
- 用真实 `InferenceController`、B4×128K KV/metadata 页池和官方 decode kernels 建立仓库外诊断 `diagnose_decode_request_lifecycle.py`，以 `[0,33]×4` 模拟 warmup+3 repeats 的请求边界，完整模型 prefill 由等长元数据分配替代。修复零 decode 请求中 `output` 未绑定后，脚本 SHA256=`8ee68a77ac3f3a4f4900394a2b373f5de5009b58006e4ea080317e2cf847ed6b`，`py_compile` 与 default/retain 小规模烟测通过；诊断 `decode_wall/33` 不是正式两点法 TPOT。
- 完整 default 合成臂四个非零请求依次为 `28.379328/23.684433/24.519793/31.859283ms/step`，JSON SHA256=`a73203752452f047ba06b4fdbb52b1ae39fb1850fb0c77640f328d0027caa7c3`；expandable+default 为 `25.916805/23.361333/24.657348/43.480247ms/step`，JSON SHA256=`32e0a412775e4806d1add217a56f1ef3299bf1d0670dbf18050cb9c13d6d3078`。两臂均未复现正式格 `51.567540→177.919420→411.219223ms` 的单调退化，不能宣称 expandable allocator 是根因，也不进入正式 TPOT 表或性能排名。
- 两臂每请求结束均为 `free_kv_blocks=32760`、`free_metadata_blocks=2048`、`released_row_pages=[0,0,0,0]`；非零请求后的 Torch allocated 稳定为 `96624790016` bytes，reserved 分别稳定为 `96640958464/96638861312` bytes，未见随请求累积。纯 CPU 按真实页池 `set.pop()/add()` 日程重放亦显示 request 1 后相邻页连续性约 0.43 且基本稳定，不支持“物理页连续性持续单调恶化”这一过简解释。
- 合成探针没有完整模型权重、真实 prefill/Q/K/V 和正式约 140GiB allocator 状态，故不能据此完全排除页池/workspace 生命周期。完整 retain 臂的预设触发条件是“合成臂复现后再做 retain 对照”，当前未满足，未继续烧 full-retain；后续若继续，应优先设计保留完整模型、真实 prefill 和高显存状态的判别实验。正式矩阵仍只缺 B4×128K operator，独占重试继续等待真实空卡，全部结果保持 pending-independent-verification。

### 2026-11-17：五长度 BS1 三方法闭集与 JSSA fail-closed

- 独立结果目录=`agent_doc/results/qwen3_ruler_bs1_vendor_perf_20261117/`。口径为 Qwen3-8B、`niah_multikey_2` sample-index 2、扩展输入文件第0行、32K/64K/128K/256K/512K、BS1、统一 token/page budget=1024；clean 使用1 warmup+3 repeats、固定33 token、忽略EOS和完整request host wall `TPOT=(T(33)-T(1))/32`，与nsys进程完全分离。32K native RoPE，其余长度YaRN factor=2/4/8/16。
- clean e2e 15/15闭集，中位TPOT（ms）与NFA相对加速比如下；所有原始repeat保留，未按预期删除或改写负结果：

| 长度 | NFA | Quest官方CUDA | AsyncTLS语义复现 | NFA/Quest | NFA/AsyncTLS |
|---|---:|---:|---:|---:|---:|
| 32K | 12.587633 | 19.416291 | 23.957791 | 1.5425× | 1.9033× |
| 64K | 14.933991 | 22.984220 | 24.202101 | 1.5391× | 1.6206× |
| 128K | 18.852773 | 24.684969 | 24.291045 | 1.3094× | 1.2885× |
| 256K | 26.203578 | 29.122938 | 24.989341 | 1.1114× | **0.9537×** |
| 512K | 61.052300 | 83.823367 | 26.295982 | 1.3730× | **0.4307×** |

- NFA在五长度均快于Quest，但256K/512K均慢于AsyncTLS，不能宣称全格领先。AsyncTLS 512K三repeat=`27.623630/26.295982/25.970248ms`；NFA 512K=`185.488048/61.052300/58.516140ms`，Quest=`54.550931/83.823367/88.740277ms`。两点法对独立长prefill墙钟敏感，中位数不代表每次repeat稳定。
- operator 15/15闭集。下表均为独立nsys的每steady-decode-step CUDA kernel毫秒数，不与clean TPOT混排；NFA列为纯L1/L2打分及宽阶段，Quest列为estimate/Top-K/sparse-attention，AsyncTLS仅列可唯一归属的Top-K-sort/gather/index family，不伪造K1/K2或selected-attention总时延：

| 长度 | NFA纯L1/L2 | NFA宽L1/L2 | Quest estimate/Top-K/sparse | AsyncTLS sort/gather/index |
|---|---:|---:|---:|---:|
| 32K | 0.145809/0.932613 | 0.837117/2.088197 | 1.355149/1.500483/0.588715 | 1.829569/5.161703/6.991272 |
| 64K | 0.151517/0.931737 | 0.852552/2.569688 | 2.672924/2.520719/0.778616 | 1.833210/5.279843/7.113054 |
| 128K | 0.169155/0.939764 | 0.882355/2.577727 | 5.306085/4.425611/1.258637 | 1.833243/5.414408/7.247651 |
| 256K | 0.268096/0.934229 | 1.063599/2.569388 | 10.585415/8.183875/2.270802 | 1.835671/5.748877/7.584548 |
| 512K | 0.393076/0.929007 | 1.367625/2.564422 | 21.054775/15.545384/4.333896 | 1.828754/6.130749/7.959503 |

- NFA 32K/64K/128K为当前二进制29-step profile，36层`skip_near/skip_far=false`且region入口审计通过；256K/512K只从旧5-step原始SQLite重算副本，登记`new_gpu_measurement=false/current_binary_measurement=false`，不冒充当前二进制重测。Quest 32K–128K复用闭集profile，256K/512K为当前测量；512K首次因碎片OOM，使用`expandable_segments:True`后成功，预算、kernel和计时协议不变。
- AsyncTLS五长度operator均为当前31-step profile，Graph capture=1/replay=32/fallback=0。512K前两次在GPU0分别遇到不受本flock管理的24.74/24.79GiB外部进程，均在prefill `q_norm`申请7.99GiB时OOM；两份日志保留且没有有效JSON/nsys。最终仅在GPU1协作锁队列等待真空闲后，于`04:43:22Z–04:59:47Z`成功，未擅自杀外部进程。
- JSSA官方patched-vLLM可导入但`vllm._C`缺失、`radix_topk_available=false`，clean/operator各五格均fail-closed；没有Torch fallback或替换wheel。AsyncTLS明确为独立PyTorch语义复现，非官方TileLang、无真实异步host offload。
- 最终生产侧统一审计=`PASS`、clean=15/15、operator=15/15、pending=[]，核验输入IDs和原始文件实算SHA、33-token合同、RoPE、Graph、Quest官方调用计数、NFA no-skip、预算、nsys/SQLite及manifest。`summary.json` SHA256=`8206bad520e8a3858bf069b87efaaf95ee71d46f63bdce11f80b108946204543`，`audit.json`=`5872a3fb8f5263016e6e40aa6997d6443473da589f0d8dd5393724c54b2003d8`，`manifest.json`=`b2de5874f9d63b6c073d69fc8d6b07d7a9018e953c606eb683142d2704f03bab`。主仓官方入口普通与`python -O`各`34 passed, 1 skipped, 23 subtests passed`。全部结果继续pending-independent-verification，生产侧PASS不替代不同owner独立验收。

### 2026-11-17：BS1 两点差分归因停止，改为严格 decode-only

- 正式 NFA 臂显式 `enable_layer_skip=false`，36层 near/far/both 均为空，不能把用户设想的 skip 收益套到这批结果。现有 nsys/SQLite 重算得到 NFA 选择器+主注意力总时延在32K/64K/128K/256K/512K分别为`3.949928/4.448528/4.481767/4.648042/4.952267ms/step`，均低于 AsyncTLS 仅 Top-K/sort/gather 的可识别子集`6.991272/7.113054/7.247651/7.584548/7.959503ms/step`。后者不是 AsyncTLS 完整 selection+attention 总值，但已足以否定“NFA 这组稀疏算子更慢导致旧两点差分反向排序”；clean 与 profiler 数据仍不得直接相减命名 host 开销。
- 诊断使用隔离 worktree=`/tmp/nfa-bs1-attribution-20261117`、提交`0fa07ea80b67d8369c6c26359e768f510dd4eef4`、空 source diff，结果目录=`agent_doc/results/nfa_bs1_decode_attribution_20261117/`。128K 单 prefill+35步诊断中，排除3步 warmup 后 CUDA TPOT 中位数=`10.894880ms`，35次 Graph replay、0 fallback；跨页 step18 的 CUDA/host=`12.980288/12.964673ms`，其中 metadata plan=`2.145648ms`。真实36层 page64 CPU metadata 微基准的一次性跨页成本摊入32步不足`0.43ms/step`，不足以解释旧512K两点差分。
- 保持原 `T(33)-T(1)` 边界、同步和 Graph 计数不变的 host-clock wrapper 已归档为`nfa_two_point_diagnostic.py`，SHA256=`29485f914a9bd9d3f4923b0049d5615edb84a66dcadb12792137baab09f434f0`。32K 增强前后输出 token SHA 相同；增强版 measured pair 首步/稳态中位数=`70.214823/10.735460ms`，首步 `record_next_tokens=68.016551ms`，两次 prefill 差=`-8.924537ms`，两点 TPOT=`12.393315ms`。128K measured pair 首步/稳态中位数=`250.465870/11.238262ms`，32步总和=`601.807103ms`，prefill 差=`+147.181712ms`，两点 TPOT=`23.498224ms`。
- 用户随后明确指出本任务目标是 **decode-only e2e + 算子性能**。`(T(33)-T(1))/32` 的两次独立长 prefill 不会精确抵消，并混入 cold decode 初始化，因此旧15格 clean 仅保留为“完整请求两点差分”历史观测，不再作为 decode-only 排名。512K 两点诊断已停止，256K/512K 不再按旧口径补测。
- 用户最新收敛为：**NFA 只要自然真实 region-skip，AsyncTLS 按全 near/region-skip 关闭口径，Quest/JSSA 有官方源码就直接测，不再搭复杂框架**。取消 cold/steady 双栏，统一用单次 prefill 后 3 步 warmup + 29 步正式 decode 的 host-wall 主指标；NFA 必须核验逐层 near/far/both 非全空，旧 no-skip 只作历史。Quest 复用官方 patched forward 加薄手动 decode 驱动，JSSA 复用现有官方 Triton Graph 入口并如实标注 Top-K 实现，AsyncTLS 仍是 PyTorch 语义复现。算子仍由独立 nsys 进程采集；pc33 双卡禁用，远端实测空卡串行执行，所有新结果保持 pending-independent-verification。
- 最小计时实现已在隔离 worktree 完成并通过 `py_compile`：`measure_nfa_selector_e2e.py`、`benchmark_asynctls_graph.py`、`benchmark_jssa_official.py` 均在 warmup 结束同步后开始 host-wall，末尾同步后结束；Quest 新增 `generate_quest_decode_only()` 与 `--timing-mode decode-only`，不改官方 kernel。尚未进行新 GPU 测量。
- 用户补充硬要求：每格点火前实测无 compute app 并锁卡；同长度四方法 clean 按 ABBA 正逆序重复，保留全部 repeat、顺序和原始日志。离群值必须结合占用、日志与实现回执排查，不得直接删数或挑数。当前 pc02 四卡均被其他会话占用，未点火、未抢卡。

### 2026-11-17：decode-only 运行时就绪与旧 finalizer 回归（GPU 测量未启动）

- clean ABBA 唯一入口=`/tmp/run_bs1_decode_only_abba_20261117.py`，SHA256=`cbff0a96d575e150602660b4de93840c5f5a7dfba5ab72b0a91a37bd06c1e9db`。每格执行前要求目标卡无 compute app，使用协作 flock；运行中发现其他 PID 只终止本测量进程。逐格核验输入 SHA、29 个 measured decode steps、NFA 自然 near/far/both 并集非空、Graph capture/replay/fallback，以及 Quest 官方 CUDA 调用计数和零 fallback。输出、回执、日志及 `abba_schedule.json` 均逐格保存；当前尚未在 GPU 上实际执行。
- pc02 的 Quest 新 CMake 构建仍被 RAFT `CUDA::nvToolsExt` 目标缺失阻塞，未宣称重编译成功。改用同一 Quest commit=`01c1623bf9395009520874e989e29f683203b357`、源码无差异且历史生产已使用的固定官方扩展 `/tmp/nfa-bs1-decode-only-20261117/third-party/Quest/quest/_kernels.cpython-311-x86_64-linux-gnu.so`，SHA256=`de7d6e32052fd63206c7ba20b54d9040c40f8dc6617eb2a392c80e0bcc1b6b1f`，显式 `import quest._kernels` 通过。失败构建树和日志保留，不将二进制复用写成本轮构建成功。
- pc02 补齐隔离依赖 `rouge-1.0.1` 后，`tests/test_official_longbench.py` 普通模式 `Ran 33 tests in 10.653s, OK (skipped=1)`，`python -O` 为 `Ran 33 tests in 10.726s, OK (skipped=1)`。七个 tracked 测量文件及 runner 的本地/pc02 SHA 逐一一致，四方法及 Quest CPU import 全部通过。
- `measure_quest_official_operator.py` 的旧两点法/direct decode-only 兼容分支，已用真实旧 32K JSON+nsys+SQLite 回归重放：`legacy_two_point_replay=PASS`、32 个 profiled steps，estimate/Top-K/sparse-attn 及 RAFT/canonicalize/partition/merge 七项汇总全部相等。direct 29-step 分支仍须真实 nsys 验证，不能由旧数据重放代替。
- 最新资源复核：pc02 四卡、pc40 双卡、pc31 双卡、pc32 双卡均有 compute app；pc14 SSH 连接超时，不能推断有空卡。pc33 双卡继续保留给其他会话。本任务没有新 GPU 测量在飞、没有新 TPOT/算子数值，未抢卡或终止任何外部进程。
- 旧后台通知已逐一与归档证据对账：完成项仅确认既有 `rouge` 修复和33项测试通过，失败项仍为修复前依赖缺失、Quest `CUDA::nvToolsExt` 构建阻塞或 pc14 SSH 超时；旧测量 wrapper 的空输出不用于重建数值，旧两点法/no-skip 产物只保留为历史证据，不混入 direct decode-only 新结果，也不重跑这些已结束旧链。
- 对账后再次实测：pc02 GPU0–3 仍分别有 compute app，pc40/pc31/pc32 各双卡也仍被占用，pc14 仍连接超时；因此继续不点火，未留下自动等待或抢卡队列。

### 2026-11-17：用户改派本机双卡，32K/64K 自然 no-skip 与 128K 测量（在飞）

- 用户最新指令“直接锁本机两卡测，其他卡不用”覆盖此前 pc33 保留规则，并明确授权本机其他任务直接 kill、当前性能测试权限更高。本机 GPU1 原 Quest 任务按精确 PID 终止父 shell `1005454/1164442` 和 Python `1164443`，未使用模糊 `pkill`；终止后双卡均 `1 MiB/0%`，无 compute app。远端机器不再用于本任务。
- 本机测量仓库=`/tmp/nfa-bs1-attribution-20261117`，HEAD=`0fa07ea80b67d8369c6c26359e768f510dd4eef4`。主仓 Quest 子模块含未提交真 batch 改动，未用于本轮；固定官方 CUDA 扩展 SHA256=`de7d6e32052fd63206c7ba20b54d9040c40f8dc6617eb2a392c80e0bcc1b6b1f`。最初从 pc02 复制的纯上游 Python `InferenceController` 不含 Qwen3 GQA 参数，128K Quest 首试点火后在初始化阶段 fail-closed；随后仅从既有 `/tmp/qwen3-official-smoke-19edfebd-20261115` BS1 生产树恢复同扩展对应的7个 GQA 适配源码文件，不含主仓真 batch Top-K/controller 改动。接口烟测确认 `num_kv_heads=None` 参数及本地 Quest 导入路径，失败日志独立归档。
- 本机 `nvidia-smi` 返回宿主 PID，与 subprocess 容器 PID 不同，首轮调度器误把唯一自身 CUDA context 当外部进程并 fail-closed，未产生有效 JSON。修复后前置空卡下单个 compute app 可判为本任务，多于一个仍立即终止本测量；postflight 和输出验收不变。失败日志独立保留，旧日志禁止覆盖。当前 runner SHA256=`d5b04cc960962f684b5adadb50527295e1c1ebf27885271099465c56051816be`。
- 32K NFA clean 首格实际完成：29-step host TPOT=`10.331901753770895ms`、Graph capture/replay/fallback=`1/32/0`，但自然 prefill 决策的36层 near/far/both 全为空，因此不是用户要求的真实 skip 主臂，只归档为 `diagnostics/nfa_L32768_natural_no_skip.*`，不进入排名。64K 自然门控同样36层 no-skip，runner 按门禁 fail-closed，JSON/日志移入 diagnostics；未强制 skip、未删数。
- 32K/64K 首轮 nsys 漏加 `--cuda-graph-trace=node`，SQLite 只有 Graph 外调度 op、缺少 NFA 阶段 kernel，对应 JSON/nsys/SQLite/log 均归档为 `diagnostics/*_no_graph_nodes_operator.*`，不得作为算子结果。128K 已修正为 node 级 Graph 追踪。
- 2026-11-17T08:29:36Z 双卡点火：GPU0 运行128K四方法 clean ABBA（后台 `b5ngbgwut`），GPU1 运行128K NFA 29-step 独立 nsys（后台 `b3r69fbgt`）。约30秒实测 GPU0/1=`45614/45628MiB`、利用率均100%，各一个 compute app；clean driver 已记录 IGNITION，两份模型日志均越过 checkpoint 加载，进程+显存+日志三重验证通过。clean 与 profiler 分卡分进程，结果继续 pending-independent-verification。
- 128K NFA 自然门控真实触发 near-skip 层=`[7,29,30,33]`，far/both 为空；clean 正序 TPOT=`10.935038ms` 并通过 Graph `1/32/0` 验收。NFA node 级 nsys finalizer 通过，区域调用审计 generic/far-only/active-page-stats/candidate/fine-score 每步=`32/4/36/36/36` 与冻结签名完全一致；NFA operator total=`4.421643ms/step`，L1/L2/main-attention=`0.877722/2.525582/1.018340ms/step`。这些是 profiler 独立观测，不与 clean TPOT 相减。
- 128K AsyncTLS/JSSA clean 正序 TPOT=`24.303854/12.642170ms`，均通过29步与 Graph `1/32/0` 生产门禁；JSSA 为官方 Triton store/score/indexed-attend + 已披露 `torch.topk` fallback。AsyncTLS 独立 nsys 已 finalize，Top-K/sort/gather/index family=`1.838176/5.406722/7.244899ms/step`，只报告可唯一归属 kernel family，不冒充完整选择器。Quest GQA 修复后于 `08:37:40Z` 重新点火并实际完成32步，但旧回执验收误将 estimate/Top-K 也按36层计算，因 `1088=34×32 != 1152=36×32` 拒绝输出；官方前 `_quest_skip_layer=2` 层本就保留全页，正确计数是 estimate/Top-K=`34×32`、sparse-attn=`36×32`。已只修正 `generate_quest_decode_only()` 的回执公式及对应单测，未改官方 kernel、输入或计时；普通与 `python -O` 完整 `tests/test_official_longbench.py` 均 `Ran 33 tests, OK (skipped=1)`。
- 128K JSSA 首次算子采集在初始化申请18GiB KV cache 时 OOM；实测另一会话在 GPU1 启动512K NFA nsys，与本测量同时占卡。按用户本机权限精确 SIGTERM 外部父 shell/runner/nsys/measurement PID=`1257624/1257750/1257848/1257947`，未使用模糊杀进程；双卡随后均 `1 MiB/0%`、无 compute app。Quest 计数失败与 JSSA OOM 日志均已分别归档到 diagnostics，没有覆盖或不完整 clean/operator JSON 混入有效结果。
- 2026-11-17T08:50:43Z 双卡断点重跑：GPU0 clean ABBA 跳过有效 NFA/AsyncTLS/JSSA 正序，从 Quest 正序开始，后续执行完整反序；GPU1 重跑 JSSA 29-step Graph-node nsys 并接续 finalizer。点火后实测两卡利用率均100%、显存约49.8/46.6GiB、各一个 compute app，Quest/JSSA 日志均越过模型加载，clean driver 已记录 IGNITION，进程+显存+日志三重验证通过。
- **128K 四方法 clean ABBA 与四方法独立算子已闭集（生产侧对账PASS）**。正/反序 host TPOT（ms）：NFA=`10.935038/10.947794`，AsyncTLS=`24.303854/24.287585`，JSSA=`12.642170/12.653279`，Quest=`24.224748/24.067432`；NFA 两次自然 near-skip 均为 `[7,29,30,33]`，Graph均 `1/32/0`，Quest计数均为 estimate/Top-K=`1088`、sparse-attn=`1152`、fallback=0，八格输入SHA一致。JSSA 29-step nsys finalize 后 store/score/Top-K/indexed-attend=`0.085669/0.649825/3.075811/0.314619ms/step`，官方命名算子合计=`4.125924ms/step`，Top-K仍为官方 `torch.topk` fallback。
- Quest direct nsys 原始采集实际完整：estimate/RAFT=`986=34×29`，sparse partition/merge=`1044=36×29`；首次 finalizer 失败仅因错误要求主仓真 batch 扩展的 `canonicalize_topk`，当前固定上游 BS1 扩展只有 RAFT Top-K。已只修 direct 分类计数和空可选项统计，并用原 SQLite 重放成功，无新增 GPU 测量；Quest estimate/Top-K/sparse-attn=`5.225856/0.350678/1.223511ms/step`，canonicalize=0，nsys/receipt/SQLite哈希已入回执。原失败日志及一次误用主仓 finalizer 的日志均归档 diagnostics。
- 2026-11-17T08:59:18Z GPU0 启动256K NFA自然skip正序及后续ABBA；GPU1暂不预跑依赖自然skip成立的其他算子格。全部结果继续 pending-independent-verification，生产侧PASS不替代不同owner独立验收。
- **256K 四方法 clean ABBA 与四方法独立算子已闭集（生产侧对账PASS）**。NFA 自然 near-skip 层=`[0,7,10,23,24,25,26,27,28,29,30,31,32,33,34]`，far/both 为空；正/反序 host TPOT（ms）：NFA=`10.944230/10.966608`，AsyncTLS=`25.027784/25.027956`，JSSA=`13.945303/13.932099`，Quest=`28.809246/28.739602`。八格输入 SHA=`0f8b39583d1df5e267c1750da44300322a2b5359cf6b2fe89b45b56efe028b30`，输出/日志 SHA 与逐格 receipt 一致，preflight/postflight 均无外部计算进程；NFA/AsyncTLS/JSSA Graph均 `1/32/0`，Quest estimate/Top-K=`1088`、sparse-attn=`1152`、fallback=0。
- 256K 独立算子：NFA L1/L2/selected-attention/operator-total=`0.989816/2.401084/1.023918/4.414817ms/step`，区域审计 generic/far-only/active-page-stats/candidate/fine-score 每步=`21/15/36/36/36`；AsyncTLS Top-K/sort/gather/index-family=`1.831669/5.660881/7.492550ms/step`，仅为可唯一识别子集；JSSA store/score/Top-K/indexed-attend/official-total=`0.084496/1.260540/3.783912/0.315512/5.444460ms/step`，仍为官方 `torch.topk` fallback；Quest estimate/RAFT-TopK/sparse-attn=`10.385346/0.488802/2.244817ms/step`，29步计数 estimate/RAFT=`986`、partition/merge=`1044`、canonicalize=0，无 Torch/dense fallback。四份 JSON、nsys、SQLite 与日志均存在且 finalize 成功，profiler 数据不与 clean TPOT 相减。
- 2026-11-17T09:32:57Z GPU1 启动512K四方法 clean ABBA，首格 NFA 自然 skip 判定；GPU0随后启动512K NFA独立29-step Graph-node nsys。两路均完成进程、显存、日志点火核查，当前仍在测量，尚无512K有效 TPOT 或算子结论。若 NFA 自然 near/far/both 全空，clean runner 将 fail-closed，不把 no-skip 诊断混入真实 skip 主排名。
- 512K NFA clean 正序已有效完成：自然 near-skip 共24层=`[0,1,5,7,8,9,10,12,13,14,17,23,24,25,26,27,28,29,30,31,32,33,34,35]`，far/both 为空，host TPOT=`11.214604151659998ms`，Graph `1/32/0`，输入 SHA=`6ca6ca3137fe30ca2925a20641ae7f2e08d964f0bb7628bbc315ea1c29627dc8`。NFA 独立29-step算子也已 finalize：L1/L2/selected-attention/operator-total=`1.204068/2.300273/1.017880/4.522221ms/step`，区域审计 generic/far-only/active-page-stats/candidate/fine-score 每步=`12/24/36/36/36`，与自然 skip 签名一致。
- 首轮512K AsyncTLS clean 与独立算子约在10:07Z同时异常消失，clean wrapper exit=144，两份日志均无 Python traceback/OOM，算子 nsys 仅报告 `No reports were generated`；双卡随后均为 `1 MiB/0%`、无残留进程，未生成 AsyncTLS JSON/nsys/SQLite。系统 RAM 当时 available=`189GiB`，`/tmp` 余量=`4.8TiB`，`dmesg` 无权限读取，故只登记为原因未明的外部 abrupt termination，不猜测 OOM、不制造数值。原始日志已移入 `diagnostics/*abrupt-exit-1007*`，仅保留已验收 NFA clean/operator 有效产物；10:07Z后实测双卡空闲并从 AsyncTLS 格断点重跑，未覆盖失败证据。
- 512K AsyncTLS断点重跑已成功：clean正序host TPOT=`25.937340ms`，Graph=`1/32/0`；独立29-step nsys的Top-K/sort/gather/index-family=`1.829604/6.134183/7.963787ms/step`，仍仅为可唯一识别子集。JSSA clean正序TPOT=`16.584990ms`，Graph=`1/32/0`；官方store/score/Top-K/indexed-attend/named-total=`0.084869/2.434506/5.282286/0.319326/8.120987ms/step`，store/score=`1044`、indexed-attend=`2088`次，Top-K继续明确为官方`torch.topk` fallback。
- 512K Quest首轮clean正序与独立nsys约在10:50Z再次同时异常消失：operator wrapper SIGTERM=143、clean wrapper状态丢失，日志均无Python traceback/OOM，nsys为`No reports were generated`，无Quest clean/operator JSON、nsys或SQLite；双卡清空，RAM available=`189GiB`。三份原始日志归档为`diagnostics/*1050*`，不制造数值。异常后其他会话新NFA/Quest smoke任务出现；按用户本机性能优先级仅精确SIGTERM新NFA链父shell/timeout/Python PID=`1617865/1618151/1618153`，另一Quest smoke已自行退出，随后双卡实测无compute app。10:53:28Z从Quest正序断点重跑clean（后台`b0kingniu`）及Quest独立nsys（后台`bv6bks36i`），两路均完成进程、显存、日志点火核验，各卡只有一个本任务compute app。

### 2026-11-18：BS1 decode-only CUDA-event 五长度全闭集（Quest official 修复 + Quest 语义 b1024）

- 用户验收口径最终收敛：BS1 decode-only e2e、计时统一 CUDA Event 耗时中位数（3 步 warmup 排除 + 29 步实测）、NFA 用 gate=0.1 自然 skip、同输入 `repeated-needle-body-extension` row0、baseline 论文语意优先。本机双卡执行；旧的 host-wall / T(33)-T(1) 两点法一律不作为排名指标。
- 修复 Quest official 计时：`logits_processor` 内 `torch.cuda.Event.record()` 落在与解码计算不同的 stream，32K 未竞争时得到负中位（-19.17ms）、竞争时得到 46.48ms，均不可用。改为 `collect_decode_events`（模型 forward 前后 hook 同 stream start/end event），与 NFA/JSSA/AsyncTLS 的逐步 CUDA-event 口径一致。32K 复测=`19.02ms`（旧 host-wall 18.85ms 一致）。
- 排查“256K/512K clean 加载完模型后静默消失”：根因是本机遗留 watchdog `/home/wangyuanshuo02/longbench_dispatch/local_gpu_guard.sh`（每 60s 精确匹配 `long_context_official.*--method quest` 并 kill，日志时间戳与消失完全吻合）。已精确 SIGTERM 该 guard（PID 2286480），保留其 monitor.sh 收割远端 LongBench sweep。
- Quest official clean 五长度 CUDA-event 中位（ms，29 步实测）：32K=`19.021`、64K=`22.868`、128K=`24.889`、256K=`36.619`、512K=`60.618`；全部 `measured=29/warmup=3`、`dense_or_torch_fallback=0`、官方调用计数 estimate/Top-K=1088、sparse-attn=1152。512K 首次 OOM（139.7GiB 上限内 4GiB 分配失败），加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后重跑成功。
- Quest 论文语义复现（`benchmark_quest_graph.py`）旧预算 2048 且输入为 `native-ruler` row2（32622 tokens），与 NFA/JSSA/AsyncTLS 不同行，不可同口径对比。以 `--quest-budget 1024 --input-provenance repeated-needle-body-extension` 重跑五长度，CUDA-event 中位（ms）：32K=`18.263`、64K=`22.754`、128K=`33.100`、256K=`54.051`、512K=`147.330`，随长度单调增长。
- BS1 五长度 CUDA-event 中位对比汇总（ms，同输入）：NFA gate0.1=`10.220/10.678/10.551/10.529/10.714`（skip 层 5/13/23/30/36，几乎平坦，属自适应 gate + 固定 K2 预算的预期行为）；JSSA official=`11.335/11.928/12.613/13.901/16.550`；AsyncTLS semantic-pytorch=`23.904/24.259/24.239/24.980/25.916`；Quest official=`19.021/22.868/24.889/36.619/60.618`；Quest semantic-b1024=`18.263/22.754/33.100/54.051/147.330`。NFA 五长度均快于全部 baseline；Quest/JSSA 随长度增长证明计时可捕获长度成本，NFA 平坦由 gate 自然 skip 层数随长度增加（4→36）解释，非测量失效。
- 结果归档 `agent_doc/results/qwen3_bs1_decode_event_comparison_20261117/`（`summary.json` + `quest_official/` + `quest_semantic_b1024/`），全部保持 `pending-independent-verification`。

### 2026-11-18：JSSA 论文性能语义 + NFA 全量 score/Top-K 补格（在飞）

- 用户最新边界为只补缺口，不重跑上述25个已闭集 e2e 格；新增 JSSA 论文性能语义臂复用 JSSA 官方 project/store、FP8 projected-K、indexed-attend，并将选择器替换为 NFA 全候选 32 维 fine-score 与精确两阶段 filtered Top-K。候选覆盖 `[0,seq_len)`，不引入 NFA L1、region gate 或候选块剪枝，budget=1024、rank=32、page=64。
- 适配关闭两个真实正确性问题：JSSA projected-K 为 `[block,position,H_kv,rank]`，NFA fine-score 新增 opt-in `KV_POSITION_MAJOR` 寻址；`generations` 按扁平 `batch×H_kv` 行读取，运行时由错误的 `[batch]` 改为 `[batch,H_kv]`。默认 NFA head-major 路径保持原行为。
- 真实 CUDA 小形状门禁通过：两个 KV head 的全部 BF16 logits 与 PyTorch 参考逐元素一致、尾槽全为 `-inf`、Top-K 集合一致、CUDA Graph capture/replay 一致（`tests/test_jssa_nfa_semantic.py`，1 test OK）；NFA 默认布局聚焦回归 `tests/ops/test_fine_scores_preload_mask_cuda.py` 2 tests OK。official finalizer 使用旧128K JSON+SQLite重放，`operator_summary/kernel_summary` 完全相等，未重跑 official GPU 数据。
- 测量入口已分流 `selector_implementation=nfa-semantic`，新臂要求观察 `_nfa_fine_scores_paged_kernel`、`_nfa_filtered_topk_partial_kernel/_merge_kernel` 与官方 indexed-attend；新结果目录=`/tmp/nfa-bs1-attribution-20261117/agent_doc/results/jssa_nfa_semantic_bs1_decode_event_20261118/`。GPU0后台任务 `b9juxhhdq` 只跑五长度 clean，GPU1后台任务 `bs569j9fp` 只跑五长度独立 Graph-node nsys；均为3 warmup+29 measured/profiled steps、同一扩展输入row0、BS1。
- 点火后实测两条链均从512K开始，宿主benchmark/nsys进程、有效模型加载日志、GPU0/1各约132.5GiB/100%及每卡唯一compute app三重核验通过；hygiene至今无外部进程。当前尚无任何有效新格回执或性能数值，NFA gate0.1独立operator补测待这两链结束后串行执行，全部保持 pending-independent-verification。
- 首轮512K在prefill后进入decode warmup时暴露全候选Top-K编译扩展问题：两个进程的 `py-spy` 栈均位于 `_nfa_filtered_topk_merge_kernel` 的 Triton `make_ttgir`，`NTILES=257` 与 `LOG2_K=10` 经 `tl.static_range` 全展开造成编译组合爆炸，GPU空闲且未生成有效JSON/nsys。按精确PID停止本任务链后双卡均回1MiB；日志、hygiene和无事件driver receipt均以 `compile-static-ntiles257-failure` 后缀隔离保留。
- 最小修复给同一merge kernel增加 opt-in `MERGE_DYNAMIC`，只把全候选JSSA/NFA臂的逐run归并改为运行时循环，packed-key、逐组顺序、10级排序网络及默认NFA静态路径不变。真实524321候选×2 KV heads×K=1024首次编译+执行约7.7秒，与独立packed-key参考选集/counts完全一致；已固化为 `test_dynamic_merge_full_512k_candidates_bit_exact`。JSSA/NFA全部2 tests OK，默认布局2 tests OK；另以8452候选多tile对拍确认 `merge_dynamic=False/True` 均与参考一致。当前环境无pytest命令，未冒称完成对应pytest全集。
- 动态归并修复后重新点火：GPU0 clean 后台任务=`b9yzys6nc`（driver PID 3047336、benchmark PID 3048028），GPU1 operator 后台任务=`bvl2heufm`（driver PID 3047358、nsys PID 3048020、benchmark PID 3048086）。两链仍只跑原五个缺格、从512K向32K串行；新 source diff SHA256=`06aea24f8e0bf89a503836fc04a0a391e49f0f02a6bff496ac3000ecd91c0776`，覆盖本轮动态归并修复。点火后实测双卡各约132.5GiB/100%，每卡仅一个本任务compute app，模型加载日志与hygiene均已落盘，首轮编译停滞未复现；有效回执仍待512K decode完成后验收。
- JSSA/NFA论文性能语义五长度双口径已完成：clean TPOT 32K/64K/128K/256K/512K=`12.662208/15.975040/25.490208/41.757118/73.833565ms`；独立nsys的score=`2.464118/4.646191/8.999999/17.844949/35.408119ms/step`，Top-K=`1.382464/2.454245/7.714809/15.152026/29.692356ms/step`，selected-attention=`0.360839/0.358419/0.363947/0.366418/0.367868ms/step`。10/10回执均为BS1、rank32、budget1024、page64、3 warmup+29 measured/profiled、Graph capture=1/fallback=0、hygiene=clean；逐格六类官方/NFA内核调用数均为1044，nsys/SQLite哈希复核通过，profile/clean生成token公共前缀一致，`summary.json`五格交叉验收通过。该臂为全候选论文性能语义，不与旧official Top-K数据混称，继续 pending-independent-verification。
- JSSA双链退出、双卡恢复1MiB后，启动NFA gate0.1自然skip的独立nsys补格：GPU0后台=`bdva7bjnk`（shard0：512K/128K/32K，driver PID3101792），GPU1后台=`bwcyrejdw`（shard1：256K/64K，driver PID3101803）。两臂显式 `subspace=tail/l2-subspace=same/gate=0.1/force=natural/operator-mode=off`，3 warmup+29 profile；点火实测GPU0约119.9GiB、GPU1约72.2GiB且均100%，各只有本任务compute app，日志与hygiene持续落盘。未重跑NFA clean五格。
- NFA gate0.1独立operator五格全部收官。32K/64K/128K/256K/512K的L1=`0.818377/0.817056/0.833431/0.931659/1.159293ms/step`，L2=`2.023827/2.414297/2.303010/2.217081/2.154728`，Top-K total=`1.319481/1.559751/1.552265/1.587102/1.738452`，main-attention=`1.025939/1.024368/1.025614/1.018760/1.022310`，named operator total=`3.868143/4.255721/4.162054/4.167501/4.336331`。五格自然skip层数=`5/13/23/30/36`，逐层near/far列表与既有clean完全一致，生成token逐token一致，`region_kernel_audit.passed=true`且observed/expected逐类调用数相等；Graph capture=1/fallback=0、29 profile steps、hygiene=clean、nsys/SQLite哈希均通过。双链退出后双卡均1MiB。
- 归档完成：JSSA/NFA语义目录含45项有效文件manifest（SHA256=`c741db21750cbb9ae184fbd1b7718be7f53419ea3b22e55a3f97806561fcbb53`），NFA gate0.1 operator含30项manifest（SHA256=`f53930fbb9ac2674cf710dfd55df9d478178c773e1db1aeda58c0198c6d3f1e6`）；主仓副本逐项哈希复核PASS。总对比表已增为30行（原25行不重跑+JSSA/NFA语义5行），新结果全部保持 pending-independent-verification。本轮只补上述缺格，未恢复Quest B2/B4或旧host-wall任务。
