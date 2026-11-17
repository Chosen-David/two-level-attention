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
