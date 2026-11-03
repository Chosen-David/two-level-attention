# NFA decode 热路径优化记录

## 生产口径与目标

- 模型/输入：Qwen3-8B，原生 RULER 32K `niah_single_1` 样本 0，B=1，固定生成 128 token。
- 目标：NFA eager/Graph TPOT 分别低于 FA2 eager/Graph，同时保持生成 token 一致。
- FA2：eager 22.915487 ms；Graph 12.560864 ms。
- 当前 NFA：eager 136.930237 ms；Graph 64.760643 ms。
- 正式计时与 profiler 分离；短 decode 仅用于画像/迭代信号，不作为最终验收。

## Baseline：receipt 热路径清理后

- eager 136.930237 ms，Graph 64.760643 ms；修改前后 128/128 token 一致。
- Nsight Systems：`profile_graph_nodes_32k_b1.sqlite`，node 级追踪 7 次 replay。
- replay kernel 总时间 63.561937 ms/步，与未 profile TPOT 64.760643 ms 相符。
- 主要分解（ms/步）：
  - `_nfa_fine_paged_kernel`：33.6849（0.93569 ms/层 × 36）。
  - 通用 vectorized/elementwise：约 11.23。
  - `_nfa_page_stats_kernel`：3.9401。
  - 模型 skinny GEMM（按 kernel 名初步归组）：约 4.5。
  - radix sort：1.4720；index elementwise：1.2673。
  - `_gather_attend_grouped_kernel`：1.0019。
  - 其他选择/拷贝/reduce/ranking：余量。
- 结论：Graph 差距主要是图内 GPU 工作，fine probability 是第一优化对象；不能用 host launch 或 gather attention 解释主差距。

## Iter 1：fine score 三阶段拆分

- 修改：paged K 每 tile 只读一次；并行写 BF16 score，按原 tile/head 顺序计算 FP32 online-softmax 状态，再并行归一化和 GQA 平均。
- 正确性：`tests/test_nfa_current_query_selector.py` 6/6 通过。
- 生产 Graph 8-token 信号：TPOT 44.009762 ms，steady 43.959936 ms；同画像口径 baseline 为 65.583199/65.390450 ms，未 profile 128-token baseline 为 64.760643 ms。
- 结果：`iter1_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 后续画像：score/state/probability 分别 1.1941/11.2698/0.2729 ms 每步；8-token 前缀与 128-token baseline 一致。

## Iter 2：tile 部分 softmax 并行归并

- 修改：score kernel 同步输出每 tile 的 FP32 max/sum；状态 kernel 只按 tile 顺序合并部分和，不再对全部候选串行 exp。
- 数值说明：跨 tile 归并舍入与原实现不同，必须检查最终 token identity。
- 正确性：`tests/test_nfa_current_query_selector.py` 6/6 通过。
- 生产 Graph 8-token 信号：TPOT 33.729343 ms，steady 33.560993 ms；相对 iter1 再降约 10.28 ms，相对未 profile baseline 降约 31.03 ms。
- 结果：`iter2_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。

## Iter 3：L1 score/geometry/pool 融合

- 修改：单次 `_nfa_l1_scores_kernel` 同时计算 min/max 与可选 avg 的 GQA 均值、动态 skip、区域可见性、far/near pool 和非对齐 bypass 边界，直接输出两路 Top-K 分数；删除热路径上的 coarse/fine 中间张量、pool expand、region mask 与 where 链。
- candidate keep 改为取回已入选 L1 分数并判断非 `-inf`，不再构造 `[M,HKV,KT]` pool bool 张量。
- 正确性：`tests/test_nfa_current_query_selector.py` 6/6 通过。
- 生产 Graph 8-token 信号：TPOT 29.114752 ms，steady 29.107520 ms；相对 iter2 再降约 4.61 ms，相对未 profile baseline 降约 35.65 ms。
- 结果：`iter3_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 117.143715 ms，steady 116.762512 ms；相对 iter2 eager 140.511459 ms 降约 23.37 ms，说明 host/aten 融合有效，但距 FA2 eager 仍远。

## Iter 4：candidate expansion/completion/rank metadata 融合

- 修改：`_candidate_pack_batched_kernel` 直接从 far/near Top-K 块展开 token，处理 sink/SWA 非对齐 bypass、候选过滤、sink+SWA 补全与稳定 compaction；同一 kernel 生成 skip-aware rank lower/upper/quota。
- 删除生产热路径 `_batched_geometry`、`_candidate_assembly`、`_complete_pack_batched` 及其 cat/gather/expand/arange/clamp/where 中间链。
- 正确性：`tests/test_nfa_current_query_selector.py` 6/6 通过。
- 生产 Graph 8-token 信号：TPOT 24.750401 ms，steady 24.732384 ms；相对 iter3 再降约 4.36 ms，相对未 profile baseline 降约 40.01 ms。
- 结果：`iter4_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 87.709885 ms，steady 87.374832 ms；相对 iter3 eager 再降约 29.43 ms。Graph/eager 的前 8 个生成 token 均与 baseline 一致。

## Iter 5：final canonical union + physical map 融合

- 修改：`_final_union_physical_kernel` 在 request-level head 0 上直接生成 middle、sink/SWA protected 和 first-decode seed 的 union；一次 Triton sort 后用 `cumsum` 去重压缩，并直接写物理 gather slot 与 visible length。
- 删除热路径 protected/seed expand、`middle_valid`、concat、两遍 PyTorch radix sort、canonical 临时张量、单独 physical map 和 visible zero。
- 正确性：`tests/test_nfa_current_query_selector.py` 6/6 通过。
- 生产 Graph 8-token 信号：TPOT 20.792641 ms，steady 20.779696 ms；相对 iter4 再降约 3.96 ms，相对未 profile baseline 降约 43.97 ms。
- 结果：`iter5_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 64.081856 ms，steady 63.912752 ms；相对 iter4 eager 再降约 23.63 ms。Graph/eager 的前 8 个生成 token 均与 baseline 一致。

## Iter 6：持久页统计 + 最新块增量更新

- 修改：按 layer/request slot 持久化 min/max/sum；首次、重复长度或非连续长度全量重算，正常 `length == previous + 1` 仅重算最新逻辑块，其余块复制持久统计。独立 commit kernel 更新长度，PAD capture 不消费真实状态，freeze 复用 slot 时清零长度。
- 页统计直接输出 BF16 sum 舍入后的 avg，删除 `k_sum -> fp32 -> avg` 两个 aten 节点；skip 区域仍由 L1 pool 屏蔽，持久统计保存真实 KV 以允许后续 unskip。
- 正确性：`tests/test_nfa_current_query_selector.py` 6/6 通过。
- 生产 Graph 8-token 信号：TPOT 17.430431 ms，steady 17.415009 ms；相对 iter5 再降约 3.36 ms，相对未 profile baseline 降约 47.33 ms。
- 结果：`iter6_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 64.407234 ms，steady 64.325840 ms；与 iter5 eager 基本持平（略慢 0.33 ms），说明 Graph 的 GPU 收益被 eager host/launch 开销掩盖。Graph/eager 前 8 token 均与 baseline 一致。

## Iter 7：L1 直读持久页统计，删除整层统计复制

- 修改：页统计 kernel 只更新 request-slot 持久 min/max/sum，不再写三份 `[M,KT,HKV,D]` batch 输出；L1 按 layer/request slot 直读持久统计，并在使用点由 sum 重建 avg，保留 BF16 sum 与 BF16 avg 两次中间舍入。
- 正确性：`tests/test_nfa_current_query_selector.py` 6/6 通过。
- 生产 Graph 8-token 信号：TPOT 16.917631 ms，steady 16.897728 ms；相对 iter6 再降约 0.51 ms，相对未 profile baseline 降约 47.84 ms。
- 结果：`iter7_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 64.788734 ms，steady 64.536816 ms；与 iter6 eager 基本持平，host/launch 仍是主矛盾。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 30 秒 GPU hygiene 旁路日志。

## Iter 8：softmax 状态按 query head 并行

- 修改：tile 状态归并从每 `(layer,request,KV-head)` 一个 16-warp program 改为每 query head 一个单 warp program；133 个 tile 的归并顺序和每次浮点运算不变，消除低 occupancy 与无效线程。
- 正确性：`tests/test_nfa_current_query_selector.py` 6/6 通过。
- 生产 Graph 8-token 信号：TPOT 16.695200 ms，steady 16.674192 ms；相对 iter7 再降约 0.22 ms。
- 结果：`iter8_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 65.812515 ms，steady 65.460258 ms；未获得 eager 收益，本轮不作为 eager 改善证据。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 9：候选单遍展开，删除无效稳定压缩

- 修改：candidate kernel 从两遍计数/压缩改为一遍按原 virtual slot 写 ID/valid；fine score 改为直接读取 valid mask。下游 ranking 的键与候选位置无关且本来就消费 valid/unique，因此候选集合与排序语义不变。
- 正确性：`tests/test_nfa_current_query_selector.py` 6/6 通过。
- 生产 Graph 8-token 信号：TPOT 16.064064 ms，steady 16.044512 ms；相对 iter8 再降约 0.63 ms。
- 结果：`iter9_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 64.360222 ms，steady 64.505936 ms；相对 iter8 改善但仍与 iter6/7 同量级，host/launch 主瓶颈未变。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 10：fine score GQA 点积改用 Tensor Core

- 修改：每个 tile 的 4 次标量广播点积改为一次 BF16 `tl.dot`、FP32 累加，同时生成所有 query head 的 score 与 tile partial max/sum；输入仍含原 BF16 舍入，FP32 累加顺序变化由 token identity 验收约束。
- 正确性：`tests/test_nfa_current_query_selector.py` 6/6 通过。
- 生产 Graph 8-token 信号：TPOT 15.663136 ms，steady 15.642096 ms；相对 iter9 再降约 0.40 ms。
- 结果：`iter10_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 65.690941 ms，steady 65.104530 ms；GPU kernel 改善未转化为 eager 收益。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 11：请求元数据提升到 step 级

- 修改：Cache 新增固定 `valid_req_ids_buf` 与 `query_req_ids_buf`，在 `plan_common` 通过 CPU staging 每步刷新一次；selector 不再逐层执行 `where + zeros_like + arange`。
- 正确性：`tests/test_nfa_current_query_selector.py` 6/6 通过。
- 生产 Graph 8-token 信号：TPOT 15.544832 ms，steady 15.528288 ms；相对 iter10 再降约 0.12 ms。
- 结果：`iter11_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 61.932865 ms，steady 61.802385 ms；相对 iter10 降约 3.76 ms，确认逐层 aten 元数据节点是 eager 可触达开销。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 12：生产 ranking 跳过冗余设备断言

- 修改：`rank_unique_batched` 保留默认 `validate=True`，生产 selector 对由 candidate kernel 固定几何生成的 lower/upper/quota 使用 `validate=False`，删除逐层比较、归约和 `_assert_async` 节点。
- 正确性：`tests/test_nfa_current_query_selector.py` 6/6、`tests/test_unique_ranking.py` 12/12 通过。
- 生产 Graph 8-token 信号：TPOT 15.000896 ms，steady 14.991216 ms；相对 iter11 再降约 0.54 ms。
- 结果：`iter12_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 58.812447 ms，steady 58.114336 ms；相对 iter11 降约 3.12 ms。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 13：far/near Top-K 合并启动

- 修改：两路独立 Top-K 合并为一个 2D-grid Triton launch；每路保留原 packed key、降序 sort 与小索引优先的 tie 规则。
- 正确性：`tests/test_nfa_current_query_selector.py` 6/6 通过。
- 生产 Graph 8-token 信号：TPOT 14.785120 ms，steady 14.782672 ms；相对 iter12 再降约 0.22 ms。
- 结果：`iter13_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 58.251263 ms，steady 58.247215 ms；相对 iter12 再降约 0.56 ms。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 14：页统计按 16 块分组启动

- 修改：页统计 program 从每块一个改为每 16 块一个；稳态仅拥有最新块的 program 执行一次迭代，全量重建按块组并行，逐块统计公式不变。
- 正确性：`tests/test_nfa_current_query_selector.py` 6/6 通过。
- 生产 Graph 8-token 信号：TPOT 14.762016 ms，steady 14.757664 ms；相对 iter13 仅降约 0.023 ms，说明空 program 启动不是该 kernel 的主要成本。
- 结果：`iter14_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 58.063934 ms，steady 57.745487 ms；基本持平。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 15：fine score/probability 改为 4 warp

- 修改：fine Tensor Core score 与 64 元素 probability tile 的 warp 数从 16 降为 4；softmax state 保持 Iter 8 的单 warp。
- 正确性：`tests/test_nfa_current_query_selector.py` 6/6 通过。
- 生产 Graph 8-token 信号：TPOT 14.390944 ms，steady 14.376880 ms；相对 iter14 再降约 0.37 ms。
- 结果：`iter15_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 58.706623 ms，steady 58.365328 ms；GPU 收益未转化为 eager 收益。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 16：最后一级 Top-K merge 与 finish 融合

- 修改：生产 `rank_unique_batched` 在只剩两路 sorted run 时，由 `_merge_finish` 在寄存器中完成最后 bitonic merge 并直接执行原 `_finish`；删除最后一次中间 Top-K 写读和一次 launch，通用 `tree_topk` 不变。
- 正确性：`tests/test_nfa_current_query_selector.py` 6/6、`tests/test_unique_ranking.py` 12/12、`tests/test_exact_batch_selection.py` 24/24 通过；后者新增 R=4096 以覆盖融合分支及 CUDA Graph 可变 quota/bounds 对拍。
- 生产 Graph 8-token 信号：TPOT 14.643039 ms，steady 14.440192 ms；相对 iter15 steady 基本持平（+0.063 ms），未形成稳定 Graph 收益。
- 结果：`iter16_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 57.362335 ms，steady 56.720385 ms；相对 iter15 steady 降约 1.65 ms，说明减少逐层 launch 对 eager 有效。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 17：selector/ranking 临时 buffer 跨层复用

- 修改：Cache 持有固定 selector workspace，Q、L1、far/near Top-K、candidate、fine score/softmax/probability 与 ranking 中间张量按独立名称、形状和 dtype 复用；owners 仍每次原地填 -1。中间结果只在同 stream 当前层内消费，持久页统计和逐层 gather 输出不共享。
- 正确性：`tests/test_nfa_current_query_selector.py` 6/6、`tests/test_unique_ranking.py` 12/12、`tests/test_exact_batch_selection.py` 24/24 通过。
- 生产 Graph 8-token 信号：TPOT 14.467200 ms，steady 14.457312 ms；相对 iter16 steady 基本持平，allocator host 优化不改变图内 GPU 工作。
- 结果：`iter17_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 53.893696 ms，steady 53.627647 ms；相对 iter16 steady 降约 3.09 ms，确认临时分配仍是 eager 可触达开销。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 18：unique 判定并入 fine score

- 修改：删除无消费者的 `valid` 输出及独立 `_owner_validity_batched_kernel`；fine score 按原 candidate tile 读取 owner 并写出 `unique`，保留最大 candidate slot 获胜和 ID 范围检查语义。
- 正确性：`tests/test_nfa_current_query_selector.py` 6/6、`tests/test_exact_batch_selection.py` 24/24 通过。
- 生产 Graph 8-token 信号：TPOT 14.390976 ms，steady 14.376816 ms；相对 iter17 steady 降约 0.080 ms。
- 结果：`iter18_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 53.183617 ms，steady 52.589920 ms；相对 iter17 steady 降约 1.04 ms。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 19：candidate 展开与 owner 初始化/更新合并

- 修改：candidate kernel 在每个 row 的单一 program 内先初始化 owner 表，再展开候选并以原 `atomic_max` 规则更新；删除独立 `fill_` 和 owner kernel launch，fine score 直接消费该表。
- 正确性：`tests/test_nfa_current_query_selector.py` 6/6、`tests/test_exact_batch_selection.py` 24/24 通过。
- 生产 Graph 8-token 信号：TPOT 14.491680 ms，steady 14.463568 ms；相对 iter18 steady 慢约 0.087 ms，说明 owner 初始化并行度下降抵消了 launch 收益。
- 结果：`iter19_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 52.347103 ms，steady 51.640465 ms；相对 iter18 steady 降约 0.95 ms。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 20：Triton CompiledKernel 直发与完整 specialization 缓存

- 修改：selector/ranking 的生产 launch 复用首次 JIT 编译产物，后续直接调用 `CompiledKernel.run`；初版仅按调用点缓存会在多几何测试中误用 variant 并触发 illegal memory access，修复为由 Triton binder 生成完整 specialization/options，连同设备与调用点组成缓存键。
- 正确性：修复后 `tests/test_nfa_current_query_selector.py`、`tests/test_unique_ranking.py`、`tests/test_exact_batch_selection.py` 共 42/42 通过。
- 生产 Graph 8-token 信号：TPOT 14.487104 ms，steady 14.460896 ms；相对 iter19 steady 基本持平。
- 结果：`iter20_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 52.927265 ms，steady 52.808992 ms；相对 iter19 steady 慢约 1.17 ms，保留 binder 的直发未形成 eager 收益。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 21：轻量严格参数键的 CompiledKernel 直发

- 修改：删除逐次 binder 与 `str(specialization)`；缓存键严格覆盖全部位置标量、constexpr/编译选项、tensor dtype/device 和指针 16 字节对齐，后续直接发射已缓存 variant。该键比 Triton 实际特化更严格，避免跨几何或对齐误复用。
- 正确性：selector、unique ranking、exact batch selection 共 42/42 通过。
- 生产 Graph 8-token 信号：TPOT 14.478240 ms，steady 14.452240 ms；相对 iter20/19 steady 均基本持平。
- 结果：`iter21_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 51.280895 ms，steady 51.048208 ms；相对 iter20 steady 降约 1.76 ms，相对 iter19 steady 降约 0.59 ms。两臂均有 GPU hygiene 旁路日志。

## Iter 22：selector 固定几何令牌驱动直发

- 修改：生产 selector 以严格固定几何令牌替代每次 launch 的逐张量 dtype/device/data_ptr 查询；令牌覆盖 batch/head/页表/Q strides 与对齐、visible bucket 和 Top-K 配置，kwargs 仍完整进入编译键。独立 ranking 调用继续走完整逐参数键。
- 正确性：selector、unique ranking、exact batch selection 共 42/42 通过。
- 生产 Graph 8-token 信号：TPOT 14.482816 ms，steady 14.473136 ms；相对 iter21 steady 基本持平，符合只减少 host 工作的预期。
- 结果：`iter22_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 47.745792 ms，steady 47.539104 ms；相对 iter21 steady 降约 3.51 ms，相对 iter19 降约 4.10 ms。两臂均有 GPU hygiene 旁路日志。

## Iter 23：selector 级执行上下文复用

- 修改：每层 selector 入口只查询一次当前 CUDA stream、Triton launch hooks 与 runtime debug；selector/ranking 直发复用该上下文，无 enter hook 时直接传 `launch_metadata=None`，删除每次 launch 的 stream/device index 与 metadata 查询。capture 每层仍重新取当前 stream，独立 ranking 无上下文时保留原查询路径。
- 正确性：selector、unique ranking、exact batch selection 共 42/42 通过。
- 生产 Graph 8-token 信号：TPOT 14.465184 ms，steady 14.443264 ms；相对 iter22 steady 基本持平，符合仅减少 host 工作的预期。
- 结果：`iter23_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 46.312096 ms，steady 45.809584 ms；相对 iter22 steady 降约 1.73 ms。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 24：paged KV 写入与 gather attention 直发

- 修改：生产 paged write、grouped/small-head gather attention 复用严格参数键的 `CompiledKernel.run` 直发；KV 写入先于本层 selector，因此写入前单独刷新当前 stream/hooks，避免 Graph capture 使用旧上下文。通用无 workspace 调用保持原 JIT 路径。
- 正确性：selector、unique ranking、exact batch selection 共 42/42，`test_graph_ready_paged_region_cache.py` 4/4 通过。
- 生产 Graph 8-token 信号：TPOT 14.480064 ms，steady 14.455152 ms；相对 iter23 steady 基本持平。
- 结果：`iter24_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 41.626625 ms，steady 41.977312 ms；相对 iter23 steady 降约 3.83 ms。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 25：fine probability 并入 ranking tile key

- 修改：新增 `_tile_keys_from_scores`，直接消费 BF16 fine score 与 FP32 softmax 状态，在 ranking tile 内完成逐 head 归一化、逐 head BF16 舍入、原顺序 GQA 累加和 KV-dtype 概率舍入后立即打包排序键；删除独立 probability launch 与 BF16 probability 中间张量。通用 probability ranking 路径保留。
- 正确性：selector、unique ranking、exact batch selection 共 42/42 通过。
- 生产 Graph 8-token 信号：TPOT 14.258272 ms，steady 14.246928 ms；相对 iter24 steady 降约 0.21 ms。
- 结果：`iter25_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 42.397888 ms，steady 41.852016 ms；相对 iter24 steady 降约 0.13 ms，launch 收益大部分被融合 tile 的额外计算/占用成本抵消。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 26：调用点 launch plan 与几何对象复用

- 修改：selector/ranking/paged write/gather 统一使用 `_fast_launch`；按调用点缓存 bound `CompiledKernel.run`、function、packed metadata、kwargs 尾项和 grid，稳态以几何对象身份做 O(1) 校验，删除逐 launch 的完整参数键构造与 kwargs 排序。几何/debug 变化仍回退完整严格键；selector 每层复用相等的旧几何对象。
- 正确性：selector、unique ranking、exact batch selection 共 42/42 通过，`test_graph_ready_paged_region_cache.py` 4/4 通过。
- 生产 Graph 8-token 信号：TPOT 14.365568 ms，steady 14.264832 ms；相对 iter25 steady 基本持平，符合只减少 host 工作的预期。
- 结果：`iter26_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 35.775776 ms，steady 35.611488 ms；相对 iter25 steady 降约 6.24 ms。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 27：launch 执行上下文提升到 model-forward 级

- 修改：Qwen 每层 0 的 decode attention 入口按 model forward 统一创建/刷新 selector workspace 的 stream/hooks/debug 上下文；后续 36 层 KV write、selector 和 gather 不再重复查询。独立 selector 的 CUDA Graph capture 仍逐次强制刷新，避免复用 warmup/default stream。
- 正确性：初版直接 selector capture 回归发现旧 stream 问题，补 capture 强制刷新后 selector、unique ranking、exact batch selection、paged cache 共 46/46 通过。
- 生产 Graph 8-token 信号：TPOT 14.284768 ms，steady 14.259008 ms；相对 iter26 steady 基本持平，符合仅减少 host 工作的预期。
- 结果：`iter27_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 35.024033 ms，steady 34.608032 ms；相对 iter26 steady 降约 1.00 ms。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 28：页统计长度提交并入 L1

- 修改：页统计完成后，由 L1 每个 `(layer,request)` 的首个 program 原时序提交持久统计长度；删除独立 `_nfa_page_stats_commit_kernel` 及每层一次 launch，PAD 请求仍不提交。
- 正确性：selector、unique ranking、exact batch selection、paged cache 共 46/46 通过。
- 生产 Graph 8-token 信号：TPOT 14.238880 ms，steady 14.214864 ms；相对 iter27 steady 降约 0.044 ms。
- 结果：`iter28_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 34.681889 ms，steady 34.598064 ms；相对 iter27 steady 基本持平，单次 launch 消除不足以改变当前 eager 主瓶颈。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 29：final union 改为小排序 + 有序归并

- 修改：利用 ranking middle 已按逻辑 token 升序的性质，只对 protected+seed 的生产 260 项（ padded 512）排序；反转第二路后与 1024 项 middle 做 2048 项 bitonic merge，替代原 2048 项完整 `tl.sort`。canonical 顺序、cumsum 去重、物理 slot 映射和 seed 语义不变。
- 正确性：selector、unique ranking、exact batch selection、paged cache 共 46/46 通过。
- 生产 Graph 8-token 信号：TPOT 13.872768 ms，steady 13.856896 ms；相对 iter28 steady 降约 0.358 ms。
- 结果：`iter29_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 34.009727 ms，steady 33.719856 ms；相对 iter28 steady 降约 0.878 ms。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 30：generation-tagged owner，删除候选整表清零

- 修改：candidate 每行递增 64-bit generation，owner 改写为 `generation * width + slot` 的 64-bit `atomic_max`；fine score 同时校验 generation 与最大 candidate slot。owners/generation 仅首次分配置零，稳态不再清理 32K 容量表。
- 正确性：selector、unique ranking、exact batch selection、paged cache 共 46/46 通过。
- 生产 Graph 8-token 信号：TPOT 13.799008 ms，steady 13.777376 ms；相对 iter29 steady 降约 0.080 ms。
- 结果：`iter30_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 34.366657 ms，steady 33.887808 ms；相对 iter29 steady 慢约 0.168 ms，未形成 eager 收益，64-bit owner/atomic 成本可能抵消清零节省。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 31：删除 GraphRunner.step 重复 plan_common

- 修改：删除 `step()` 内第一次 `cache.plan_common(padded_ids)`，统一由 `_prepare_metadata` 在 visible bucket 确定后准备本步页表/长度；capture 路径不变。
- 正确性：selector、unique ranking、exact batch selection、paged cache 共 46/46 通过。
- 生产 Graph 8-token 信号：TPOT 13.799680 ms，steady 13.774400 ms；相对 iter30 steady 基本持平（-0.003 ms）。
- 结果：`iter31_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 34.638622 ms，steady 34.633839 ms；相对 iter30 steady 慢约 0.746 ms，未形成 eager 收益。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 32：fine score/softmax tile 从 64 提升到 128

- 修改：`_fused_fine_scores_paged` 使用 128 候选/tile、4 warp；partial-softmax tile 数减半，减少状态归并迭代与中间流量。跨 tile 归并舍入顺序变化，继续以 token identity 验收。
- 正确性：selector、unique ranking、exact batch selection、paged cache 共 46/46 通过。
- 生产 Graph 8-token 信号：TPOT 13.743296 ms，steady 13.736864 ms；相对 iter31 steady 降约 0.038 ms。
- 结果：`iter32_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 33.991329 ms，steady 33.938030 ms；相对 iter31 steady 降约 0.696 ms，但与 iter30 steady 基本持平。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 33：fine score/softmax tile 提升到 256

- 修改：fine tile 从 128 提升到 256，保持 4 warp；partial-softmax tile 数进一步减半。
- 正确性：selector、unique ranking、exact batch selection、paged cache 共 46/46 通过。
- 生产 Graph 8-token 信号：TPOT 13.605568 ms，steady 13.563904 ms；相对 iter32 steady 再降约 0.173 ms。
- 结果：`iter33_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 35.285343 ms，steady 34.771471 ms；相对 iter32 steady 慢约 0.833 ms，大 tile 的 eager 同步/占用成本抵消归并收益。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 34：按 eager/capture 选择 fine tile（失败，plan 键缺模式）

- 修改：eager 使用 128 tile、capture 使用 256 tile，复用现有 `is_current_stream_capturing` 检测。
- 正确性：selector、unique ranking、exact batch selection、paged cache 共 46/46 通过，但生产 Graph 验收失败。
- 生产 Graph 8-token：TPOT 14.724960 ms，steady 14.688240 ms；生成序列为 `[576, 0, 0, 0, 0, 0, 0, 0]`，与 baseline 不一致，本轮结果无效。
- 根因：warmup 与 capture 的 geometry 未包含 capture 模式，`_fast_launch` 因几何对象身份相同而错复用另一 tile 的 plan。结果保留为 `iter34_graph_32k_b1_8tok.json`，未跑无效修改的 eager 臂；不自行 commit。

## Iter 35：capture 模式纳入严格几何键

- 修改：geometry 增加 `capturing`，隔离 eager 128 tile 与 Graph 256 tile 的 compiled variant、grid 和 launch plan；修复 Iter 34 token 回退。
- 正确性：selector、unique ranking、exact batch selection、paged cache 共 46/46 通过。
- 生产 Graph 8-token 信号：TPOT 13.703872 ms，steady 13.645440 ms；相对 Iter 33 steady 慢约 0.082 ms，处于小幅运行波动，保留接近 256-tile 最佳臂的性能。
- 结果：`iter35_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 34.530174 ms，steady 34.114992 ms；相对 Iter 32 steady 慢约 0.177 ms，未形成额外 eager 收益。Graph/eager 前 8 token 均恢复并保持一致；两臂均有 GPU hygiene 旁路日志。

## Iter 36：gather attention B1 tile 提升到 256（编译资源失败）

- 修改：B1 grouped gather 的 `BT` 从 128 提升到 256，保持默认流水级数。
- 正确性：selector、unique ranking、exact batch selection、paged cache 共 46/46 通过；生产 Graph capture 在 Triton 编译阶段报 `OutOfResources`。
- 失败原因：该 variant 需要 348192 B shared memory，超过 H20 的 232448 B 上限；未产生有效性能或 token 结果，不自行 commit。

## Iter 37：gather 256 tile 改单流水（双路径回退）

- 修改：为 256 tile 设置 `num_stages=1`，使 shared memory 满足设备上限。
- 正确性：selector、unique ranking、exact batch selection、paged cache 共 46/46 通过；Graph/eager 前 8 token 均与 baseline 一致。
- 生产 Graph 8-token 信号：TPOT 13.913952 ms，steady 13.892512 ms；相对 Iter 35 steady 慢约 0.247 ms。
- 同修改 eager 8-token：TPOT 34.933281 ms，steady 34.437311 ms；相对 Iter 35 steady 慢约 0.322 ms。
- 结果：`iter37_graph_32k_b1_8tok.json`、`iter37_eager_32k_b1_8tok.json`，两臂均有 GPU hygiene 旁路日志。双路径均回退，撤销本实验并恢复 128 tile 默认流水；不自行 commit。

## Iter 38：生产 host 校验与 tensor 转换快路径

- 修改：全局 selector options 每次 model forward 只在 layer 0 校验；生产 ranking 以 `prevalidated=True` 跳过固定几何的重复 shape/dtype/device 检查；selector helper 对已有 tensor 直接复用，不再重复 `torch.as_tensor`。
- 正确性：selector、unique ranking、exact batch selection、paged cache、selector options 共 55/55 通过。
- 生产 Graph 8-token 信号：TPOT 13.606112 ms，steady 13.584384 ms；相对 Iter 35 steady 降约 0.061 ms，基本持平。
- 结果：`iter38_graph_32k_b1_8tok.json`。按共享会话要求本轮不自行 commit。
- 同修改 eager 8-token：TPOT 33.941090 ms，steady 33.947952 ms；相对 Iter 35 steady 降约 0.167 ms。Graph/eager 前 8 token 均与 baseline 一致；两臂均有 GPU hygiene 旁路日志。

## Iter 39：prepare-Q 内联到 L1/fine score

- 修改：删除生产 selector 热路径的独立 `_prepare_q` launch；L1 与 fine kernel 直接按 layer/request/head stride 读取 post-RoPE Q，并保留 `(q_fp32 * scale).to(bf16).to(fp32)` 的缩放与中间舍入口径。
- 正确性：selector、unique ranking、exact batch selection、paged cache 共 46/46 通过；Graph/eager 前 8 token 均与 baseline 一致。
- 生产 Graph 8-token 信号：TPOT 13.567712 ms，steady 13.541696 ms；相对 Iter 38 steady 降约 0.043 ms，收益很小。
- 同修改 eager 8-token：TPOT 34.143105 ms，steady 34.107824 ms；相对 Iter 38 steady 慢约 0.160 ms，独立 launch 的删除被 raw-Q 重复读取/计算成本抵消。
- 结果：`iter39_graph_32k_b1_8tok.json`、`iter39_eager_32k_b1_8tok.json`，两臂均有 GPU hygiene 旁路日志；未观察到其他进程或离散平台污染。本轮不作为双路径收益，不自行 commit。

## Iter 40：page-stats 与 prepare-Q 横向融合

- 修改：L1/L2 共用同一份 Q 时，将相互独立的持久页统计与 Q 收集/缩放横向合成一个 1D kernel；抽出共享 Triton device body，独立路径和融合路径复用同一运算，保留全部 BF16 中间舍入。不同 Q 维度的通用路径不变。
- 正确性：selector、unique ranking、exact batch selection、paged cache 共 46/46 通过；Graph/eager 前 8 token 均与 baseline 一致。Iter 39 raw-Q 内联已精确回退后再实施本轮。
- 生产 Graph 8-token 信号：TPOT 13.495936 ms，steady 13.485504 ms；相对 Iter 38 steady 降约 0.099 ms。
- 同修改 eager 8-token：TPOT 32.338558 ms，steady 32.147345 ms；相对 Iter 38 steady 降约 1.801 ms，确认减少逐层 launch 是 eager 的有效杠杆。
- 结果：`iter40_graph_32k_b1_8tok.json`、`iter40_eager_32k_b1_8tok.json`，两臂均有 GPU hygiene 旁路日志，未见其他进程或离散平台污染；不自行 commit。

## Iter 41：ranking merge-finish 与 final union 融合（双路径回退）

- 修改：在 ranking 末级 `_merge_finish` 的同一 CTA 内，由 KV-head 0 继续执行 final canonical union 与物理映射；中间以 CTA barrier 保证 middle/count 可见，通用非两级末级路径保持独立实现。
- 正确性：selector、unique ranking、exact batch selection、paged cache 共 46/46 通过；Graph/eager 前 8 token 均与 baseline 一致。
- 生产 Graph 8-token 信号：TPOT 13.712480 ms，steady 13.697584 ms；相对 Iter 40 steady 慢约 0.212 ms。
- 同修改 eager 8-token：TPOT 32.714783 ms，steady 32.679649 ms；相对 Iter 40 steady 慢约 0.532 ms。
- 结果：`iter41_graph_32k_b1_8tok.json`、`iter41_eager_32k_b1_8tok.json`，两臂均有 GPU hygiene 旁路日志。融合大 kernel 的资源压力超过少一次 launch 的收益，撤销本实验并保留 Iter 40；不自行 commit。

## Iter 42：decode K/V 写入融入 page-stats + prepare-Q

- 修改：Qwen decode 不再提前独立启动 paged-write；生产子空间路径由页统计最新逻辑块的 owner program 先写当前 `(request,KV-head)` 的 K/V，CTA 内同步后复用原页统计 body，同时保留横向 prepare-Q。非融合几何仍在 selector 内调用原 `write_decode`，有/无写入使用独立 launch key。
- 正确性：selector、unique ranking、exact batch selection、paged cache 共 46/46 通过；Graph/eager 前 8 token 均与 baseline 一致。
- 生产 Graph 8-token 信号：TPOT 13.379520 ms，steady 13.366448 ms；相对 Iter 40 steady 再降约 0.119 ms。
- 同修改 eager 8-token：TPOT 32.193409 ms，steady 32.154114 ms；TPOT 相对 Iter 40 降约 0.145 ms，steady 基本持平（慢约 0.007 ms）。
- 结果：`iter42_graph_32k_b1_8tok.json`、`iter42_eager_32k_b1_8tok.json`，两臂均有 GPU hygiene 旁路日志，起跑前 GPU0/GPU1 均为 1 MiB、0% 且无 compute apps；保留本轮，不自行 commit。

## Iter 43：B1 grouped gather 的 split-KV FlashDecoding

- 修改：Graph B1 grouped gather 从仅 8 个 CTA 改为 8 个连续 token-block 分区并行，写 FP32 `(acc,m,z)` partial，再由独立 combine kernel 按 split 序号确定性归并；eager 保留原 grouped kernel 并提前返回，不分配 split workspace。
- 中间修复：初版 eager 因逐层 `is_current_stream_capturing()` 查询回退到 steady 33.867840 ms；复用 selector capture 标志后为 32.492735 ms；独立 split kernel 但通用 host 参数分支为 32.735344 ms。最终将 eager 原 `_fast_launch` 隔离并立即返回，恢复 eager 性能。
- 正确性：selector、unique ranking、exact batch selection、paged cache、frozen runtime 共 60/60 通过；Graph/eager 前 8 token 均为 `[576,1372,374,220,19,17,17,21]`，与 baseline 一致。
- 生产 Graph 8-token 信号：TPOT 12.689696 ms，steady 12.661792 ms；相对 Iter 42 steady 降约 0.705 ms，距 FA2 Graph TPOT 12.560864 ms 尚约 0.101 ms。
- 同修改 eager 8-token：TPOT 32.262081 ms，steady 32.095999 ms；相对 Iter 42 steady 降约 0.058 ms，确认 host 回退已消除。
- 结果：`iter43_graph_32k_b1_8tok.json`、`iter43_eager_32k_b1_8tok.json`，两臂均有 GPU hygiene 旁路日志，未见其他进程或离散平台污染；保留本轮，不自行 commit。

## Iter 44：split-KV token-block 均衡分区

- 修改：split 边界由 `ceil(num_blocks/NSPLIT)` 改为 floor 比例边界；生产约 10 块、8 split 时从仅 5 个 split 有工作改为 8 个 split 全部参与，单个 split 最多 2 块。
- 正确性：60/60 回归通过；Graph 前 8 token 与 baseline 一致。仅 Graph split 路径变化，eager 仍走 Iter 43 已验证的原 grouped kernel。
- 生产 Graph 8-token 信号：TPOT 12.640416 ms，steady 12.629328 ms；相对 Iter 43 steady 再降约 0.032 ms，距 FA2 Graph TPOT 尚约 0.068 ms。
- 结果：`iter44_graph_32k_b1_8tok.json` 与 GPU hygiene 旁路日志；保留本轮，不自行 commit。

## Iter 45：split-KV 分区数提升到 10（Graph 回退）

- 修改：`NSPLIT` 从 8 提升到 10，使生产约 10 个 token block 每 split 一块。
- 正确性：60/60 回归通过，Graph 前 8 token 与 baseline 一致。
- 生产 Graph 8-token 信号：TPOT 12.756704 ms，steady 12.752832 ms；相对 Iter 44 steady 慢约 0.124 ms。80 个 CTA 的占用/第二波尾及 combine 增量超过负载均衡收益，撤销并恢复 split8。
- 结果：`iter45_graph_32k_b1_8tok.json` 与 GPU hygiene 旁路日志；不自行 commit。

## Iter 46：split-KV 分区数调到 9（Graph 回退）

- 修改：`NSPLIT=9`，以 72 个 CTA 避开 split10 的 80 CTA 超 SM 尾部，同时减少双块分区。
- 正确性：60/60 回归通过，Graph 前 8 token 与 baseline 一致。
- 生产 Graph 8-token 信号：TPOT 12.673472 ms，steady 12.667808 ms；相对 Iter 44 steady 慢约 0.038 ms，撤销并恢复 split8。
- 结果：`iter46_graph_32k_b1_8tok.json` 与 GPU hygiene 旁路日志；不自行 commit。

## Iter 47：fine-score 末 CTA 融合 softmax-state（仅保留 eager）

- 修改：candidate 清零每行 tile counter；fine-score 各 CTA 写 partial 后以 acq_rel 原子计数，最后完成的 CTA 按原 tile 顺序执行全部 query-head 状态归并，删除独立 state launch。
- 正确性：60/60 回归通过；Graph/eager 前 8 token 均与 baseline 一致。
- 双路径初测：Graph TPOT/steady=12.919328/12.860288 ms，相对 Iter 44 回退约 0.231 ms；eager TPOT/steady=32.165054/31.821984 ms，steady 相对 Iter 43 改善约 0.274 ms。保留 eager 收益，Graph 改回独立 state kernel。
- Graph 分流复测：TPOT/steady=12.679392/12.668352 ms，前 8 token 一致；接近 Iter 44，但本臂尚无新增收益。最终代码仅 eager 使用末 CTA 归并，Graph 使用独立 state。
- 结果：`iter47_graph_32k_b1_8tok.json`、`iter47_eager_32k_b1_8tok.json`、`iter47_graph_splitpath_32k_b1_8tok.json` 与各臂 GPU hygiene 日志；不自行 commit。

## Iter 48：eager ranking 单 CTA 归并树

- 修改：生产 5 个 ranking tile 在 eager 下由单 CTA 依次合并全部降序 run，保留 Top-1024 并复用原 finish 数值逻辑，将 5→3、3→2、finish 三次 launch 减为一次；Graph 保留原并行归并树。
- 正确性：60/60 回归通过；eager 前 8 token 与 baseline 一致。
- 生产 eager 8-token 信号：TPOT 31.952127 ms，steady 31.621248 ms；相对 Iter 47 steady 再降约 0.201 ms。首次步含新 kernel JIT，不进入 steady 口径。
- 结果：`iter48_eager_32k_b1_8tok.json` 与 GPU hygiene 旁路日志；保留本轮，不自行 commit。

## Iter 49：eager Top-K 与 candidate expansion 融合

- 修改：eager 在每个 `(request,KV-head)` CTA 内依次完成 far/near Top-K，CTA 同步后复用原 candidate body，删除两次 kernel 之间的 host 往返；Graph 保持两路 Top-K 并行和独立 candidate。
- 正确性：60/60 回归通过；eager 前 8 token 与 baseline 一致。
- 生产 eager 8-token 信号：TPOT 31.335680 ms，steady 31.149376 ms；相对 Iter 48 steady 再降约 0.472 ms。首个计时步含大 fused kernel JIT，不进入 steady 口径。
- 结果：`iter49_eager_32k_b1_8tok.json` 与 GPU hygiene 旁路日志；保留本轮，不自行 commit。

## Iter 50：eager fine-score 末 CTA 融合 ranking tile keys

- 修改：复用 `_tile_keys_from_scores_body`，由每行最后完成的 fine-score CTA 在 softmax-state 归并后依次生成全部 ranking tile；eager 跳过独立 rank-tile launch，Graph 仍走原独立 kernel。rank buffer 由 selector 与 ranking 共用同一 workspace 项，launch key 显式包含 `fuse_rank`。
- 中间修复：初版 3/60 回归失败；逐 key 对拍发现第一次 fused tile 与独立 tile 不同、第二次一致。根因是 `FUSE_RANK` 缩进在末 CTA 条件之外，所有 fine-score CTA 并发写 ranking tile；移入 `finished == NUM_TILES - 1` 后 60/60 通过。volatile 读取试验无效并已撤销。
- 生产 eager 8-token 信号：TPOT 31.125441 ms，steady 30.893952 ms；相对 Iter 49 steady 再降约 0.255 ms，前 8 token 为 `[576,1372,374,220,19,17,17,21]`，与 baseline 一致。
- 编译成本：首个计时步 1068310.25 ms，包含 5 个 `tl.sort` 静态展开的大 fused variant JIT；不进入 steady 中位数，但必须披露，后续应尝试动态 tile 循环降低编译与代码体积。
- Graph 复测：TPOT 12.685568 ms，steady 12.652640 ms，前 8 token 一致；相对 Iter 44 best steady 差约 0.023 ms，公共 refactor 未造成显著回退，但仍高于 FA2 Graph 12.560864 ms。
- 结果：`iter50_eager_32k_b1_8tok.json`、`iter50_graph_32k_b1_8tok.json` 与 GPU hygiene 旁路日志；eager 仅见本进程 PID 254015，Graph 仅见本进程 PID 429801，未见其他进程污染；保留本轮，不自行 commit。

## Iter 51：fused ranking tile 改动态循环

- 修改：末 CTA 的 5 个 ranking tile 从 `tl.static_range` 静态展开改为运行时 `range`，复用一份 sort 代码，tile 顺序和数值语义不变。
- 正确性：60/60 回归通过；eager 前 8 token 与 baseline 一致。
- 生产 eager 8-token 信号：TPOT 30.737728 ms，steady 30.485424 ms；相对 Iter 50 steady 再降约 0.409 ms，相对 Iter 49 累计降约 0.664 ms。
- 编译成本：首个计时步从 Iter 50 的 1068310.25 ms 降至 127965.71 ms，约减少 88.0%；仍含新 fused variant JIT，不进入 steady 中位数。
- 结果：`iter51_eager_32k_b1_8tok.json` 与 GPU hygiene 旁路日志；仅见本进程 PID 464163，未见其他进程污染；保留本轮，不自行 commit。
- Nsight 补充（4-token profiler 口径，3 个 decode step）：`_nfa_fine_scores_paged_kernel` 平均 238.267 µs/层，约 8.58 ms/步，已成为 eager 最大单 kernel；page-stats/prepare-Q、gather、Top-K/candidate、merge-tree-finish 分别约 55.001、28.246、15.547、14.726 µs/层。说明 fused rank 的单 CTA 串行排序/占用是主要 GPU 浪费，不能只按 host launch 分析。画像为 `iter51_profile_eager.nsys-rep/.sqlite`。

## Iter 52：eager fused fine/ranking 提升为 8 warp（回退）

- 修改：仅 eager fused variant 从 4 warp 提到 8 warp，尝试加速单 CTA 内的 2048 项排序；Graph 保持 4 warp。
- 正确性：60/60 回归通过，前 8 token 与 baseline 一致。
- 生产 eager 8-token 信号：TPOT 31.714560 ms，steady 31.442688 ms；相对 Iter 51 steady 回退约 0.957 ms。更多 warp 降低了 fine-score/末 CTA 整体效率，未形成排序收益，撤销并恢复 4 warp。
- 结果：`iter52_eager_32k_b1_8tok.json` 与 GPU hygiene 旁路日志；仅见本进程 PID 581690，未见其他进程污染；不自行 commit。

## Iter 53：Graph split-KV 末 CTA 融合 combine（无收益，回退）

- 修改：每个 `(query,KV-head)` 行的 split CTA 以 acq_rel 原子计数，最后完成者按 split 序号归并 4 个 query heads 的 `(acc,m,z)` 并写输出；counter 归并后自清零供后续 step/replay 复用，删除独立 combine launch。eager grouped 路径不变。
- 正确性：60/60 回归通过，Graph 前 8 token 与 baseline 一致。
- 生产 Graph 8-token 信号：TPOT 12.688096 ms，steady 12.665504 ms；相对 Iter 50 steady 慢约 0.013 ms、相对 Iter 44 best 慢约 0.036 ms，未获得理论上的 1.95 µs/层收益。末 CTA 归并增加 split kernel 资源/串行尾，抵消 launch 与独立 combine 节省，撤销并恢复独立 combine。
- 结果：`iter53_graph_32k_b1_8tok.json` 与 GPU hygiene 旁路日志；仅见本进程 PID 680085，未见其他进程污染；不自行 commit。
- 回退落实：已删除 fused-combine helper/counter/分支并恢复独立 `_gather_attend_split_combine_kernel` launch；`py_compile` 与统一 60 项回归通过。

## Iter 54：eager 改并行 rank tiles + 256-token fine tile（回退）

- 画像依据：Iter 51 eager fused fine/ranking 为 156 registers、238.267 µs/层；Iter 44 Graph 拆分路径的 fine score 与 rank tiles 分别约 21.099/13.223 µs。尝试消除单 CTA 串行排序及其对整个 fine kernel 的寄存器压力。
- 修改：eager fine tile 从 128 改为 256，关闭 fine-score 末 CTA 的 rank-tile 融合，恢复独立并行 `_tile_keys_from_scores`；保留 eager fused softmax-state 和单 CTA merge-tree。Graph 语义不变。
- 正确性：60/60 回归通过；前 8 token 为 `[576,1372,374,220,19,17,17,21]`，与 baseline 一致。
- 生产 eager 8-token 信号：TPOT 31.566208 ms，steady 31.356977 ms；相对 Iter 51 steady 30.485424 ms 回退约 0.872 ms。新增 launch 与 lost fusion 的代价超过并行 kernel 的 GPU 收益，恢复 Iter 51 的 128-token fused rank tiles。
- 结果：`iter54_eager_32k_b1_8tok.json` 与 GPU hygiene 旁路日志；仅见本进程 PID 825903，未见其他进程污染；不自行 commit。

## Iter 55：decode 页统计每行单 CTA

- 修改：`q_job` decode 路径将 `BLOCKS_PER_PROGRAM` 从 16 改为 `KT_CAP`，稳态每 `(request,KV-head)` 只启动一个页统计 CTA 并仅更新最新块；非增量时同一 CTA 顺序覆盖全部块，保持重建语义。无 `q_job` 的全量路径仍为 16 块/CTA。
- 正确性：60/60 回归通过。
- 生产 Graph 8-token 信号：TPOT 12.662944 ms，steady 12.641088 ms；相对 Iter 50 公共基线 steady 12.652640 ms 改善约 0.012 ms，相对 Iter 44 best 12.629328 ms 仍慢约 0.012 ms，且仍高于 FA2 Graph 12.560864 ms。空 CTA 不是页统计的主要成本，保留候选但尚需 eager 与复测判断。
- eager 同修改复测：TPOT 30.855265 ms，steady 30.534752 ms；相对 Iter 51 best steady 30.485424 ms 慢约 0.049 ms，未形成收益。
- 结论：双路径均无稳健收益，且会牺牲非增量重建并行度，恢复 16 块/CTA。
- 结果：`iter55_graph_32k_b1_8tok.json`、`iter55_eager_32k_b1_8tok.json` 与 GPU hygiene 旁路日志；不自行 commit。

## Iter 56：删除非 head-0 fine/state/ranking 死计算

- 停滞复评：Iter 53--55 无稳健收益后重新检查当前 cProfile、Nsight 与全部迭代方向；发现 `_final_union_physical_kernel` 只消费 request-level KV head 0 的 `middle/counts`，其余 7 个 KV heads 的 fine score、softmax state、rank tiles 和 merge-tree 均无 downstream 消费者。
- 修改：candidate/L1 仍按原几何生成，进入 fine 前将 candidates/owners/generations/counters 切到 head 0；fine 支持 `HKV=1` 视图，rank workspace 改为每 request 一行，final union 的 middle stride 改为 `HKV=1`。head 0 的 Q/K 地址、candidate 顺序、BF16 中间舍入和 ranking 算法均不变。
- 正确性：60/60 回归通过。
- 生产 Graph 8-token 信号：TPOT 12.099008 ms，steady 12.087344 ms；相对 Iter 44 best steady 12.629328 ms 降低约 0.542 ms，低于 FA2 Graph 12.560864 ms 约 0.474 ms。该短口径首次超过 Graph 目标，但仍须完整 128-token 验收。
- eager 同修改复测：TPOT 30.834175 ms，steady 30.749248 ms；相对 Iter 51 steady 30.485424 ms 慢约 0.264 ms，尚未转化为 eager 收益，需重新画像确认 head-0 fused variant 的实际 kernel/host 构成。
- 结果：`iter56_graph_32k_b1_8tok.json`、`iter56_eager_32k_b1_8tok.json` 与 GPU hygiene 旁路日志；Graph/eager 前 8 token 均与 baseline 一致，未见其他进程污染；不自行 commit。

## Iter 57：head-0-only 上推到 L1/Top-K/candidate

- 修改：L1 输出与 grid 从每 request 8 个 KV-head 行改为仅 head 0；输入统计仍使用原 stride，`pid_h=batch` 与原 head-0 Q 地址完全一致。下游 far/near Top-K、candidate expansion、owner/generation 因此也只计算 head 0，不再先算 8 个头再切片。
- 正确性：60/60 回归通过。
- 生产 Graph 8-token 信号：TPOT 12.046208 ms，steady 12.036560 ms；相对 Iter 56 steady 再降约 0.051 ms，低于 FA2 Graph 12.560864 ms 约 0.524 ms。
- eager 同修改复测：TPOT 31.214464 ms，steady 30.694080 ms；相对 Iter 56 eager steady 改善约 0.055 ms，但仍未优于 Iter 51 的 30.485424 ms。
- 结果：`iter57_graph_32k_b1_8tok.json`、`iter57_eager_32k_b1_8tok.json` 与 GPU hygiene 旁路日志；Graph/eager 前 8 token 均与 baseline 一致，未见其他进程污染；不自行 commit。
- Nsight 对照：head-0-only eager 的 fused fine/ranking 从 238.267 降至 145.409 µs/层，但 grid 已从 `(8,67)` 变为 `(1,67)`；减少其他头只降低总工作量，head-0 串行排序尾仍在。画像为 `iter57_profile_eager.nsys-rep/.sqlite`。

## Iter 58：head-0 eager 拆出并行 rank tiles（回退）

- 修改：仅 eager 关闭 fine 末 CTA 的 rank-tile 融合，保持 128-token fine tile、fused softmax-state 和单 CTA merge-tree；与 Iter 54 不同，不改变 tile 几何，且此时只计算 head 0。
- 正确性：60/60 回归通过。
- 生产 eager 8-token 信号：TPOT 32.466625 ms，steady 32.328192 ms；相对 Iter 57 steady 30.694080 ms 回退约 1.634 ms。即使 GPU 串行尾可缩短，新增独立 launch/host 成本仍显著更高，恢复 fused rank tiles。
- 结果：`iter58_eager_32k_b1_8tok.json` 与 GPU hygiene 旁路日志；前 8 token 与 baseline 一致，仅见本进程 PID 1226958；不自行 commit。

## Iter 59：最后 5 个 fine CTA 并行 rank tiles（回退）

- 修改：最末 CTA 归并 state 后 release 发布；其余最后 4 个完成的 CTA acquire 等待并各处理一个 rank tile，最末 CTA 处理最后一组。等待仅发生在所有 fine CTA 已完成后，rank tile 间无语义顺序。
- 正确性：60/60 回归通过。
- 生产 eager 8-token 信号：TPOT 31.557344 ms，steady 31.291615 ms；相对 Iter 57 steady 30.694080 ms 回退约 0.598 ms。原子自旋、分支与调度成本超过排序并行收益，恢复最末 CTA 顺序 rank tiles。
- 结果：`iter59_eager_32k_b1_8tok.json` 与 GPU hygiene 旁路日志；前 8 token 为 `[576, 1372, 374, 220, 19, 17, 17, 21]`，与 baseline 一致；GPU0 仅见本进程 PID 26262。回退后重新执行 `py_compile` 与统一回归，60/60 通过；不自行 commit。

## Iter 60：并行 rank tiles + 末 CTA 原地 merge-tree（回退）

- 修改：eager fine 仅融合 softmax-state；独立 rank kernel 的 5 个 CTA 并行生成 tiles，最后完成的 CTA 通过 release/acquire 计数直接执行原顺序 merge-tree 与 finish。与原路径同为两次 launch，且没有 Iter 59 的自旋等待；counter base 显式使用 padding 前的 `fine_num_tiles`。
- 正确性：初版在 padding 后计算 counter base，导致末 CTA 判定不成立、2/60 失败；修正后完整 60/60 回归通过。
- 生产 eager 8-token 信号：TPOT 31.160255 ms，steady 31.125216 ms；相对 Iter 57 steady 30.694080 ms 回退约 0.431 ms，但优于 Iter 58 的独立 rank + 独立 merge。并行排序及 launch 数守恒仍不足以抵消 fine/rank 拆分的调度与 host 成本，恢复 Iter 57 的 fine 末 CTA 顺序 rank + 独立 merge。
- 结果：`iter60_eager_32k_b1_8tok.json` 与 GPU hygiene 旁路日志；前 8 token 为 `[576, 1372, 374, 220, 19, 17, 17, 21]`，与 baseline 一致；GPU0 仅见本进程 PID 1604032；不自行 commit。

## Iter 61：eager 最后一个 rank tile 从 2048 缩为 1024（回退）

- 修改：生产 R=8452 的最后一个 rank 组仅 260 个真实槽，原实现仍排序 2048 个 key。新增严格等价的小尾块路径：前 4 组保持 2048 排序，末组仅排序包含全部真实槽的 1024 个 key；由于其余位置均为零 key，前 1024 个输出与原实现逐项一致。非小尾块几何保留原路径。
- 正确性：60/60 回归通过。
- 生产 eager 8-token 信号：TPOT 30.961952 ms，steady 30.764977 ms；相对 Iter 57 steady 30.694080 ms 没有收益（慢约 0.071 ms，属噪声量级），预期的排序缩减未转化为端到端收益，恢复原统一 2048 尾块路径。
- 结果：`iter61_eager_32k_b1_8tok.json` 与 GPU hygiene 旁路日志；前 8 token 为 `[576, 1372, 374, 220, 19, 17, 17, 21]`，与 baseline 一致；GPU0 仅见本进程 PID 1696886；不自行 commit。

## Iter 62：eager fine 末 CTA 继续融合 merge-tree/finish（回退）

- 修改：末 CTA 顺序生成全部 5 个 rank tiles 后，直接在同一 CTA 调用抽出的 `_merge_tree_finish_body`，省掉独立 `_merge_tree_finish` launch；保持原 tile、merge 和 finish 数值顺序，Graph 仍走独立 ranking 路径。
- 正确性：`py_compile` 通过，统一回归 60/60 通过；前 8 token 为 `[576, 1372, 374, 220, 19, 17, 17, 21]`，与 baseline 一致。
- 生产 eager 8-token 信号：TPOT 31.484800 ms，steady 31.406880 ms；相对 Iter 57 steady 30.694080 ms 回退约 0.713 ms。merge/finish 融入大型 fine kernel 的寄存器与调度代价超过一次 launch 节省，恢复 Iter 57 的独立 merge-tree/finish。
- 结果：`iter62_eager_32k_b1_8tok.json` 与 GPU hygiene 旁路日志；GPU0 仅见本进程 PID 1960084，未见其他进程污染；不自行 commit。

## 128-token 正确性审计：当前 Graph 不得验收

- Iter 57 Graph 的 `iter57_graph_32k_b1_128tok.json` 虽测得 TPOT/steady 均为约 11.815 ms，但与初始 `baseline_32k_b1_graph.json`、`baseline_32k_b1_eager.json` 对照均在生成索引 11 首次分叉；当前序列为 `... 21, 22, 624, 785, ...`，baseline 为 `... 21, 22, 13, 576, ...`。因此该低 TPOT 不能作为超过 FA2 的有效候选。
- 初始 `candidate_32k_b1_graph.json` 与 `candidate_32k_b1_eager.json` 均与对应 baseline 128/128 一致，说明分叉由此后保留优化引入；此前 8-token 前缀门禁不足，后续候选至少增加 16/32-token 早期门禁，最终仍须 128/128。
- 上述 Graph 128-token receipt 的 hygiene 命令误用非法 `gpu_index` compute-apps 字段，监控日志无效；正确性修复后必须用合法 GPU index/memory + GPU UUID/PID/memory 双查询重跑正式版本。
- Iter 62 回退后已重新通过 `py_compile` 和统一回归 60/60；下一步用同生产输入的 Graph 16-token receipt 复现索引 11 分叉并做最小化 bisection。
- `identity_iter57_graph_32k_b1_16tok.json` 已稳定复现：相对两个初始 baseline 均 `first_diff=11`，当前 16-token 为 `[576, 1372, 374, 220, 19, 17, 17, 21, 15, 21, 22, 624, 785, 3281, 10963, 1372]`；对应 hygiene 日志仅见本进程 PID 2207389。后续 bisection 可用 16-token 作为充分复现口径。
- `identity_iter57_eager_32k_b1_16tok.json` 与初始 eager/Graph baseline 的前 16 token 均逐项一致，索引 11 保持 baseline 的 `13`；TPOT 30.530657 ms、steady 30.502128 ms。分叉范围因此收敛到 Graph-only 路径，优先单变量检查 split-KV，其次 Graph fine tile=256 与独立 state/ranking。
- **首个根因已定位：Graph-only split-KV。** 仅将 `split_kv=False`，其他 Iter 57 结构不变，`identity_graph_nosplit_32k_b1_16tok.json` 与初始 baseline、当前 eager 的前 16 token 均逐项一致，索引 11 恢复为 `13`。Iter 43/44 的 8 分区 online-softmax partial 归并改变了原 grouped kernel 的浮点归约顺序，足以在长序列索引 11 翻转 argmax。该 16-token 臂 TPOT 12.626592 ms、steady 12.620576 ms；继续保持禁用 split-KV，并进行完整 128-token 验收。
- 禁用 split 后统一回归仍为 60/60；但 `identity_graph_nosplit_32k_b1_128tok.json` 在索引 45 再次偏离两个初始 baseline，证明至少还有一个更晚暴露的数值差异，16-token 不能替代最终 128-token 门禁。该完整臂 TPOT 12.593216 ms、steady 12.592320 ms，GPU0 仅见本进程 PID 2358755；下一步用 48-token 复现第二分叉，并检查 eager 是否同样受影响。
- `identity_iter57_eager_32k_b1_48tok.json` 同样在索引 45 偏离初始 eager baseline，且其前 48 token 与禁用 split 的 Graph 完全一致；TPOT 30.378241 ms、steady 30.361920 ms。第二分叉属于 eager/Graph 公共数值路径，而非剩余 Graph 分流。下一单变量实验将 fine score/softmax tile 统一恢复为 baseline 的 64，判别 Iter 2/32 的跨 tile 归并顺序。
- `identity_eager_tile64_32k_b1_48tok.json` 仍在索引 45 分叉，且与 tile128 eager 的 48-token 序列完全一致；TPOT 31.418207 ms、steady 31.401888 ms。仅恢复 64 tile 不能修复第二分叉，保持 tile64 后继续单变量恢复 baseline 标量 FP32 点积，判别 Iter 10 的 Tensor Core 累加顺序。
- **第二根因已定位：Iter 10 的 Tensor Core fine 点积。** `identity_eager_tile64_scalar_32k_b1_48tok.json` 与初始 eager baseline 前 48 token 完全一致，索引 45 恢复为 `8110`；相对同 tile 的 Tensor Core 臂仅改回逐 query head `tl.sum(k*q)`。该臂 TPOT 30.866592 ms、steady 30.856736 ms。下一步恢复 eager128/Graph256 tile，确认点积修复不依赖 tile64。
- `identity_eager_scalar_32k_b1_48tok.json` 使用 eager128 tile 后仍与 baseline、tile64 scalar 前 48 token 完全一致；TPOT 30.953953 ms、steady 30.914624 ms。因此保留 eager128/Graph256 分流，仅修复点积与 split-KV；统一回归再次 60/60 通过，开始完整 128-token 双路径身份验收。
- **eager 128/128 identity 已恢复。** `identity_eager_scalar_32k_b1_128tok.json` 与初始 eager、Graph baseline 的完整 128 token 均逐项相同；TPOT 30.591265 ms、steady 30.585328 ms，GPU0 仅见本进程 PID 2690957。该 receipt 证明 Iter 10 Tensor Core fine 点积是第二分叉根因，但 eager 性能仍未达到 FA2。
- **Graph 128/128 identity 也已恢复。** `identity_graph_scalar_nosplit_32k_b1_128tok.json` 与初始 baseline、修复后的 eager 完整 128 token 均逐项相同；TPOT 12.589600 ms、steady 12.589280 ms，GPU0 仅见本进程 PID 2755259。当前正确性基线相对 FA2 Graph 12.560864 ms 仅慢 0.028416 ms，后续优化必须保留 scalar fine 点积与禁用 split-KV 两个修复。

## Iter 63：B1 gather attention 按 query head 增加 CTA（回退）

- 修改：保持每个 query head 的 token-block 顺序与 grouped online-softmax 不变，仅将 B1 grid 从 8 个 KV-head CTA 扩为 32 个 query-head CTA；不使用 split-KV，意图通过增加 occupancy 补回正确性修复的性能。
- 正确性：`py_compile` 与统一回归 60/60 通过。
- 生产 Graph 16-token 信号：`iter63_graph_32k_b1_16tok.json` TPOT 12.710080 ms、steady 12.700576 ms；相对同口径禁用 split 的 12.620576 ms 回退约 0.080 ms。每个 CTA 重读同一 KV-head 的 K/V，失去原 4 个 query head 共享加载的收益，超过 occupancy 改善，恢复原 grouped B1 路径；不自行 commit。

## Iter 64：B1 grouped gather 从 4 warp 提升到 8 warp（回退）

- 修改：仅调整 B1 grouped gather 的 warp 数；BT=128、token 遍历顺序、每 head online-softmax 和 Tensor Core 数据路径不变。
- 正确性：Iter 63 回退后与本修改共同通过 `py_compile` 和统一回归 60/60。
- 生产 Graph 16-token 信号：`iter64_graph_32k_b1_16tok.json` TPOT 12.806240 ms、steady 12.806016 ms；相对默认 4 warp 的 12.620576 ms 回退约 0.185 ms。更多 warp 降低小 GQA tile 的资源效率，恢复默认 4 warp；不自行 commit。
- 回退落实：已恢复 `kernel_kwargs = {}`，`py_compile` 与统一回归 60/60 通过；未保留 Iter 62--64 实验代码，不自行 commit。

## 正确基线停滞复评：scalar + no-split 双口径画像

- 稳态 eager 单步 cProfile：GraphRunner.step 总采样 38.33 ms，其中 `_layer_batched_select_impl` 11 ms、`_fast_launch` 252 次共 4 ms。画像会放大 host 成本，只用于定位，不作为 TPOT。
- Nsight 最后 36 次/层均值：eager fused fine/state/rank 为 163.131 µs，独立 merge-tree/finish 为 14.469 µs；Graph 的 fine/state/rank 拆分路径中 fine 仅 9.221 µs、rank tiles 13.113 µs，表明 eager 单 CTA 串行 ranking 尾是主要 GPU 浪费。
- eager 最后 36 层窗口 GPU busy 16.413 ms、GPU 空隙 21.638 ms（nsys 放大口径）；selector+gather 每层中位数为 238.848 µs busy + 107.904 µs gap。后续必须同时处理串行 ranking 和 host 分发，不再继续调 warp。
- 画像文件：`scalar_nosplit_steady_eager_cprofile.prof/.json`、`scalar_nosplit_profile_eager.nsys-rep/.sqlite`、`scalar_nosplit_profile_graph.nsys-rep/.sqlite` 及 hygiene 日志；仅见对应本进程。

## Iter 65：eager fused softmax-state 按 GQA lane 向量化

- 修改：末 CTA 的 G 个 query-head 状态由外层串行循环改为 `G_PAD` lanes；每个有效 head 内仍按 tile 0..NUM_TILES-1 顺序执行完全相同的 FP32 max/exp/乘加， inactive lanes 掩码读写，不改变跨 tile 归约顺序或 BF16 中间舍入。
- 正确性：`py_compile` 与统一回归 60/60 通过；前 8 token 为 `[576,1372,374,220,19,17,17,21]`，与 baseline 一致。
- 生产 eager 8-token 信号：`iter65_eager_32k_b1_8tok.json` TPOT 30.880417 ms、steady 30.692351 ms；与近期同口径约 30.6--30.7 ms 基本持平，未形成可辨识端到端收益，符合 eager host 空隙掩盖小部分 GPU state 收益的画像。
- 结果：GPU0 仅见本进程 PID 3275344，未见其他进程污染；不自行 commit。该改动数值等价且代码更简单，暂保留作为后续结构优化的公共基线，但最终不作为独立性能收益宣称。

## Iter 66：eager merge-tree/finish 融合 final union（回退）

- 修改：生产多 rank-group eager 路径由独立 `_merge_tree_finish` + `_final_union_physical_kernel` 改为同一 request CTA 连续执行；同时绕过通用 ranking host 包装，减少一次 launch。Graph 与短序列小组几何不变。
- 正确性：`py_compile` 与统一回归 60/60 通过。
- 生产 eager 8-token 信号：`iter66_eager_32k_b1_8tok.json` TPOT 31.118719 ms、steady 30.925296 ms；相对 Iter 65 steady 30.692351 ms 回退约 0.233 ms。union 的宽排序/寄存器压力污染 merge kernel，抵消 launch 与 host 包装节省，恢复独立 merge/finish 与 final union；不自行 commit。
- 回退落实：删除 fused merge/union kernel、恢复通用 ranking 与独立 final union，`py_compile` 和统一回归 60/60 通过；保留 Iter 65，不自行 commit。

## Iter 67：固定几何 eager selector 调用计划重放

- 轻量诊断：无 cProfile 的单稳态步 `perf_counter` 包装显示 selector host 约 8.435 ms，其中 `_fast_launch` 合计 3.990 ms，helper 参数/形状/workspace 组织约 4.445 ms；包装本身会放大总 TPOT，只用于归因。记录于 `scalar_nosplit_steady_eager_wallclock.json`。
- 修改：首次 eager 调用记录固定几何下 page-stats/L1/candidate/fine/ranking/final-union 的 launch 参数；后续步在 geometry 令牌完全一致时仅替换当前 Q/K/V（及 slot）并原序重发。这里不是 CUDA Graph capture/replay，所有 Triton kernel 仍逐次 eager launch；短 prompt 的临时 padding 路径不缓存，Graph 路径不使用该计划。
- 正确性：`py_compile` 与统一回归 60/60 通过；kernel 序列、参数和数值路径不变。
- 生产 eager 8-token 信号：`iter67_eager_32k_b1_8tok.json` TPOT 27.545631 ms、steady 27.378576 ms；相对 Iter 65 steady 30.692351 ms 降低 3.313775 ms，验证 helper host 组织是主要可消除浪费。距 FA2 eager 22.915487 ms 仍约 4.463 ms，保留本轮，不自行 commit。

## Iter 68：调用计划上拆出并行 rank tiles（回退）

- 修改：eager fine 不再由末 CTA 串行生成 5 个 rank tiles，改由独立 kernel 并行生成；与 Iter 58 不同，新增 launch 已进入 Iter 67 调用计划，不再支付 helper host 包装。Graph 本就走独立 rank，不受影响的公共数值逻辑不变。
- 正确性：`py_compile` 与统一回归 60/60 通过。
- 生产 eager 8-token 信号：`iter68_eager_32k_b1_8tok.json` TPOT 28.288448 ms、steady 27.929104 ms；相对 Iter 67 steady 回退约 0.551 ms。当前 eager 仍有大量 host 空隙，串行 rank 的 GPU 时间并不完全暴露在关键路径；新增 launch 反而增加关键路径成本，恢复 eager fused rank；不自行 commit。

## Iter 69：预绑定 eager CompiledKernel.run 调用计划

- 修改：在 Iter 67 的固定几何调用计划中进一步缓存 `CompiledKernel.run`、function、packed metadata、完整 launch 参数元组及 grid 尾维；稳态在无 enter hook 时直接调用预绑定 run，绕过 `_fast_launch` 的字典查找、specialization 和参数重组。当前 Q/K/V 与 slot 仍按原参数位置替换，kernel 顺序和数值路径不变。
- 正确性：`py_compile` 与统一回归 60/60 通过；前 8 token 为 `[576,1372,374,220,19,17,17,21]`，与 baseline 一致。该短序列门禁不能替代后续 48/128-token identity。
- 生产 eager 8-token 信号：`iter69_eager_32k_b1_8tok.json` TPOT 26.638624 ms、steady 26.229856 ms；相对 Iter 67 steady 27.378576 ms 降低 1.148721 ms，相对 Iter 65 累计降低 4.462496 ms。距 FA2 eager 22.915487 ms 仍约 3.314369 ms，保留本轮，不自行 commit。
- hygiene：`iter69_eager_32k_b1_8tok_gpu_hygiene.log` 仅含起跑前 GPU 显存样本，未记录运行期 compute-apps，因此本轮没有全程独占 GPU 的旁路证明；后续正式门禁必须修正监控并保留完整日志。
- 48-token 延伸门禁：`identity_iter69_eager_32k_b1_48tok.json` 与 `identity_eager_scalar_32k_b1_128tok.json` 前 48 token 逐项一致，已覆盖此前索引 45 分叉点；TPOT 25.931904 ms、steady 25.931007 ms。完整 hygiene 日志仅见本进程 PID 3763409，未见其他计算进程。最终仍须完整 128-token identity/TPOT，不能用该结果提前验收。
- Graph 兼容性门禁：`identity_iter69_graph_32k_b1_48tok.json` 与正确 Graph 基线前 48 token 逐项一致；TPOT 12.743520 ms、steady 12.739648 ms。hygiene 仅见本进程 PID 3840433。该数值比正确基线 12.589280 ms 慢约 0.150 ms，说明当前尚不能宣称 Graph 超过 FA2；需用后续优化与正式复测区分 run-to-run 波动和实际回退。

## Iter 70：稳态跳过已消费的 first-decode seed 回写

- 修改：`mark_first_decode_seeded` 先过滤 `state.first_decode_seeded=True` 的请求；首步之后的稳态不再创建 GPU 索引张量、执行高级索引写入或重复设置 Python 状态。首步消费语义不变，eager 与 Graph 的 step 边界均受益。
- 正确性：`py_compile` 与统一回归 60/60 通过。
- 生产 eager 8-token 信号：`iter70_eager_32k_b1_8tok.json` TPOT 27.518528 ms、steady 26.660384 ms；相对 Iter 69 的 26.229856 ms 慢约 0.431 ms。严格删除的稳态工作未转化为可辨识收益，本轮测量存在运行波动；先保留无语义变化的短路，待后续同口径复测判断，不据此宣称性能改善，不自行 commit。
- 身份与 hygiene 审计：前 8 token 已与正确 eager baseline 核对一致；`iter70_eager_32k_b1_8tok_gpu_hygiene.log` 仅含起跑前 GPU 显存样本，未记录运行期 compute-apps，不能作为全程独占证明。

## Iter 71：page stats 只计算实际消费的 KV head 0

- 修改：融合 page-stats/prepare-Q/KV-write kernel 中，所有 KV head 仍按原路径写当前 K/V，但仅 head 0 执行页 min/max/sum。当前 `_l1_scores` 已固定 `Hkv=1`，final canonical union 也只消费 head 0，因此其余 7 个头的页统计是没有消费者的死计算；head 0 的加载、归约和 BF16 写回顺序完全不变。
- 正确性：`py_compile` 与统一回归 60/60 通过。
- 生产 Graph 48-token 信号：`iter71_graph_32k_b1_48tok.json` TPOT 12.473088 ms、steady 12.469584 ms；相对 Iter 69 Graph 同长度 steady 12.739648 ms 降低 0.270064 ms，并比 FA2 Graph 12.560864 ms 低 0.091280 ms。该结果首次在 48-token 口径越过 FA2 Graph，但最终仍须完整 128-token identity/TPOT 验收，不自行 commit。
- Graph 身份与 hygiene 审计：前 48 token 已与正确 Graph 128-token baseline 逐项核对一致，覆盖此前索引 11 和 45 分叉点；`iter71_graph_32k_b1_48tok_gpu_hygiene.log` 运行期仅见本进程 PID 3955580，未见其他计算进程。
- 生产 eager 48-token 信号：`iter71_eager_32k_b1_48tok.json` TPOT 25.561792 ms、steady 25.546400 ms；相对 Iter 69 eager 同长度 steady 25.931007 ms 降低 0.384607 ms。距 FA2 eager 22.915487 ms 仍约 2.630913 ms，继续优化 eager host/launch，不自行 commit。
- eager 身份与 hygiene 审计：前 48 token 已与正确 eager 128-token baseline 逐项核对一致；`iter71_eager_32k_b1_48tok_gpu_hygiene.log` 仅含起跑前样本，未记录运行期 compute-apps，不能作为全程独占证明。

## Iter 72：固定几何 eager grouped gather 高层调用重放

- 修改：复用 selector 已建立的固定几何、launch context 与 grouped gather 的 `CompiledKernel.run` 计划；稳态跳过 gather 前的重复形状/几何检查和 `_fast_launch` 包装，仅替换当前 Q 与本次新分配的输出。kernel、默认 4 warp、BT=128、参数顺序和 online-softmax 数值路径均不变，Graph capture 不走该 eager 捷径。
- 分层修复：首版计划全局共享，导致后续层误用首层 KV pool/gather buffer，48-token 在索引 1 分叉；改为按 `id(cache_layer)` 隔离每层计划后重新通过 `py_compile` 与统一回归 60/60。错误臂 receipt 已被修复后的同名单次生产运行覆盖，不作为性能证据。
- 生产 eager 48-token 信号：`iter72_eager_32k_b1_48tok.json` TPOT 25.122112 ms、steady 25.114016 ms；相对 Iter 71 同长度 steady 25.546400 ms 降低 0.432384 ms，距 FA2 eager 22.915487 ms 仍约 2.198529 ms。保留本轮，不自行 commit。
- 身份与 hygiene 审计：修复后前 48 token 与正确 eager 128-token baseline 逐项一致；2 秒旁路日志仅见本进程 PID 4148580，未见其他计算进程。

## Iter 73：eager selector 公共入口直接重放固定几何计划

- 修改：把 Iter 67/69 的预绑定重放提取为公共 helper，并在 `_layer_batched_select_impl` 之前尝试执行。生产 Cache 配置在 decode 期间固定；稳态仍检查非 capture、batch、visible bucket、Q shape/dtype/alignment/strides，满足后跳过页表、子空间、scale 与 ranking host prelude，失配则回退完整路径。kernel 顺序和数值路径不变。
- 正确性：`py_compile` 与统一回归 60/60 通过。
- 生产 eager 48-token 信号：`iter73_eager_32k_b1_48tok.json` TPOT 24.679585 ms、steady 24.652112 ms；相对 Iter 72 steady 25.114016 ms 降低 0.461904 ms，距 FA2 eager 22.915487 ms 仍约 1.736625 ms。保留本轮，不自行 commit。
- 身份与 hygiene 审计：前 48 token 与正确 eager 128-token baseline 逐项一致；2 秒旁路日志仅见本进程 PID 4185717，未见其他计算进程。
- 主机画像：`iter73_eager_cprofile.prof` 确认 252 次生产 selector 中 216 次走入口重放、仅首步 36 次进入完整 `_layer_batched_select_impl`；cProfile 将 8-token steady 放大到 29.550064 ms，只用于验证路径与归因，不作为 TPOT。

## 当前版本 Graph 128-token 生产口径复验

- `iter73_graph_32k_b1_128tok.json`：127 次 Graph replay，TPOT 12.338400 ms、steady 12.338160 ms；相对 FA2 Graph 12.560864 ms 为约 1.018x。当前仅略快于 FA2，距用户要求的 Graph 2x（约 6.280432 ms）仍有明显差距，继续以 Graph 为唯一生产优化口径。
- 完整身份与 hygiene 审计：128/128 token 与初始 Graph baseline、正确 Graph baseline、正确 eager baseline 均逐项一致；2 秒旁路日志全程仅见本进程 PID 60518，未见其他计算进程。后续全量实验在 Graph 性能优化收官后启动。

## Iter 74：每 KV group 独立 Top-K 改为 radix-select + 精确压实

- 背景：P0 修复后生产选择恢复为每个 KV group 独立打分和 Top-K，旧 head-0 共享选集的性能数字不再适用。pc40 同输入基线为 eager 27.375008 ms、Graph 23.588048 ms steady TPOT；`_nfa_filtered_topk_kernel` 重新成为主要 selector 热点。
- 修改：保留 signed-BF16 ordered key、`S-token` 同分优先级、quota/无效项/补洞 finish 语义。对 `TOKEN_BITS=16` 用 32 次二分归约定位第 K 个 packed key，再以 `tl.cumsum` 精确压实大于阈值与等于阈值的元素，最后复用原 finish；`TOKEN_BITS=32` 仍走原单 CTA `tl.topk`。
- 正确性：新增原 kernel 与 radix 路径的直接 bit-exact 对拍，覆盖大量 BF16 同分、`-inf`、区间外 token、`quota=0`、有效候选不足、多 request 和多 KV group；相关 selector/cache/GraphRunner 回归 35/35 通过。
- 算子微基准：真实生产形状 `[M=1,Hkv=8,R=8320]`、K=1024、capacity=32768，原实现 0.230451 ms，候选 radix+finish 0.132365 ms，**1.741x**；选中 indices 与 counts bit-exact。该数字只描述 Top-K 算子，不等同完整 decode 加速。
- 生产 Graph 16-token 迭代信号：baseline steady 23.588048 ms，iter74 steady 15.390720 ms，约 **1.532x**；candidate TPOT median 15.397056 ms，15 次 Graph replay。结果文件为 `/tmp/nfa_hisa_baseline_graph.json` 与 `/tmp/nfa_hisa_iter74_graph.json`（pc40）。
- 尚未完成：iter74 eager、同机 FA2 eager/Graph、完整生成 token identity 与运行期 hygiene 仍在同一独立测量任务中收集；16-token Graph 结果不作为最终 2x 或 128-token 验收。
- 回收补充：iter74 eager TPOT median 25.032255 ms、steady 24.995888 ms，相对 baseline eager steady 27.375008 ms 为约 **1.095x**。Graph/eager 两侧旁路日志各仅见对应本进程，未见其他计算进程。
- **生产 identity 失败**：回收 JSON 的完整 `generated_token_ids` 后，iter74 eager/Graph 彼此一致，但二者均在生成索引 1 偏离对应 baseline；baseline token 为 `[576,1372,374,...]`，iter74 为 `[576,3281,10963,...]`。远端源码审计确认 baseline/candidate 的 `sparse_attn` 只有 `nfa_layer_batch_selector.py` 存在非 pycache 差异，因此不能把分叉归因于其他优化文件。合成同分/quota 对拍与随机微基准不足以覆盖真实选择状态，iter74 的生产 TPOT 只能作为失效实验记录，不能验收。

## Iter 75：四趟 8-bit histogram radix-select（回退）

- 修改：将 Iter 74 的 32 次全行二分归约改为四趟 8-bit `tl.histogram`，每趟在 512 bins（256 个无效填充槽）上定位第 K 个 packed key 的一个字节；后续压实与 finish 完全不变。
- 正确性：直接 bit-exact 对拍通过；微基准再次确认 indices/counts 与参考完全一致。
- 算子微基准：原实现 0.230634 ms，histogram 候选 0.213023 ms，仅 1.083x，且显著慢于 Iter 74 的 0.132365 ms。四趟 512-bin histogram/cumsum 的寄存器和归约成本超过减少全行扫描的收益，本轮回退，不运行完整 eager/Graph 生产臂。

## Iter 76：八趟 4-bit histogram radix-select（回退）

- 修改：把 Iter 75 的 4×512-bin 直方图缩窄为 8×32-bin（16 个有效 nibble 槽），减少每趟 histogram/cumsum 宽度；其余选择语义不变。
- 正确性：直接 bit-exact 对拍和微基准 indices/counts 对拍均通过。
- 算子微基准：原实现 0.230320 ms，4-bit 候选 0.160927 ms，1.431x；虽优于 8-bit histogram，但仍慢于 Iter 74 二分的 0.132365 ms。回退多趟 histogram 方向，继续以 Iter 74 为最佳 Top-K 实现。

## Iter 77：2048-token tile 部分 Top-K + 精确归并

- 修改：每个 KV group 的候选按 2048-token tile 分给独立 CTA，各 tile 用原 `tl.topk` 保留局部 K=1024 个 packed key；第二个 kernel 对全部局部结果再做原 `tl.topk` 和 finish。全局前 K 项必然包含于各 tile 的局部前 K 项，packed key、同分优先级、quota 和补洞语义均不变。
- 正确性：合成同分/quota/无效候选直接对拍通过，微基准 indices/counts bit-exact。Iter 74 已证明这些检查不足以替代生产 identity，本轮必须继续通过真实 Qwen 16/128-token 门禁才可保留。
- 算子微基准：原实现 0.230504 ms，候选 0.115675 ms，**1.993x**；比 Iter 74 的 0.132365 ms 再快约 12.6%。该结果仅限 Top-K 算子，不作为完整 decode 加速结论。
- 完整回归与生产身份：selector/cache/GraphRunner 35/35 通过；真实 Qwen eager/Graph 16/16 token 均与对应 baseline 完全一致，且两种 execution 一致。与 Iter 74 不同，本轮通过了真实生产 identity 早期门禁；最终仍需 128-token。
- 生产 16-token TPOT：eager median/steady 27.293312/27.092000 ms，Graph 15.650048/15.634048 ms。相对 baseline Graph steady 23.588048 ms 为 **1.508x**，但相对 FA2 Graph 12.602576 ms 仍慢约 24.1%；eager 也仍慢于 FA2 23.725280 ms。暂保留为当前最佳正确实现，继续优化。

## Iter 78：8192-token tile 缩减归并输入

- 修改：部分 Top-K tile 从 2048 提到 8192；真实 8320 个候选只产生 2 个局部结果集，merge 有效输入由 5120 降至 2048，`MERGE_PAD` 同步由 8192 降至 2048。两阶段包含性、packed key 和 finish 语义均不变。
- 正确性：合成同分/quota/无效候选直接对拍 1/1 通过，微基准 indices/counts bit-exact。
- 算子微基准：原实现 0.230623 ms，候选 0.115244 ms，**2.001x**；相对 Iter 77 的 0.115675 ms 仅快约 0.37%，属于很薄的算子信号，必须经完整回归和真实 Qwen identity/TPOT 判断是否保留。
- 状态：暂作为候选保留；尚未据此宣称生产加速或最终达标。
- 完整回归与生产身份：selector/cache/GraphRunner 35/35 通过；真实 Qwen eager/Graph 的 B1 `generated_token_ids` 均与 Iter 77 完全一致。
- 生产 16-token TPOT：eager median/steady 25.854368/25.773280 ms，Graph 17.468160/17.451103 ms。eager 表面优于 Iter 77，但 Graph steady 比 Iter 77 的 15.634048 ms 慢约 11.6%；微基准收益过薄且主 Graph 口径明确回退。
- 判决：恢复 `tile=2048`，不保留 8192 配置； eager 数字不作为保留候选的理由。

## Iter 79：4096-token tile 平衡局部选择与归并

- 修改：部分 Top-K tile 从 2048 调整为 4096；8320 个候选形成 3 个局部结果集，merge 有效输入为 3072、`MERGE_PAD=4096`，避免 Iter 77 的 8192-wide 归并和 Iter 78 的大局部 tile。
- 正确性：合成同分/quota/无效候选直接对拍 1/1 通过，微基准 indices/counts bit-exact。
- 算子微基准：原实现 0.230724 ms，候选 0.091244 ms，**2.529x**；相对 Iter 77 的 0.115675 ms 降低约 21.2%。该结果仅限 Top-K 算子，需完整回归与真实 Qwen 双口径验收。
- 完整回归与生产身份：selector/cache/GraphRunner 35/35 通过；真实 Qwen eager/Graph 16/16 token 均与 Iter 77 完全一致，且两种 execution 一致。
- 生产 16-token TPOT：eager median/steady 25.827711/25.787408 ms，Graph 15.497696/15.465488 ms。Graph 相对 Iter 77 steady 15.634048 ms 再改善约 1.08%，相对原 baseline 23.588048 ms 为 **1.525x**；仍慢于 FA2 Graph 12.602576 ms，eager 也仍慢于 FA2 23.725280 ms。
- 判决：保留为当前最佳正确实现；算子明显下降但完整 Graph 收益较薄，继续处理剩余 selector 与模型路径热点。

## Iter 80：利用有序 partial run 做 bitonic 归并

- 修改：partial kernel 已输出降序 run，merge 不再对 `NTILES*K` 项重新执行 `tl.topk`。逐条读入 run，反转后与当前结果组成 bitonic 序列，跨 K 比较仅保留前 K 项，再以 `log2(K)` 轮 bitonic 排序恢复降序；packed key 比较、同分优先级和 finish 不变。
- 正确性：合成同分/quota/无效候选直接对拍 1/1 通过，微基准 indices/counts bit-exact。
- 算子微基准：原实现 0.230492 ms，候选 0.052419 ms，**4.397x**；相对 Iter 79 的 0.091244 ms 降低约 42.5%。该结果仅限 Top-K 算子，需完整回归与真实 Qwen 双口径验收。
- 完整回归与生产身份：selector/cache/GraphRunner 35/35 通过；真实 Qwen eager/Graph 16/16 token 均与 Iter 79 完全一致，且两种 execution 一致。
- 生产 16-token TPOT：eager median/steady 26.325567/26.304608 ms，Graph 14.142432/14.123424 ms。Graph 相对 Iter 79 steady 15.465488 ms 改善约 8.68%，相对原 baseline 23.588048 ms 为 **1.670x**；仍比 FA2 Graph 12.602576 ms 慢约 12.1%。eager 相对 Iter 79 有噪声级回退，保留后续单独处理 host/launch 开销。
- 判决：保留为当前最佳正确 Graph 实现；下一轮转向剩余 partial Top-K、通用 elementwise/softmax 和 eager host 路径。

## Iter 81：bitonic 归并下恢复低成本小 tile

- 修改：在 Iter 80 的有序归并基础上将 partial tile 从 4096 降回 2048。此前 Iter 77 的瓶颈是 8192-wide `tl.topk` merge；该瓶颈已被 bitonic 归并消除，因此重新利用 2048-wide partial 的较低成本，归并仅顺序处理 5 条有序 run。
- 正确性：合成同分/quota/无效候选直接对拍 1/1 通过，微基准 indices/counts bit-exact。
- 算子微基准：原实现 0.230753 ms，候选 0.038990 ms，**5.918x**；相对 Iter 80 的 0.052419 ms 再降低约 25.6%。需完整回归与真实 Qwen 双口径验收。
- 完整回归与生产身份：selector/cache/GraphRunner 35/35 通过；真实 Qwen eager/Graph 16/16 token 均与 Iter 80 完全一致，且两种 execution 一致。
- 生产 16-token TPOT：eager median/steady 25.845119/25.793744 ms，Graph 12.505600/12.462192 ms。Graph 相对 Iter 80 steady 14.123424 ms 改善约 11.77%，相对原 baseline 23.588048 ms 为 **1.893x**；首次在该 16-token 迭代口径略快于 FA2 Graph steady 12.602576 ms（约 1.13%），但不是最终 ABBA/128-token 验收。
- 判决：保留为当前最佳正确实现。Graph 已出现超过 FA2 的早期信号；eager 25.793744 ms 仍慢于 FA2 23.725280 ms，继续优化 eager host/launch 与剩余设备热点。

## Iter 82：1024-token 最小 tile 回退

- 修改：将 partial tile 从 2048 降为最小合法值 1024；局部 Top-K 退化为整 tile 排序，同时有序归并需顺序处理 9 条 run。
- 正确性：合成同分/quota/无效候选直接对拍 1/1 通过，微基准 indices/counts bit-exact。
- 算子微基准：原实现 0.230508 ms，候选 0.042952 ms，**5.367x**；比 Iter 81 的 0.038990 ms 慢约 10.2%。
- 判决：直接恢复 `tile=2048`，不进入完整生产验收；更多 run 的有序归并成本超过小 partial 的收益。

## Iter 83：fine-score 与 partial Top-K 融合回退

- 修改：新增 `_nfa_fine_topk_partial_kernel`，在 2048-token tile 内直接完成 paged K 标量点积、BF16 舍入、signed packed-key 与局部 Top-K，移除 `[m,hkv,r]` logits 物化和一次 launch；merge/finish 保持 Iter 81 实现。
- 正确性：真实生产 shape `[1,8,8320]`、K=1024 的独立 paged-K 微基准中，融合前后最终 indices/counts bit-exact。
- 算子微基准：原 fine + partial + merge 为 0.078129 ms，融合 + merge 为 0.122961 ms，候选慢约 57.4%（speedup 0.635x）。融合将 fine 的 65 路 tile 并行降为 5 路，并扩大单 CTA 点积/寄存器压力，节省的 launch 与中间流量不足以补偿。
- 判决：不进入完整生产验收；恢复 Iter 81 的 fine/partial 分离路径。后续融合需保持细粒度并行或选择更小生产者 tile，不能沿用本次大 tile 直接融合。

## Iter 84：五路配对归并的 static_range 编译失败

- 修改：针对生产 `NTILES=5`，尝试同时归并 run (0,1) 与 (2,3)，再顺序归并配对结果和 run 4，将四轮归并降为三轮；其余 run 数保留 Iter 81 路径。
- 结果：Triton 编译失败，`first_group` 经 constexpr 分支赋值后被视为 `int32[]`，不能作为 `tl.static_range(first_group, NTILES)` 的起点；尚未进入对拍或计时。
- 判决：记录为失败迭代。下一轮将 `NTILES=5` 的最后一轮直接展开，通用路径保留独立 `static_range(1, NTILES)`，消除非常量起点。

## Iter 85：五路配对归并的 constexpr 修复（GPU 占用，未验证）

- 修改：`NTILES=5` 特化中直接展开 run 4 的最后归并；通用路径改用独立的 `tl.static_range(1, NTILES)`，删除 `first_group` 非常量起点。packed key、归并比较与同分规则不变。
- GPU hygiene：修改后复查本机 GPU0 used=77294 MiB，compute PID 2754683 used=77284 MiB，不满足 used<500 MiB 且无他方进程的起跑条件；未抢占、未启动微基准。
- 正确性与算子微基准：因 GPU0 被其他会话占用，本轮尚未完成编译、bit-exact 对拍和延迟测量，不能宣称改善。
- 判决：仅提交 constexpr 结构修复与阻塞记录，等待干净 GPU0 后再验证；当前最佳仍为 Iter 81 的 0.038990 ms。

## Iter 86：GPU 持续占用，未验证候选按规程回退

- GPU hygiene：再次复查 GPU0 used=75902 MiB，compute PID 2754683 used=75892 MiB；GPU0 始终不满足起跑条件，本轮仍未执行任何 GPU 编译、对拍或微基准。
- 结果：没有候选延迟或 bit-exact 证据，不能确认相对 Iter 81 的改善。
- 回退：按“无改善恢复精确版本”规程，将 `nfa_layer_batch_selector.py` 恢复为 `22dccea4` 的精确内容；恢复后 git blob SHA-1 为 `170347ca84bd7db2ffe42ddfbd64a2acfa4c9a5e`，与该提交一致。
- 判决：最终源码回到当前最佳 Iter 81（参考候选 0.038990 ms）；Iter 85 的 constexpr 修复仅保留在历史提交中，未作为已验证优化合入最终工作树。

## Iter 87：pc40 复验五路配对归并，bit-exact 失败

- 验证：在 pc40 GPU0 的隔离 worktree 中运行真实生产 shape `[1,8,8320]`、K=1024 对拍；起跑前 used<500 MiB，并有 30 秒旁路监控。
- 正确性：Iter 85 可编译运行，但最终 indices 有 7949/8192（97.0%）元素与原单 CTA 参考不一致，最大绝对差 32697；脚本在断言处停止，未执行延迟测量。
- 判决：配对归并改变了选择结果，不能进入生产或真实模型验收。主线继续保持 Iter 86 恢复后的 Iter 81 精确源码；后续若重试必须逐轮检查五路有序 run 的覆盖、反转和最后一轮索引映射。

## Iter 88：fine-score 直接生成 packed Top-K key

- 修改：16-bit token key 几何下，fine-score kernel 直接写 packed int32 key；partial Top-K 不再重读候选 ID、范围、quota 或重做 BF16 ordered-key 位变换。64K/128K 保留原 BF16 logits 路径。
- 验证：pc40 GPU0 隔离 worktree，真实生产 shape `[1,8,8320]`、K=1024；起跑前 used=1 MiB 且无 compute apps。最终 indices 与 counts 全部 bit-exact。
- 微基准：原单 CTA reference 0.230609 ms，候选 partial+merge 0.028474 ms，reference/candidate=8.098947×；相对 Iter 81 的 0.038990 ms 降低 26.97%（约 1.369×）。
- 边界：该微基准不包含 fine-score 新增打包工作，不能据此宣称生产 TPOT 同幅改善；下一步必须跑完整 NFA eager/Graph token identity 与 TPOT。
- 完整生产 128-token：eager steady TPOT 24.480000 ms，相对 Iter 81 独占基线 24.438424 ms 慢 0.17%（噪声量级）；Graph steady TPOT 12.361072 ms，相对 Iter 81 的 12.443976 ms 降低 0.67%（1.0067×）。FA2/NFA 当前为 eager 0.9383×、Graph 1.0190×，均未达到 2×。
- 生产正确性：微基准逐位通过；iter88 eager/Graph 的 128/128 token 完全一致，且分别与 iter81 参考序列完全一致，无首个分叉点。GPU 旁路日志仅见本轮两个生产 PID。
- Graph 同卡 ABBA 复验：iter81 A/B=12.413488/12.432272 ms，iter88 A/B=12.352848/12.346640 ms；中位数 12.422880→12.349744 ms，加速 1.005922×（降低 0.5888%），四个 arm 的 128-token 序列全部一致，确认小幅收益可重复。

## Iter 89：Q/K RoPE 单 launch，首次逐位对拍失败

- 修改：仅在 Qwen3-8B B1、seq_len=1 decode 用一次 Triton launch 同时处理 Q/K；两次乘法显式保留 BF16 中间舍入，最终加法写回 BF16。prefill、B>1 和非目标几何回退 HF；输出由各 attention 模块持久持有以覆盖 Graph 生命周期。
- 静态检查：三个变更文件通过 `py_compile` 与 `git diff --check`；agent 实现阶段未运行 GPU。
- 首次 GPU 验收：pc40 GPU0，起跑前 used=1 MiB 且无 compute apps，运行真实目标几何 Q=`[1,32,1,128]`、K=`[1,8,1,128]` 的 HF `apply_rotary_pos_emb` 位型对拍。
- 结果：第 1 组即失败，Q 有 935 个 BF16 元素不同；脚本在首次断言处退出，尚未得到 K 对拍和 `do_bench` 延迟。该臂不满足正确性硬门，不得进入完整生产 TPOT，也不得宣称加速。
- 原始证据：`iter89_rope_micro.log`、`iter89_rope_micro_gpu0_hygiene.log`。下一步仅定位该舍入/符号语义差异，修复后作为新一轮独立对拍。

## Iter 90：RoPE 禁用 FP fusion，恢复逐位一致

- 根因探针：Iter 89 候选输出与“`q*cos` 保持 FP32、`rotate_half(q)*sin` 先舍入 BF16、最后相加”的 PyTorch 组合逐元素完全一致（0/4096 不同），证明 LLVM 将第一次乘法与最终加法收缩为 FMA，消除了第一次 BF16 中间舍入；不是索引或 `rotate_half` 符号错误。
- 修改：仅在该 Q/K RoPE kernel launch 增加 `enable_fp_fusion=False`，不改变公式、选中集合或其他模型路径。
- 正确性：pc40 GPU0 干净起跑，16 组随机输入覆盖 scale `1/2^-8/2^8/2^-16`，Q/K BF16 位型全部一致。
- 微基准：完整 Python wrapper 的 HF `apply_rotary_pos_emb` 中位数 0.037728 ms，Triton Q/K 单 launch 0.005600 ms，HF/Triton=6.7371×。该数字只含 RoPE 算子，不能外推为完整 decode TPOT。
- 原始证据：`iter90_rope_micro.log`、`iter90_rope_micro_gpu0_hygiene.log`。下一步跑完整生产 eager/Graph 128-token identity 与 TPOT，仍不宣称达到 2×。
- 完整生产 128-token：eager steady TPOT 23.755264 ms，相对 Iter 88 的 24.480000 ms 加速 1.030508×；Graph steady TPOT 11.758976 ms，相对 Iter 88 的 12.361072 ms 加速 1.051203×。当前 FA2/NFA 为 eager 0.966878×、Graph 1.071157×，eager 仍略慢于 FA2，双口径均未达到 2×。
- 生产正确性：iter90 eager 与 iter88 eager、iter90 Graph 与 iter88 Graph、iter90 eager 与 Graph 的 128/128 token 全部一致。输入 SHA256 保持 `e851c310…99c2db`，selector 配置不变。
- 测量卫生：pc40 GPU0 起跑前和结束后均 used=1 MiB 且无 compute apps；30 秒旁路日志未观察到其他进程。本轮为两次独立执行的单次生产结果，后续仍需同卡 ABBA 复验，原始证据为 `ruler32k_iter90_128_{eager,graph}.json`、`iter90_production_{eager,graph}.log`、`iter90_production_gpu0_hygiene.log`。

## Iter 91：Qwen3 decode RMSNorm 三 kernel 路径

- 修改：仅在 BF16、contiguous、seq_len=1 decode 融合 HF RMSNorm 的 `to(fp32)+pow(2)` 与归一化/舍入/weight 外壳；中间仍调用原 `torch.mean`，保持归约 kernel 和顺序。Qwen3 attention patch 后注册全部 RMSNorm，Qwen3-8B 数量硬校验为 `36*4+1=145`；prefill、其它 dtype/布局回退 HF。
- 正确性：pc40 GPU0 独立执行，Qwen3-8B/32B/30B-A3B 共 18 组 square/mean/out 逐级 BF16/FP32 位型对拍全部通过；gate 4/4、真实 Qwen3RMSNorm 实例树 patch/unregister 7/7、ops 与 patched module 的 CUDA Graph capture/replay 2/2 通过。
- kernel 计数：HF eager 一次 forward 为 8 个 CUDA kernel，候选为 `_square_fp32_kernel + torch.mean + _norm_scale_kernel` 共 3 个；单次数值执行设备时间约 14.1 μs→5.7 μs，但这不含 Python/分配/launch 外壳，不能据此宣称 wrapper 或生产加速。
- eager wrapper 微基准：CUDA event 中位数显示九种真实 shape 全部回退，HF/候选仅 0.89–0.90×。Qwen3-8B hidden norm 为 0.0609→0.0683 ms，q_norm 为 0.0614→0.0683 ms，k_norm 为 0.0613→0.0681 ms；减少设备 kernel 被外壳开销抵消。
- 测量卫生与边界：起跑前 GPU0 used=1 MiB 且无 compute apps；5 秒旁路日志仅见本轮 1→4 MiB 显存变化，无异常共居跳变。原始证据为 `iter91_rmsnorm_micro.log`、`iter91_rmsnorm_micro_gpu0_hygiene.log`。当前只证明数值与 Graph 安全性，尚未证明生产收益；下一步必须跑完整 128-token eager/Graph token identity 与 TPOT，再决定保留或恢复 Iter 90。
- 完整生产 128-token：eager steady TPOT 27.129328 ms，相对 Iter 90 的 23.755264 ms 明显回退（Iter90/Iter91=0.875630×，耗时增加 14.16%）；Graph steady TPOT 11.151200 ms，相对 Iter 90 的 11.758976 ms 加速 1.054503×（耗时降低 5.17%）。当前 FA2/NFA 为 eager 0.846628×、Graph 1.129539×，双口径均未达到 2×。
- 生产正确性：iter91 eager 与 iter90 eager、iter91 Graph 与 iter90 Graph、iter91 eager 与 Graph 的 128/128 token 全部一致，输入身份与 selector 配置不变。说明三 kernel 路径数值可保留，但 eager 外壳开销不可接受；后续若保留必须按执行口径隔离，不能用 Graph 收益掩盖 eager 回退。
- 原始证据：`ruler32k_iter91_128_{eager,graph}.json`、`iter91_production_{eager,graph}.log`、`iter91_production_gpu0_hygiene.log`。pc40 GPU0 起跑前 used=1 MiB 且无 compute apps；旁路样本未记录到其他 PID。本轮仍是单次生产结果，Graph 收益需同卡 ABBA 复验。
