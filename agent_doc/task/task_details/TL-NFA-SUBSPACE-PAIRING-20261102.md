# TL-NFA-SUBSPACE-PAIRING-20261102

状态：修复候选，host 验证完成；原生 CUDA、Graph、质量、性能待验。来源：用户每轮 main 实质代码/性能审计。单 Agent 串行，不修改 guide/论文。

## 基线与机制

基于 main ae69e995acd7e9423d4512f8277aa789a5b310a5；
selector blob 1706d571eb750709d838301bbd9a46201a0678f0。
Qwen/GLM 层内在 write_decode 后将当前 post-RoPE Q 传入 select_topk_for_layer_batched；每层完整请求 batch 选择，随后 grouped paged attention。Graph 捕获相同生产路径，不能依赖历史逐行结论。

1. tail32 + l2_subspace=full：准备 Q[...,128]，细筛却仍传 L1 dims，kernel 读 K[48:64,112:128] 配 Q[0:32]。不是全维点积，也不是正确 tail 点积。确定性反例 Q[0]=1，K0[0]=2，K1[48]=3：应得 [2,0]，基线得 [0,3]，winner 反转。
2. enable_subspace=False（含 subspace=full）：_subspace_indices 仍返回 tail 坐标，粗筛将全维 Q reshape 成 tail 宽度，压缩比4下失败。关闭命名 highfreq/random 子空间时，L2 same 的 QAT Q 坐标也必须与细筛 K 一致。
3. 全维细筛 D=192 会把 D_BLOCK 向上补到256；两遍 Q load 原来没有 dim_mask，末行可能越界，其他行可能跨头读取。新增同 K 一致的 mask/other=0。host 只核对地址有效性，未声称已经在 CUDA 观测到越界。

## 改动与成本

只改 selector 的坐标路由和两处 Q load mask：
- L1 enabled 使用已有子空间，否则 full coordinates；
- L2 full 使用 full coordinates；enabled+same 复用 L1；disabled+same 保留原有 QAT/tail 语义；
- 同一阶段 Q/K 使用同一有序坐标；坐标对象相同时复用本次 Q workspace。没有跨步 Q 缓存。
- 不改变缩放、BF16 舍入、GQA softmax 后聚合、排名/预算/seed/缓存策略，不新建 GPU kernel。
- 复用现有 paged fine/coarse kernel 和普通 arange/视图，优于引入新投影框架。默认 tail/same 仍一份 Q、相同坐标。
- full L2 正确执行后应读取全部 D 而不是错误的 R；工作量增加是恢复语义，不能报告成退化或加速收益。full/full 路由只准备一份 Q，但基线此配置会失败，不构成有效性能对比。无 GPU 时间、流量、显存或任务质量实测。

## 验证证据

tests/test_nfa_subspace_pairing_host.py 执行实际 caller 中的维度与 Q 分支，并读取实际细筛调用实参；NumPy facade，不是 Torch/Triton 执行。
- 修前6个方法：39 failures +75 errors（含 subtests），错配分数与 reshape 均复现；
- 修后普通/-O 各6个方法通过，矩阵144组合：D128/192/256，B1/3，HQ1/4，enabled/disabled，same/full，前段/尾段/隔维坐标；
- 独立修前/修后 default tail/same 配对27组：D128/192/256，B1/3/8，HQ1/4/16，随机 FP32 Q；坐标/Q工作区值及准备调用列表全部相等。仅主机 dispatch 相等；
- py_compile通过。无 Torch/Triton/CUDA；两个 native unittest 方法均 skipped。

tests/test_nfa_subspace_pairing_cuda.py 提供：
- 实际 prepare-Q + paged fine kernel 对独立 Torch 坐标参考，非连续 request IDs、layer offset、不等候选数、GQA，D128/192/256；BF16 概率门限预先固定 atol=2e-3, rtol=1e-2，另核 winner，不以此代替质量；
- 真实 selector 五种配置 × 四种冻结 gate，不等长/非连续 IDs，eager 与 Graph replay 在更换 Q 后比较 counts/有效 support；
- D192 是 fine kernel 单测，并不宣称整个 MLA/full coarse 路径已支持192；其他生命周期/物理页复用/模型质量仍需相邻门禁。

## 远端复现（仓库根目录）

```bash
python tests/test_nfa_subspace_pairing_host.py
python -O tests/test_nfa_subspace_pairing_host.py
git show ae69e995acd7e9423d4512f8277aa789a5b310a5:sparse_attn/indexer/nfa_layer_batch_selector.py > /tmp/nfa_selector_before.py
NFA_SELECTOR_SOURCE=/tmp/nfa_selector_before.py python tests/test_nfa_subspace_pairing_host.py
PYTHONPATH=. python tests/test_nfa_subspace_pairing_cuda.py
PYTHONPATH=. compute-sanitizer --tool memcheck --error-exitcode=1 python tests/test_nfa_subspace_pairing_cuda.py
PYTHONPATH=. python -m pytest -q tests/test_nfa_current_query_selector.py tests/test_graph_ready_paged_region_cache.py tests/test_multi_request_graph_runner.py
```

修前 host 命令预期失败。CUDA 主机必须确认 native 测试实际执行而非 skipped；再验证默认路径与基线严格配对、EOS/请求槽复用/多次 wrap/重新 capture，不放宽既有门槛。

性能只在默认可运行 tail/same 上做 baseline/candidate 同机独占 ABBA；使用同一模型、数据 SHA、样本和生成长度，32/64/128K × B1/2/4，eager/graph。现有入口可用（以真实本地模型和 native RULER 根目录替换变量）：
```bash
PYTHONPATH=. python benchmark/efficiency/benchmark_nfa_end_to_end.py \
  --model-path "$NFA_MODEL_PATH" --data-root "$NFA_RULER_ROOT" \
  --output /tmp/nfa_pairing_candidate_32k_b1_graph.json \
  --context-length 32768 --batch-size 1 --method nfa --execution graph \
  --input-provenance native-ruler --max-new-tokens 128 --warmup-steps 3
```
两版本用不同输出文件，至少3组ABBA；核对输出身份后再比较 TPOT/整请求/峰值。该性能入口复制一个 prompt 组成 batch，只是均长性能负载，不是异质 batch 质量证明；未暴露本修复的全部消融 flags。full L2 需独立正确参考与真实 LongBench/RULER 配对，历史 gate 质量无效。

## 分支/PR 对照与未完成项

本轮刷新64个远端 heads，63个开发分支均与上次一致；复用精确 SHA、时间、独有提交、祖先和实际代码/任务/实验记录，12个 main 祖先为历史快照。PR10 head2449f569 无新 selector；PR30 head0bd69289 的 selector blob3e3acbf7 无当前层入口。PR16 的旧 exact producer Q 复用、PR14 的旧 GQA dtype、PR22 的旧 fine 输出复用均不解决当前坐标路由。所有30个既有PR仍open，review及仓库issue/inline comments为空；无未合并依赖。

opt/l2-fine-topk 56d503a7 是 main 祖先；opt/ako-pool-score a8eb0f08 的独有2提交是旧HybridLayer/ring布局；e175-skip-ring-kv 9b6e792e 的独有3提交是旧实验ring路径。既有 ABI/质量待验结论复用，不移植、不覆盖其分支，不凭 Git 时间认定任务正在运行。

之前 both-skip 选择旁路、增量页摘要、prefill广播复制等候选保留；本修复不替代那些性能实验，不替代 head0支持集、seed、回执与模型端到端验收。

