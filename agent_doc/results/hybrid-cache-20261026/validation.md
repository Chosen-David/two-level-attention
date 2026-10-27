# Hybrid cache 验证记录（2026-10-26）

基线 `900f79b4c714164e5e0b36a62c4e330bc739a346`；候选为本记录所在 PR 的源码。
环境：Python 3.12、PyTorch 2.8.0+cpu、Transformers 4.56.2、pytest 9.1.1。
项目正式环境要求 Python 3.11；本次 CPU 验证不代替该环境及 CUDA 验收。

## 实际执行

原版源码回放最初新增的 9 个 CPU 用例：全部失败，其中直接复现了全量 static 分配和第二次 refill 向 17 槽 ring 写入 27-token prompt 的错误。
原版 `test_qwen_static_mask.py` 另有 4 个失败，原因是 AST 测试入口漏注入 `batch_decode_ctx`。
相邻 padding AST 测试也遗漏 batch context 和 Llama 从 Qwen 导入的共享 helper；修复了测试入口，没有删除断言或放宽容差。

B=1 初版相关套件：**66 passed, 1 skipped, 178 subtests passed**。batch 升级后的结果见下节。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  tests/test_far_skip_hybrid_cache.py tests/test_qwen_static_mask.py \
  tests/test_e139_graph_contract.py tests/test_batch_decode_context.py \
  tests/test_decode_padding_query.py tests/test_baseline_integration.py \
  tests/test_e152_request_timing.py -q --tb=short
```

其中新测试使用真实 HF Qwen3、真实 TLIIndexer、真实 cache/consumer；模型为随机小模型，不下载权重。
验证 sink/near/SWA KV 内容、单次写入、物理槽位容量、原 storage 释放、线性层恢复、不同 prompt 长度及多次 ring wrap。
CPU eager/static logits 对拍容差为 rtol=2e-5、atol=2e-6；主 KV 保留值按逐位相等比较。
标准 near4096/swa128/sink4/page64 的容量公式在长度 1..20000 检查，32831 的反例证明 4228 槽不足、4291 槽足够。

以下已尝试但被环境阻塞，**不计通过**：

- `python exp/bench/test_e127_gates.py`：`Torch not compiled with CUDA enabled`。
- `python exp/bench/test_tli_batch_gates.py`：`需要 GPU（Triton/eager 消费端）`。
- 新增整模型 CUDA capture/replay、第二请求重新 capture 测试：无 CUDA，skip。

## GPU 验收

先在项目正式环境运行上述测试和两项硬门。新 CUDA 用例无需模型权重。
随后在 E152v4 现行冠军参数上显式加 `--tli_far_skip_backend compact --tli_far_skip_cache evict` 和同一套冻结 far gate 配置。
用 `--graph-mode eager` 对拍 `--graph-mode graph`，覆盖至少两个请求、页边界和多次 ring 环绕；生产长度另测 32K/64K/128K。
该 staged 对拍验证 capture，不代替 legacy/FullKV 的任务质量评测。

新增 `kv_storage_bytes` 只计主 K/V；`decode_cache_allocated_gib` 是释放原 DynamicCache 后、capture 前的整个进程 allocated 显存，包含权重和其他分配；原 `peak_mem_delta_gib` 仍计全过程峰值相对请求起点的增量。
三者不可互换，也不可把主 KV 减少比例当作总显存降幅。另记录 fill、每请求 capture 和稳态 TPOT；dense prefill 的完整 KV 需求仍存在。

## PR #9 batch/graph 升级

实现范围与使用方式见 [任务记录](../../task/task_details/TL-HYBRID-CACHE-20261026.md)。
最终关联回归：**85 passed, 7 skipped, 178 subtests passed**（88.47 秒）。7 个跳过项均要求真实 CUDA；`git diff --check` 与修改模块语法编译通过。

新增真实随机 Qwen3 测试包括 BS=1/2/8/32、长度 1/7/23/31、逐请求动态 gate 冻结、独立完整历史参考、多次环绕、全层 patch、EOS 后 ring/linear/native dense KV 不变、原 prefill storage 可释放、连续不同形状 batch，以及 inference_mode 几何缓存失效。
常规预算 batch logits 对拍 rtol=3e-5、atol=3e-6；KV 冻结按逐位相等比较。
CPU 测试关闭 eager oracle 的 GPU-only Triton dispatch，运行实际 torch 计算，没有伪造 selector/attention 输出。
关联 runtime-shape AST harness 原先遗漏 `_graph_state_fn`、KV dtype/device signature 和 full-dispatch keyword，补齐这些测试依赖并保留原断言；该方法生产分支未因测试而更改。

开发中短序列对拍在 length=8 复现旧 graph/eager logits 最大绝对差约 0.03085，定位为 K2 未裁到实际长度；同时修正同页 bypass 重复。L1=128、capacity=17 的真实模型测试覆盖 quota 大于实际页数与尾页越界。
K2=32/near_pages=1 的超额预算会消费零分补位候选；该情形测试预算、无 padding 泄漏以及 batch 与独立固定宽请求一致，不拿未规定的零分 topk tie 顺序冒充 legacy bitexact。

CUDA 测试已加入 BS=2/8/32 × float32/float16，整模型 capture、38 步 replay、EOS、固定地址与第二 batch 重新 capture；本机全部因无 CUDA 跳过。
两项 GPU 硬门本次再次执行，仍分别在 `Torch not compiled with CUDA enabled` / `需要 GPU` 处失败，未获得 GPU 验收证据。

升级后的回归命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  tests/test_hybrid_batch_cache.py tests/test_far_skip_hybrid_cache.py \
  tests/test_qwen_static_mask.py tests/test_e139_graph_contract.py \
  tests/test_batch_decode_context.py tests/test_decode_padding_query.py \
  tests/test_baseline_integration.py tests/test_e152_request_timing.py \
  tests/test_ruler_batch_treatment_snapshot.py tests/test_tli_runtime_shape_contract.py \
  -q --tb=short
```

大 BS 功能支持不等于吞吐验证：当前 attention/index 操作仍逐行，graph 批量记录；没有 GPU latency、显存峰值、真实模型质量的新结论。GPU 上应先跑新 CUDA 测试和硬门，再用相同 RULER 参数对拍 `TLI_CUDA_GRAPH=0/1`，覆盖 BS=1/2/8/32 与 32K/64K/128K。

## GPU 验收补记（2026-10-27，pc31 H20-3e）

本节补齐上文欠账的两项 GPU 硬门与 CUDA 用例。载体：**精确推送物**
（17e56e8 + 当日 in-place 修复 add() 逐层释放源 KV，隔离 worktree），
tar 部署于 pc02 `~/wt_hybrid_gpu`（H20-3e），未触碰宿主仓库与 shadow 包；
双卡空闲窗口执行，无并发污染。同套用例另在 pc31 先行复跑一轮，结果一致。

- hybrid 全套测试（含 CUDA 用例 BS=2/8/32 × float32/float16、整模型 capture、
  38 步 replay、EOS 冻结、固定地址、第二 batch 重新 capture、graph/eager 对拍）：
  **34/34 PASS**。
- E127 优化门禁（含 `tli_layer_skip_mask.json` 完整 079 门控）：**15/15 PASS**。
- tli_batch 门禁：**59/59 PASS**。
- 同日修复：`HybridBatchCacheBuilder.add()` 逐层消费源 DynamicCache（拷入即释放），
  构建期峰值从 `Σ源+Σ目的` 压到逐层 `max(源,目的)`；`test_storage_is_row_specific_
  and_prefill_history_is_released` 加严为 add() 返回即释放断言。

仍未闭合（如实声明）：真实大模型 TPOT / 显存峰值实测；70% skip 比率的 RULER
质量门（三方：dense / 70% hybrid / 全层紧凑）；每批重 capture 成本。闭合前
"稳态 KV 节省"不得引用为峰值显存降幅或质量无损结论。
