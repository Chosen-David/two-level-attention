# Hybrid cache 验证记录（2026-10-26）

基线 `900f79b4c714164e5e0b36a62c4e330bc739a346`；候选为本记录所在 PR 的源码。
环境：Python 3.12、PyTorch 2.8.0+cpu、Transformers 4.56.2、pytest 9.1.1。
项目正式环境要求 Python 3.11；本次 CPU 验证不代替该环境及 CUDA 验收。

## 实际执行

原版源码回放最初新增的 9 个 CPU 用例：全部失败，其中直接复现了全量 static 分配和第二次 refill 向 17 槽 ring 写入 27-token prompt 的错误。
原版 `test_qwen_static_mask.py` 另有 4 个失败，原因是 AST 测试入口漏注入 `batch_decode_ctx`。
相邻 padding AST 测试也遗漏 batch context 和 Llama 从 Qwen 导入的共享 helper；修复了测试入口，没有删除断言或放宽容差。

最终相关套件：**66 passed, 1 skipped, 178 subtests passed**。

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
