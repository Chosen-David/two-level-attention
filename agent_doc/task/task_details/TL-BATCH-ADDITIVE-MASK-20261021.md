# TL-BATCH-ADDITIVE-MASK-20261021

Task-ID: TL-BATCH-ADDITIVE-MASK-20261021
Date: 2026-10-21
Base: `12ff5269e2e5b180e242a85cf67abcaa1ac41116`

## 问题与真实路径

`benchmark/RULER/pred_ruler_batch.py:146-159` 为 decode 构造 4D 加性
mask：真实列为 `0`，left-pad 列为 dtype 最小值。FASA 的 Qwen3 adapter 在
首层以外的每个单 token decode 都调用
`sparse_attn/baseline_runtime.py:167-198` 交叉校验该 mask 与
`BatchCtx.seq_lens`。

基线 `batch_valid_len_from_mask` 对非 bool mask 统计 `row != 0`，实际得到
padding 数而非真实 token 数。固定两行、KV 长5、真实长度 `[3,4]` 的输入
被反算为 `[2,1]`，抛出：

```text
ValueError: batch vendor sparse decode mask derived lengths [2, 1] != ctx.seq_lens [3, 4]
```

因此 `pred_ruler_batch --method fasa_vendor` 会在首个稀疏 decode 层
fail-closed；这不是选集数值容差问题。bool mask 的 `True=有效` 分支没有该
错误。

## 修复与验证

`sparse_attn/baseline_runtime.py:146-164` 的加性分支改为精确统计零值；负值
padding 不再计入有效长度，非零自定义 bias 仍不能伪装成可见 token，保持
fail-closed。未改 batch 上下文、FASA 选择、预算、chunk2048、sink/SWA、
kernel 或输出协议。

`tests/test_batch_decode_context.py:100-170` 以无 torch 的最小 tensor 协议回归
真实调用 `require_unmasked_decode`，同时覆盖加性 mask 修复和 bool 路径不变。
输入（shape `[2,1,1,5]`、上述 mask、`seq_lens=[3,4]`）规范 JSON 的
SHA-256 为
`e6b81437f90756b7b305c93bdd11b10c30817ac62dc87c97d6e9c1a4621a8099`。

```text
python -m unittest -v tests.test_batch_decode_context
Ran 4 tests in 0.002s -- OK

python -O -m unittest -v tests.test_batch_decode_context
Ran 4 tests in 0.002s -- OK

python sparse_attn/ops/tests/test_batch_explicit_quota_guard.py
python -O sparse_attn/ops/tests/test_batch_explicit_quota_guard.py
Ran 6 tests -- OK  # each mode

python -m compileall -q sparse_attn/baseline_runtime.py \
  benchmark/RULER/pred_ruler_batch.py tests/test_batch_decode_context.py
git diff --check
# both silent / success
```

环境为 Linux x86_64、Python 3.12.14、glibc 2.39；当前环境未安装 torch，
没有可用 CUDA/GPU，GPU 架构、驱动、CUDA、Triton、框架版本、编译器和
kernel 配置均不可登记。相邻 `test_batch_sink_guard.py` 与
`test_batch_topk_block_size.py` 在导入期因缺 torch 未执行，不记作产品失败。
因此未执行真实 Qwen3/FASA、选集/输出对拍、prefill→decode 交接、速度、
显存、kernel 数或 host 同步测量，不作性能或质量提升声明。
历史上该入口若有“成功”产物，需要先核对是否实际派发 FASA batch 路径；本
缺陷通常直接报错，不据此自动判任何历史分数失效。持续任务仍保持开放。
