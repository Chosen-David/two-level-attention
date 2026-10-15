# MLA 不支持调用的 KV 状态保护

## 范围与版本

main 基线：`9f3f1ca7212e248f2012bb5e4743d39d7e56530c`；本轮 fresh fetch 未发现新提交，继续未完成的 MLA 请求/状态边界检查。`e138-ablation` 为 `2ed6a1feb78a7b5bcf3cac922b4510c3968714a5`，未修改或合并。此前 sink 页坐标修复不重复审查。

**TL-E142-REJECT-CACHE-20261015 / P2 / confirmed → fixed，CPU rechecked**：实际 attention forward 对不支持的 continuation 执行 `cache.update` 后才抛出错误。若调用者捕获异常并复用 cache，失败调用的 token 留在后续上下文。空输入也会把空 cache 初始化成空 tensor 后才报错。

## 根因、触发和修复

入口：`sparse_attn/patches/patch.py` 注册 MLA forward → `deepseek_v2_attn_forward` → `past_key_value.update` → post-update prefill/decode 路由及硬守卫。

基线 `sparse_attn/patches/deepseek_v2_attn_patch.py:133–136` 写入 cache；`:182–187` 才拒绝已有 prefix 下的 `q_len != 1` 或 `batch != 1`。例如 prefix 长3，尝试续接2 token，收到“不支持 chunked prefill”异常，但 cache 长度已变5。用原 cache 重试单 token，将读到本应被拒绝的2 token。

修后 `:74–77` 在任何投影或 cache 写入前拒绝空输入；`:118–128` 复用已有 `get_usable_length` 查询，在 cache 写入前拒绝非空可用 prefix 下不支持的 batch/query 长度。保留原 post-update 防御检查，不假设任意第三方 cache 返回值都遵守其长度声明。

合法新请求（包括 batch2 的完整 prefill、单 token 首步）及 batch1 单 token decode 的计算与路由不变；不新增 cache 查询、`.item()`、设备归约、host/device 同步或 tensor 运算。新增的是 Python 元数据条件判断，未经 GPU 测速，不宣称零耗时或加速。

这不是完整异常事务：不能保证 OOM、第三方 cache 自身异常或长度契约不一致、后续 kernel 异常时自动回滚，也不保证任意 bounded/static cache 支持。范围是已声明不支持的输入形态在已知长度合同下拒绝前不写入。

## 复现与验证

新增3项测试于 `sparse_attn/ops/tests/test_deepseek_request_lifecycle.py`：

1. 已有 prefix 后 batch1/q_len2、batch2/q_len1 的拒绝，不改变 K/V 对象、长度和 indexer 计数。
2. 空输入拒绝不初始化 cache。
3. 拒绝 chunk 后重试，结果与从未收到该 chunk 的独立 cache 逐位相同。

测试调用 AST 提取的实际 forward；模型投影、cache、indexer 为小规模 doubles，RoPE 为 identity。因此验证真实 host 路由和 CPU attention，不是实际模型或生产 GPU 验收。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s sparse_attn/ops/tests -p test_deepseek_request_lifecycle.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s sparse_attn/ops/tests
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -O -m unittest discover -s sparse_attn/ops/tests
```

本机执行额外指定 `PYTHONPATH=/tmp/twolevel-cpu-deps:.`。原始短日志：

```text
baseline: Ran 12 tests; FAILED (failures=4)
retry: Mismatched elements: 8 / 8 (100.0%)
Greatest absolute difference: 0.042731836438179016
fixed:    Ran 37 tests in 1.731s; OK
fixed -O: Ran 37 tests in 1.550s; OK
legal_route_exact_parity=18 cache_call_counts_unchanged=True
```

合法路径额外对照：从 `git show HEAD:sparse_attn/patches/deepseek_v2_attn_patch.py` 提取基线 forward，与当前 forward 使用相同投影/独立 cache 比较。float32/bfloat16、batch1/2、长度1/3/5、非连续 hidden 共12个 prefill，加 batch1 各自6个 decode，共18例 output/可返回 weights 逐位相同；计数 cache 的 `get_usable_length` 和 `update` 次数一致。该对照为本轮 CPU 检查，不外推 GPU 数值或时延。

重试用例 seed142（先按测试 setUp 初始化投影，再生成 prefix/rejected/retry），H=2、Q/K D192、V D128、hidden D8，长度3/2/1。三段 float32 输入连续字节拼接 SHA256：`fa8193370fd58a63cea4794722fe264b6054b06019695e3a88890ec4262616c7`。

环境：Python3.12.14，x86_64，PyTorch2.8.0+cpu，`cuda=False`。GPU/驱动/内核编译器配置、实际 DeepSeek 权重与完整请求性能未验证。测试耗时不是 benchmark。

## 历史数据及未完成项

尚无证据表明既有 E142a 实验捕获了这种异常后复用污染 cache；不宣称历史分数已受影响。若运行日志出现上述拒绝且复用了相同 cache，应从干净请求状态重跑。未改任何历史实验结果。

此前确认的 **TL-E142-DECODE-MASK-20261015**（decode 忽略标准4D mask 内 padding/bias）仍未修复。本轮不是 padding 支持，也未增加逐层 mask 内容同步检查。请求级契约仍需覆盖真实模型入口和直接 forward，不能仅在 benchmark 中检查后宣称生产安全。GPU compact-block/overlap 候选继续等待真实设备验证；未修改 kernel、selector、预算、chunk2048、guide。

代码实现与优化技能本轮限定为 CPU 可验证的状态修复，并保留合法路径对照与未验证边界。
