# E142 MLA mask 契约：prefill 因果修复与 decode padding 待办

## 版本、范围与状态

基线 main `b6f39b25f7b68d2d24204cdb6d14700f411fe452`；本次 fetch 后无新增 main 代码。实验分支 e138-ablation 仍为 `2ed6a1feb78a7b5bcf3cac922b4510c3968714a5`。本次继续上轮明确未完成的 MLA mask 检查；没有重审已修复的 Graph 问题或合并实验分支。

| 稳定 ID | 严重度/状态 | 触发与影响 |
|---|---|---|
| TL-E142-PREFILL-CAUSAL-20261015 | P1，confirmed → fixed/CPU rechecked | fresh cache、q_len>1、attention_mask=None 时，prefill 读未来 token |
| TL-E142-MASK-LAYOUT-20261015 | P2，confirmed → fixed/CPU rechecked | 2D padding/bool mask 被当 additive bias 广播；错误长度在写入 cache 后才报错 |
| TL-E142-DECODE-MASK-20261015 | P1，confirmed，**未修复** | 标准 4D additive mask 含被屏蔽位置时，decode 仍忽略其值；B=1 也可触发 |

已修复范围是 `sparse_attn/patches/deepseek_v2_attn_patch.py` 与对应 CPU 回归；未修改 protected guide、indexer 数学、kernel、chunk2048、预算和结果数据。

## 真实入口与原因

`sparse_attn/patches/patch.py:106–116` 按 DeepseekV2 类名前缀及 MLA 属性注册，覆盖 eager/FlashAttention2 等 attention 类；替换 forward 不改变外层模型产生 mask 的方式。2026-10-15 核查作者 [DeepSeek-V2-Lite-Chat 固定版本 modeling_deepseek.py](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/935c70aa5c82e55e69834581e72aec45bfa29430/modeling_deepseek.py)：FlashAttention2 外层在无 padding 时传 None，有 padding 时保留 2D mask；另一分支构建 4D causal mask。这里只据源码确认可达，不声称执行了真实模型/GPU。

基线 patch `:138–140` 只在 mask 非 None 时相加，None 时直接 softmax，丢失原后端自带因果语义。基线 `:117–120` 先更新 cache，再到 mask 广播运算；错误形状可能先污染可复用状态。2D 的 0/1 是可见性，不是 additive bias；bool 同样不能直接加到分数。

修复后 `:74–81,115–120` 使用 shape/dtype/device 元数据校验标准 `[B,1,Q,K]` 浮点 mask，在 cache.update 前拒绝非法布局。非标准 head-specific bias 也明确不支持。`None` fresh prefill 在 `:156–163` 补上上三角 -inf；单 token 不分配该临时 mask；临时 mask 在 softmax 前释放 Python 引用。显式 4D additive 路径的数学运算不变。

## 修前失败与修后验证

环境：Python 3.12.14 / x86_64 / torch 2.8.0+cpu；CUDA 不可用，未运行 GPU、驱动/架构检查、Triton 编译、memcheck、真实权重或质量数据集。沿用既有测试工具，不新增依赖或外部 Agent。

最小可运行回归：`sparse_attn/ops/tests/test_deepseek_request_lifecycle.py`。真实 forward AST 执行，模型/cache/indexer 为可控 doubles；identity RoPE 隔离阶段/掩码问题，不验证 RoPE 与生产 kernel。独立参考是显式绝对因果 mask 的 dense 数学；另用固定前缀、替换后缀检验不变性。

```text
初版 3 个新增测试对修前代码：Ran 7 tests ... FAILED (failures=6, errors=1)
None prefill length=3: output maxdiff=0.3192758858
None prefill length=5: output maxdiff=0.2771155238
替换后缀：前两行 16/16 元素变化，maxdiff=9.1263408661
修后最终：Ran 34 tests in 2.850s ... OK
修后 -O：Ran 34 tests in 1.322s ... OK
git diff --check: PASS
```

完整套件含原 29 项与新增 5 项，覆盖长短 1/3/5、batch=1/2、float32/bfloat16、非连续 hidden、None/显式 causal 对拍、固定前缀替换后缀、错误 dtype/维数/长度/device，以及非法 decode mask 拒绝后已有 cache 不变。不是长序列 GPU 覆盖。后缀用例输入 x 后接 changed 的原始字节 SHA256：`4c9928ca99eecccf7946cc54e79a79bd004406b4c0d90590d588f89625248815`；模型 seed=142，Q/K=192、V=128、H=2，hidden width=8。

```bash
PYTHONPATH=/tmp/twolevel-cpu-deps:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s sparse_attn/ops/tests
PYTHONPATH=/tmp/twolevel-cpu-deps:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -O -m unittest discover -s sparse_attn/ops/tests
```

## 未修复项：decode 4D padding 的独立证据

修后 `:187–190` 仍仅有全零 mask 合同，没有检查/消费非零值；文档删除了“padding 已显式拒绝”的错误承诺。下面在全选 indexer double 下把第 0 个 K 屏蔽，真实 forward 输出与正确 masked dense 输出 maxdiff=`0.1200865060`。这不是新的 GPU 性能或 e2e 质量结论。

```python
import torch
from sparse_attn.ops.tests.test_deepseek_request_lifecycle import DeepseekRequestLifecycleTest, CacheDouble
t = DeepseekRequestLifecycleTest(); t.setUp()
c = CacheDouble(); t.call(torch.randn(1, 3, 8), c)
h = torch.randn(1, 1, 8)
m = torch.zeros(1, 1, 1, 4); m[..., 0] = float('-inf')
out, _, _ = t.forward(t.attn, h, attention_mask=m,
    position_ids=torch.tensor([[3]]), past_key_value=c, use_cache=True)
q = t.attn.q_proj(h).view(1, 1, 2, 192).transpose(1, 2)
p = ((q @ c.k.transpose(-2, -1)) * t.attn.softmax_scale + m).softmax(-1)
ref = t.attn.o_proj((p @ c.v).transpose(1, 2).reshape(1, 1, 256))
print(float((out-ref).abs().max().detach()))
```

h、更新后 K/V、m 顺序拼接字节 SHA256：`c17c128adc50821b7c472757653a997c5548cacb3a9c34da1c033d6834070c39`。此例故意以 all-selected oracle 隔离消费者，不能外推真实 selector 的误差大小。

候选取舍：逐层 `mask.any().item()` 拒绝可阻止静默错误，但会给正常 decode 引入设备归约/host 同步；不能声称性能保持。仅在消费端清除 padding 会改变有效选集大小，不能冒充按有效长度完成两级选择。下一步优先在完整请求入口一次校验/拒绝 padding 并绑定请求合同，覆盖所有受支持入口和直接 forward 的安全处理；完整 padding 支持则需要 selector 的有效位置/保护区语义及消费者联合设计。未验证前不把性能敏感的新 GPU/异步错误协议合入 main。

## 成本、历史数据及重测范围

- 本轮是正确性修复，**没有实测加速比、TTFT、显存峰值或 GPU 质量提升**。标准 4D 路径仅增加 Python 元数据检查，不增加 `.item()` 或设备值读取。
- None prefill 新增 `[L,L]` bool 暂存和 masked_fill，空间 O(L²)，16K 时单 mask 理论 256 MiB（不是实测峰值）；这是恢复因果契约的代价，不与旧错误的双向计算做速度收益宣称。原本 eager logits/P 物化的性能浪费仍存在；强基线应为保留 MLA softmax_scale/异维 V 的 causal SDPA/Flash 路径，但没有 GPU 验证不替换后端。
- 当前 E142a verdict 自述 eager 长 prompt；LongBench `pred.py` 为 DeepSeek 走 generate。没有证据说明这些已跑数据触发了 None 路径或 padding，不整体作废 E142a。需要按实际 loader/backend/mask 日志判断；凡实际用 mask=None 的多 token prefill 应重跑质量，4D padding decode 请求同样需重测或排除。
- 后续：真实模型 None/4D 路由对拍、长序列内存、CUDA dtype/架构验证、请求级 padding 合同；既有 compact-block 分支保持默认关闭且不合 main。此次不是“MLA 全面支持 padding”或“prefill 已提速”。
