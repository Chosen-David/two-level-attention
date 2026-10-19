# TL-DECODE-PADDING-Q-20261019 独立复核

2026-10-19；独立执行者 `twolevel_padding_review`。仅审查、独立 CPU 验证和本报告写入；未修改生产实现、guide、TASK 或远端。沿用代码实现优化技能的正确性/证据边界，不因 CPU 通过宣称 GPU 或模型质量通过。

## 版本与结论

- 基线：`bc968288c7e69ec3d93b6d2acf546a3dcd26f962`。
- 审核候选文件 SHA256：`qwen3_attn_patch.py` = `f030593c2a94390d7ccfcf1e5bcbf89325e9830ace9025da442dc5077713fca8`；`llama3_attn_patch.py` = `3fa398361256bdd8e935c179d128c601ca3f71740e50886005e5b525d5931c06`。
- 结论：**未发现阻止这个有界、非 kernel 包装正确性修复发布的问题**。以主控相邻回归、fresh fetch 和远端回读完成为条件，可接受。不是性能优化验收，不为任何未执行的 GPU、真实 selector、全模型质量测试背书。

## 代码审查

原 Q 长度为 1、缓存后 K/V 长度为 S，`unpad_tensor(Q, [1,S] mask)` 在 S>1 时维度不匹配。新增守卫确认 `[1,S]` bool 且当前最后位置有效，因此不裁 Q、只裁 K/V 是正确的单 query 包装契约。压紧顺序保留原始有效 K/V 的顺序；RoPE 已在压紧前应用，不应重新旋转或把 Q 再按 mask 裁剪。有效 K 数构造的累计长度与 selector 的紧凑 query 位置对应。

形状验证采用短路 `or`，`[1,0]` 等不匹配长度不会访问 `mask[0,-1]`。`None` 分支、prefill、预算及 attention kernel 未改。Qwen 的已接受静态 4D mask 旁路仍发生在该守卫之前。

明确保留边界：

1. 失败发生在 HF cache update 之后；错误请求不能假设 cache 自动回滚。本次不是异常重试事务修复。
2. `bool(mask[0,-1])` 在 CUDA mask 上增加主机同步。本次是恢复输入契约，不宣称零同步、CUDA Graph 兼容或提速。
3. 验证中的 selector 是全选或固定稀疏选集 double；不验证真实 indexer `_inc` 生命周期、near/far 筛选质量。连续步骤保持已存在 padding 不变，仅追加有效 token；不声称支持任意替换先前有效位。
4. 当前短测试即使 `TLI_E127_FAST=1`，S<16K 仍会回退 dense。不能把环境变量开启当作 torch gather 已执行的证据；下述独立长序列专门检查了实际派发。

## 独立验证方法

独立脚本 `/tmp/twolevel_padding_review_0853.py`，没有调用主控测试类或 scenario。AST 加载实际 forward/helper 函数以绕开无关 CUDA indexer import；使用真实 HF Attention、DynamicCache、生产 `eager_decoding_attn`，HF 原生 SDPA 为对照。固定随机种子 8532026，单 CPU 线程。

短序列：两家族 × fp32/bf16/fp16 × KV heads 1/2/4（Q heads=4）× prefix 1/7/129 × left/holes/仅当前有效 padding × FAST 0/1，共 **324 配置，每配置连续 3 步，972 步**。用真实 forward 做 prefill；decode 的 hidden states 末维 stride=2，mask token 维 stride=2。使用非恒等 cos/sin，比较输出及完整 cache K/V；selector 边界逐次检查原样 Q 形状、K 精确压紧结果、cu 与 query 位置。

长序列：两家族 × 三 dtype × KV heads 1/2/4，共 **18 配置，每配置连续 3 步，54 步**。用固定随机 K/V 初始化真实 DynamicCache（prefix=20000，避免不必要的平方 prefill），每 8 个前缀屏蔽 1 个，紧凑长度超过 16K。每 KV head 选择三个位置，原始位置对照由独立映射生成 HF SDPA mask。对生产 dense 子函数设置计数 spy，所有长场景 dense 调用为 0，并消费 deferred error flag；这是 CPU torch gather 实际执行证据，不是 GPU/Triton 证据。

事先固定容忍度 `(atol,rtol)`：fp32 `(1e-6,1e-5)`，bf16 `(.01,.02)`，fp16 `(.002,.005)`；不因结果放宽。额外缓存和压紧 K 检查容忍度均为 0。

## 执行结果与短日志

命令：

```text
python /tmp/twolevel_padding_review_0853.py > /tmp/twolevel_padding_review_0853.json
python -O /tmp/twolevel_padding_review_0853.py > /tmp/twolevel_padding_review_0853_O.json
```

两次退出码均为 0，各 **342 配置、1026 decode 步**；结果 JSON 完全相同。普通模式还执行 Python assert 检查；`-O` 模式关闭这些辅助 assert，但 `torch.testing.assert_close` 数值检查仍执行。

```text
torch=2.8.0+cpu; CUDA=False; Transformers=4.56.2
cases=342; steps=1026; long_cases=18
short dense_calls/config=3; long dense_calls/config=0
fp32 max_abs_error=2.384185791015625e-07
fp16 max_abs_error=0.00048828125
bf16 max_abs_error=0.00390625
normal_O_identical=True
```

输入随机张量流 SHA256（prefix/token/长场景 K/V 转 fp32 后字节；模型随机权重由固定种子及配置确定，不包含在该摘要）：`c42da8f1eac05c0cba77e84d00456838505842a6bb98417684ec17ae1982657a`。

脚本 SHA256：`b2127ff1f7d264962e9bf39835d3126439ce49885ceb4b6f022d1d49d9150300`。
普通和 `-O` 原始 JSON SHA256 均为 `2e25ff9643f3420fb120a43bb93450bcff8e6e691822eec58e2f6a967934b5f7`。

主控追加授权后已将脚本保存为同目录 `verify_independent.py`，仅将 ROOT 改为根据本文件位置推导 repo 根目录，没有修改验证逻辑。可复现命令：`python agent_doc/results/decode-padding-q-20261019/verify_independent.py`（另加 `-O` 可复跑优化模式）。便携版脚本 SHA256：`d3de6cc2e7f34dfe5789c3b512a3ac8085ca34c114240d10061e646f404516bb`；同目录 `independent-summary.json` 保存短摘要。

未运行计时/显存测量，也未给出速度、内存或质量收益。
