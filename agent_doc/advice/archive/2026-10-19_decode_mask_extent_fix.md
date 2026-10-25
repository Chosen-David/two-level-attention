# Decode 选择 mask 覆盖长度修复

问题 ID：TL-DECODE-MASK-EXTENT-20261019。基线 main `5549339870004cb9ffb5dba6f20a88c89ac48547`；fresh clone/fetch 核验。实验分支 `e138-ablation` 当前为 `2ed6a1feb78a7b5bcf3cac922b4510c3968714a5`，只读对照，没有合并。

## 触发与根因

Qwen3/Llama/DeepSeek patch → indexer.prepare_mask → eager_decoding_attn 的公共消费端要求 mask 最后一维按 block_size 展开后覆盖 K/V token 轴。原 `_validate_gqa_shape` 只检查前置维度、head 与 K/V 布局，没有检查 mask 的 token 覆盖长度。

真实 CPU 最小例：Q/K 为0、V token值为[0,1,2]、mask仅一个True槽、block_size=1。原 dense 消费端将 `[1]` mask 广播为全3列并返回1，而不是拒绝不完整的选择。gather 遇长度不等会回退 dense，同样触发；其他短mask报底层shape错误。不能把单槽解释为通配符，因为该接口每个槽代表一个选择块。

修复位置：`sparse_attn/ops/eager_decoding.py:355–360`。共同入口检查 `mask.shape[-1] * block_size >= k.shape[-3]`；不满足时在派发前 ValueError。允许完整块的末尾padding及额外尾槽，继续由原实现裁剪。仅增加常数次host元数据运算，不读tensor值，不修改selector/kernel、预算、GQA、子空间或chunk2048。

候选比较：拒绝不完整mask采用；自动补False拒绝（将非法输入隐式解释为新语义）；单槽广播拒绝（静默扩大选集）。依据为接口块语义及 [PyTorch 2.8 broadcasting说明](https://docs.pytorch.org/docs/2.8/notes/broadcasting.html)，2026-10-19核对。标准broadcast允许单维1扩展，不能替代选择器的覆盖契约。

## 实际验证

新增 `sparse_attn/ops/tests/test_eager_mask_extent.py`。修前运行4个测试方法：

```text
Ran 4 tests in 2.150s
FAILED (failures=15, errors=9)
AssertionError: ValueError not raised
```

修后执行以下两组，普通与 `python -O` 各一次，均 `OMP_NUM_THREADS=1`：

```sh
python -m unittest discover -s sparse_attn/ops/tests -p 'test_eager_*.py' -v
python -m unittest discover -s tests -p test_decode_padding_query.py -v
```

```text
普通: Ran 19 tests in 3.056s / OK; Ran 6 tests in 1.031s / OK
-O:   Ran 19 tests in 2.167s / OK; Ran 6 tests in 0.987s / OK
```

专项覆盖24项非法入口子例和192项合法选集子例：空/单槽/一般短mask、3/5/65/16385长度、shared/per-q-head、异维V、块宽1/2/64、短尾块和额外padding，拒绝发生在公共派发前。合法QK=0场景按实际选中V的算术平均独立预期，atol=rtol=1e-6。相邻HF decode padding测试输入SHA256仍为 `713468324be3aacd8c6188a34132a8c3a7421a12f20bbbb2df6a1634efae3e85`。

独立新上下文复核记录及自建输入哈希见 `agent_doc/results/decode-mask-extent-20261019/`。修前日志SHA256 `14422f73393548eaffab5d533130e4e60ca7b6c84a98e2dc3b899f1efe947dcc`；修后eager普通 `1f2474e846111462d7f4294008aa32827540a672787f1d6970b223023506183a`，-O `2097509b8783f9e4cbf2da6c5de4d2337d0fd0467818d822778d6b292e00ad31`。以上是测试日志耗时，非性能基准。

环境：Linux6.18.55 x86_64，Intel Xeon Platinum8573C（容器可见9CPU），Python3.12.14，torch2.8.0+cpu，Transformers4.56.2；CUDA不可用，无GPU驱动/架构或新kernel编译数据。技能沿用已安装research-implement-optimize及kernel证据闭环，仅应用CPU可验证边界，不安装额外工具。

独立复核PASS：48合法随机逐头float64参考对照与12非法extent通过；8例spy确认实际CPU gather，覆盖异维V、shared/per-qh、非连续mask和长序列。最大误差2.384185791015625e-07。独立审查没有阻断项，仅指出validator历史docstring清单未列新增extent条件，不影响本次运行契约。

## 影响与未验证边界

这是非法输入被静默广播的正确性修复，不宣称有效模型任务精度或速度提升；峰值显存、kernel数、host同步及TTFT未测。合法输入保持原计算路径。独立复验与CPU gather不能代替GPU融合kernel验收。

历史实验只有在indexer输出宽度不足时受此问题影响：须检查mask.shape[-1]×block_size与实际K长度，并重跑受影响样本。正常完整/补齐mask结果不因本问题自动失效。未修改历史数据；真实selector、GPU、实模质量、固定前缀替换后缀及跨请求增量缓存仍需后续验证。候选内L2、紧凑索引、overlap等性能方向没有本轮实测证据，不合入未验kernel。
