# Prefill sink 配置修复：TL-PREFILL-SINK-CONFIG-153

作者：GPT；日期：2026-10-14。用户已授权修改独立库并验证后直推 main。

## 范围与基线

- main：`0288ed576351340f28d63ffe97f70fbc177a5b09`；本轮独立干净 worktree。
- 实验对照：`e138-ablation` = `58745c46a6c3ee0847599f0429fb5dfb2a2526cb`。该分支已使用 `sink_tokens`；没有合并其 gather、prefetch 或 group routing 实现。
- 主线入口：`sparse_attn/patches/qwen3_attn_patch.py:81–89` 的 opt-in B=1、无 padding prefill → `sparse_prefill_attn` → `_chunk_rows_attn`。默认 dense 路由不变。
- 已读任务索引、人类 GUIDE 的保护区定义、paper_figs 的 prefill 验收约束及相关 advice。人类指南未改；指南引用的远程 `/home/wangyuanshuo02/sglang/` 文件本机不可用。
- 旧问题 ID 沿用此前审计的 153；这是主线实际修复与新增 tensor 回归，不是重复报告旧发现。

## 缺陷与修复

| ID / 状态 | 触发与根因 | 修复 |
| --- | --- | --- |
| TL-PREFILL-SINK-CONFIG-153 / P2 / fixed，CPU rechecked | 旧 `eager_prefill.py:144–145` 用历史 `sink_blocks × tia_block_size` 推算 sink；`tli_indexer.py:131–151` 已以 `sink_tokens` 为唯一事实源。合法 sink=0/64、page64 配置仍向行 attention 传128，行保护区 OR 额外引入未选 token，改变 softmax 分母与输出。 | `eager_prefill.py:144–147` 直接读取 `sink_tokens`，缺属性时保留128默认值。无 kernel、selector 或数值公式改动。 |

反例使用**真实 TLIIndexer** 的 `tli_sinkswa_only=True`、sink0、SWA64、K2=128，S=273、H/Hkv=2/1、D=128；Q/K 全0，V[t]=t。末行允许集合应仅为 `[209,272]`，输出均值 **240.5**。修前额外并入 `[0,127]`，输出 **122.5**，绝对误差118；修后为240.5。这是选集错误，不是舍入误差。

默认 sink128/page64 行为保持；sink 大于128时 selector 已包含完整 sink，不能据旧传参偏小就断言丢失保护 token。固定外层 chunk2048 不变。组末 query 共享候选仍是现有近似语义，本修复没有将其变成逐 query 两级筛选。

## 实际验证

环境：Intel Xeon Platinum 8573C / x86_64，Linux 6.18.44，Python 3.12.14，PyTorch **2.8.0+cpu**，einops 0.8.2；CPU测试单线程。独立临时依赖目录，不修改项目依赖清单。无 CUDA 设备/驱动验证，无 Triton/JIT/kernel 编译、GPU 内存检查或模型任务实测。

新增 `sparse_attn/ops/tests/test_prefill_protection.py`：

1. 真实 TLIIndexer sinkswa-only 反例（上述240.5/122.5）。
2. 10组 tensor 对拍：sink0/64/128/192及page32，KV共享和per-query-head mask、GQA=2、S=273尾块、非连续Q/K/V。以确定性selector隔离适配层；逐行绝对位置集合及均值参考独立生成，不声称这10组覆盖真实排名/量化。
3. 默认2048 chunk及保护参数传递；此项替换attention回调检查路由。真实2048计算由原套件T2/T4覆盖。

同输入修前/修后短日志：

```text
baseline 0288ed5, numerical tests:
  sink=0 / shared, per-head: FAIL, max abs error 236 (V[t]=2t fixture)
  sink=64 / shared, per-head: FAIL, max abs error 27
  real TLI sinkswa-only: FAIL, max abs error 118
  Ran 2 tests ... FAILED (failures=5)
patched:
  Ran 3 tests ... OK
existing test_sparse_prefill_impl.py:
  T1–T7: 7/7 PASS
  T1/T3/T5/T6 max abs error 0; T2/T4 max abs error 5.22e-08
```

新增10组共用Q/K/V（按Q、K、V顺序，连续逻辑float32字节）的SHA256：
`2cc239e07256ba330f8702b2385140b83b7bcbecb8ba46001d3edadc6521317d`。
测试无随机输入；原7项沿用其已有固定种子。修前数值测试通过 `git show 0288ed5:sparse_attn/ops/eager_prefill.py` 加载原模块后重放相同测试，未篡改基线源码。原套件从该SHA提取到临时目录执行，测试代码未改。

复现（安装CPU torch/einops，在仓库根）：

```bash
python -m unittest discover -s sparse_attn/ops/tests -v
python test_sparse_prefill_impl.py
```

生产attention函数和真实indexer已在CPU执行；没有用NumPy原型代替它们。未测BF16/CUDA、真实HF wrapper到生成全链、长序列吞吐、TTFT或峰值显存。本次是正确性修复，**不主张性能提速**，测试耗时不作为性能基准。

## 历史影响与后续检查点

- 需筛选旧main、启用稀疏prefill、sink小于 `2×page_size` 的实验并重测质量及实际unique预算。默认128/page64数据不因153自动作废；没有完整运行清单，未认定任何具体历史分数已污染。
- 实验分支58745c4已有相同事实源修正，不能将本次结果当该分支GPU验收。
- 保留未完成工作：逐query选择与chunk2048执行tile解耦；候选内L2；全前缀无效QK/PV；紧凑索引；跨tile pipeline。优先减少算量/搬运，再验证单流与双流的真实收益。没有GPU证据，不合入高风险fused/异步kernel。
- 旧TL-PREFILL-STREAM-154不属于本次main修复：主线无实验分支的prefetch代码。未将实验分支默认关闭prefetch等同于显式开启时已安全。
- 此报告同时记录本轮有界修复检查点；不改他人负责的任务详情或实验数据，不关闭持续优化任务。
