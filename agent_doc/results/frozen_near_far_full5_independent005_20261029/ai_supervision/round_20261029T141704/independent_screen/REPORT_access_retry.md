**独立审查结论：pending，尚不能签发 `usable-with-scope`。**  
已完成旧 screen 的源码审计、100 样本覆盖与配对检查、输入重建、分数汇总、部分独立评分及 cache 记录一致性检查。主要缺口是 GovReport 官方 ROUGE 复算、独立参考/边界验证及生成复现；唯一 near-only 实例没有实际覆盖老化转 Far。

本报告中的数值是旧 screen 的核查结果，不是全量 1300 样本结论，不支持“无损”或算法等价主张。

**审查身份与范围**

我是当前用户明确指定的独立 `verify_experiment_result` 审查会话 `/root`，本轮没有参与该 screen 的生产，也没有将生产者的 pass、smoke receipt 或 parity 标记视为独立通过证据。会话身份及实际工具执行记录由外层宿主保存；本报告不伪造额外 reviewer 身份凭据。

全部检查使用只读 CPU 操作、`exec login=false`；没有联网、使用 GPU、派单、重新生成预测或修改源码、数据、项目文档。已读取指定 `result_validation_workflow.md`。

下文路径缩写：

- `R`：`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/tla/results/frozen_near_far_independent005_pool_v2_20261029`
- `S`：`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/tla/frozen005_pool_source_v2_20261029`
- `F`：`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/tla/frozen005_full_source_20261029`
- `P`：`/home/wangyuanshuo02/two-level-attention`

**六项 checks**

| Check | Verdict | 独立核查结果与证据 |
|---|---|---|
| `implementation` | **pass，限定静态实现及记录一致性** | 已追踪 post-RoPE profiling、严格独立 gate、冻结决策、分区 cache、TLI decode 调用链。72 个 manifest 绑定源码哈希全部匹配。证据：`S/benchmark/LongBench/prefill_frozen_skip.py:63–133`；`S/sparse_attn/ops/frozen_region_gate.py:22–54`；`S/sparse_attn/ops/frozen_region_cache.py:114–202`；`S/sparse_attn/ops/frozen_region_tli.py:28–82`。不包含对整个 GPU kernel 路径的独立执行认证。 |
| `reference_boundary` | **inconclusive** | 已阅读冻结、单 token decode、位置与容量等保护条件，但未执行独立张量参考或极端边界测试。原始结果有 8 个 `native_full_parity` 标记，不能替代本审查者的独立参考验证。唯一 near-only 实例未跨页，老化迁移没有运行覆盖。 |
| `data_integrity` | **pass，限定已核对资产** | 100 个唯一 `(dataset,row_index)`，与 selected_rows 完全一致；每任务 20 条，每条三臂完整，共 300 条生成结果。8 分片 protocol/identity 一致，日志分配符合取模分片。数据、答案、抽样、输入 token 哈希和输出解码均核对通过。证据：`R/manifest_rank{0..7}.json`、`R/requests/**/*.json`、`R/worker_rank*_attempt0.log`。模型权重与完整运行依赖绑定仍缺失。 |
| `numerical_sanity` | **pass，限定有限值、汇总与记录不变量** | 300 个 score 均有限且在 `[0,1]`；所有记录的 profile 指标有限且在 `[0,1]`。独立重算任务均值、差值、AVG5；检查 gate 决策、cache 容量/字节/守恒及 10,800 个末次 decode receipt，未发现不一致。GovReport 原始 score 的正确性尚未独立复算。 |
| `measurement_validity` | **inconclusive** | 三臂输入、样本分母、生成上限已核对；4 个任务共 240 个 score 经独立标准库实现复算一致。缺少 `rouge`、`jieba`，未完成官方 scorer 全量调用及 GovReport 的 60 个 ROUGE 分数复算。本轮不认证 wall time、峰值显存或性能收益。 |
| `reproducibility` | **inconclusive** | 已独立重建 100 个输入并复核 300 个输出解码；未进行模型生成重跑。运行依赖版本闭包、模型权重哈希、官方评分依赖尚未完整固定。`R` 内未发现 validation plan，现有 manifest 也不是规定的完整 `experiment-result/v1` 合同。 |

上述 pass 均有明确子范围；不能合并解释成六项全部通过。

**源码与分片身份**

`R/dispatch_plan.json` 指定的实际 screen 源码为：

```text
S HEAD = a1cdb2f4edd2447495175d90f75fa13b880d4afd
```

独立读取实际 HEAD，与 dispatch 和全部 8 个 screen manifest 一致。每个分片的 identity 都是：

```text
82b5f895017e5e4f70ef7e712615c22a756669f9e8de851ba65f314d093b86ec
```

重新对规范序列化的 protocol 计算 SHA-256，8 个均匹配。全部 72 个 `source_sha256` 条目与 `S` 当前文件匹配，结束前再次复核仍匹配。

指定全量源码实际 HEAD 为：

```text
F HEAD = 3e4a6885604ac3e719ee92a4906f87b532535314
```

在 screen 绑定的 72 个文件中，`F` 只有 `benchmark/LongBench/prefill_frozen_skip.py` 不同；其余 71 个一致。已阅读差异：增加 `samples=0` 全量抽样、抽样元数据和分片辅助函数；核心 gate/cache/TLI adapter 文件未变。

另只读查看了 `P/agent_doc/results/frozen_near_far_full5_independent005_20261029/manifest_rank*.json`：18 个 manifest 具有同一个 identity、同一份 source hash 映射，均声明 `3e4a688...`，选择数量为 `200+200+200+200+500=1300`。**这只核实声明的源码与选择范围，没有核验全量结果。**

**覆盖、输入与配对**

独立检查结果：

- 100 个 JSON 全部 `status=complete`，无重复 ID、缺项或额外样本。
- 每条准确包含 `full`、`far_0.05`、`dual_0.05`。
- 8 个分片日志分别有 `39、39、39、39、36、36、36、36` 条臂结果，均无重复，与 `index % 8` 的预期分配完全一致。
- 5 个本地数据集 SHA-256 与 manifest 一致；seed=42 重新抽样得到相同顺序，全部答案匹配原始数据行。
- 本地模型 `config.json`、`tokenizer_config.json` 哈希匹配。
- 离线加载 tokenizer，重建全部 100 个 prompt：原始长度、截断标记、最终长度、输入 token SHA-256 均匹配。
- 300 个 `generated_ids` 重新解码均等于保存的 prediction；长度与任务生成上限一致。

协议为 Qwen3-8B、BF16、context cap 31500、mean gate、阈值 0.05。生成上限依次为 Qasper 128、HotpotQA 32、Musique 32、GovReport 512、RepoBench-P 64。context cap 在套 chat template 前应用，不能直接解释为最终输入长度的硬上限；见 `prefill_frozen_skip.py:46–60`。

**质量重算：百分制**

下表由原始逐样本 score 重新计算。差值单位为百分点；AVG5 为五个任务均值的等权平均，计算前未进行任务分数舍入。

| 任务 | N | full | far_0.05 | dual_0.05 | far−full | dual−full | dual−far |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qasper | 20 | 56.100883 | 54.042059 | 54.042059 | −2.058824 | −2.058824 | 0 |
| HotpotQA | 20 | 65.523810 | 65.523810 | 65.523810 | 0 | 0 | 0 |
| **Musique** | **20** | **45.083333** | **45.083333** | **45.083333** | **0** | **0** | **0** |
| GovReport | 20 | 35.227082 | 34.835478 | 34.835478 | −0.391604 | −0.391604 | 0 |
| RepoBench-P | 20 | 72.050000 | 71.950000 | 71.950000 | −0.100000 | −0.100000 | 0 |
| **宏 AVG5** | **100** | **54.797022** | **54.286936** | **54.286936** | **−0.510086** | **−0.510086** | **0** |

评分核验层次：

- Qasper、HotpotQA、Musique：按源码的英文规范化、token F1、多个答案取最大值，独立实现并复算 180 个分数，全部匹配。
- RepoBench-P：按首个符合条件的代码行及 `difflib.SequenceMatcher` 口径，复算 60 个分数，全部匹配。
- GovReport：仅完成保存分数的聚合与有限值检查，**60 个 ROUGE 分数仍 pending**。
- 评分依据：`S/benchmark/LongBench/eval.py:181–203,264–279`、`metrics.py:35–57,109–116,133–163`。上述独立实现复算不冒称完整官方 scorer 调用。

Musique 的 20 个样本中，full 与 far 的生成 token 序列全部相同；整个 screen 的 100 个样本中，far 与 dual 的生成 token 序列全部相同。这是这些已保存序列的事实，不能推出一般等价性。

**实际 gate 触发覆盖**

以 dual 臂每任务 `20×36=720` 个层—请求决策为分母：

| 任务 | keep | far-only | near-only | both |
|---|---:|---:|---:|---:|
| Qasper | 659 | 61 | 0 | 0 |
| HotpotQA | 657 | 63 | 0 | 0 |
| Musique | 673 | 45 | 1 | 1 |
| GovReport | 664 | 56 | 0 | 0 |
| RepoBench-P | 619 | 101 | 0 | 0 |
| **合计** | **3272** | **326** | **1** | **1** |

因此：

- near-only：`1/3600 = 0.02778%`；请求覆盖 `1/100=1%`。
- both：`1/3600 = 0.02778%`；请求覆盖 `1/100=1%`。
- 两类分别只涉及 Musique 的一个请求，即各 `1/20=5%` 的 Musique 请求。
- far 臂有 327 个 far-only 决策；dual 将其中一个变为 both，另增加一个 near-only。
- 不能用本 screen 宣称 near-only/both 已获得广泛质量覆盖。

具体实例：

| 实例 | 层号，零起始 | near mean | far mean | 决策 |
|---|---:|---:|---:|---|
| `R/requests/musique/0056.json` | 11 | 0.0462979525 | 0.0344907008 | both |
| `R/requests/musique/0079.json` | 30 | 0.0467494801 | 0.3667117953 | near-only |

**机制与 cache 审计**

1. **末 256 query、严格独立 gate：静态符合。**  
   `frozen_region_gate.py:22–47` 使用最后 `min(256,length)` 个 query；对 post-RoPE Q/K 计算因果 softmax，按 query-relative 分区累加 mass，先平均 heads，再平均 query 行。`50–54` 分别执行 `near_mean < epsilon`、`far_mean < epsilon`，不存在 joint budget，也不是 `<=`。全部 10,800 个保存决策均与相应 profile、区域是否存在及臂配置一致。  
   限制：原始 Q/K 与逐 query mass 没有在本轮被独立重算。

2. **decode 冻结：源码与保存状态一致。**  
   `prefill_frozen_skip.py:95–104` 在 prefill 后决定并冻结；`frozen_region_cache.py:114–132` 拒绝变更已冻结决策；decode adapter 使用 layer 上的冻结值。全部 frozen/final 状态及末次 receipts 与决策一致；未获取逐 decode 步完整轨迹。

3. **near-only：保留 resident KV，但不参与 attention。**  
   `frozen_region_cache.py:126–130` 将 near 转为 `near_staging`，置 `attention_visible=False`；`194–202` gather 跳过不可见节点。TLI 仅接收 gather 后的可见 K/V，并将 mask 投影至可见位置，见 `frozen_region_tli.py:56–62,69–82`。

   唯一实例 `musique/0079` 层 30：冻结前后节点 KV 分配均为 **68,546,560 bytes**；staging 保留 4116 tokens，attention tokens 从 16596 降为 12480。最终 staging 为 4122 tokens，Far 仍为 12348 tokens。  
   **该请求只发生 6 次 decode cache 更新，没有跨页迁移，所以“老化转 Far”只有静态代码依据，没有该 screen 的实际迁移证据。** 迁移实现位于 `frozen_region_cache.py:167–177`。

4. **Far 淘汰与 both：保存记录一致。**  
   `frozen_region_cache.py:121–125` 移除 Far；both 同时移除 Near。所有 skip_far 实例的 frozen/final 节点中均无 Far。  
   `musique/0056` 层 11 冻结后仅 sink+SWA，132 tokens、540,672 bytes；永久丢弃计数由 14493 增至 14496，与三次 decode 更新一致。

5. **优化关闭与基线含义。**  
   运行入口明确设置：
   `TLI_OPERATOR_MODE=off`、`TLI_FINE_GATHER=0`、`TLI_SPARSE_PREFILL=0`、`TLI_DECODE_DENSE=0`，见 `prefill_frozen_skip.py:174–178`；所有 manifest 一致。运行走 tail32 TLI reference adapter。  
   **这里的 `full` 是关闭 region gate 的 TLI 臂，decode 仍使用 TLI 选择器，不是 dense full-attention decode 基线。**

6. **cache 数字不是峰值显存。**  
   已检查各节点 tokens≤capacity、分配字节、汇总字节、`stored_tokens + permanently_dropped = logical_length`、attention 可见计数及生成长度关系，未发现矛盾。该数据只计节点 K/V storage；gather 临时张量、projected scratch、allocator 状态等不在内。不能由这些数字推出实际峰值显存或延迟收益。

**关键文件 SHA-256**

以下哈希固定本轮读取的关键字节：

| 路径 | SHA-256 |
|---|---|
| `R/dispatch_plan.json` | `a8960f29aede6aacad63b92dfbf9d843cae9a7a06997d6a4347db5303ad15f4c` |
| `R/manifest_rank0.json` | `ae460b2a95babf579492379b433a7cab05f089571c1e2453be3e4b21b1c97e50` |
| `S/benchmark/LongBench/prefill_frozen_skip.py` | `2688ace8d52759f5c1fa80164b1c494b4682218e375193f2af1e3c4b5cde4630` |
| `F/benchmark/LongBench/prefill_frozen_skip.py` | `11daac6077e0985103b0f416c5749ae68ac0df8d1b5e1f259c6ce2cc584260a4` |
| `S/sparse_attn/ops/frozen_region_gate.py` | `a5ae01fea121425f908053821984ebd23c1f51a12f1a4fc18f02a90b0bf2d065` |
| `S/sparse_attn/ops/frozen_region_cache.py` | `1d5c46c25d29a907b6e584247b57256347af34cfa4f49d2c1125875c0692daa9` |
| `S/sparse_attn/ops/frozen_region_tli.py` | `62842a6200f45878a994e6bcca11b97ee45b4fed1d3ee4b0fa9787875ceef446` |
| `S/benchmark/LongBench/eval.py` | `dd945b78e93c0f267efa9099e4834b2081fd32268fe554f763f5372a0124f325` |
| `S/benchmark/LongBench/metrics.py` | `29a325fa4b4cb81660b1b560bc102d4682fdb39414b75eaa28ad19a2cffe04a0` |
| `R/requests/musique/0056.json` | `9d0c78d24d0d5f84a4a6b3e6c29f93a9916ac662c83a56584cc4e7c90e20e425` |
| `R/requests/musique/0079.json` | `a29cc7acc451ed495a302e77a705b082a2aede9b0c4276a60259c77fc620bd07` |

全部 100 个原始 JSON 的清单摘要：

```text
6404104e4c5fcd711415924b3d125b89df59be1a594d44404db51d6dd23c93ea
```

计算规则：按完整路径排序，将每个文件拼为 `相对R路径 + 空格 + 文件SHA256 + "\n"`，再对拼接文本计算 SHA-256。审查结束前重新计算，摘要未变。该摘要便于复核，但不替代正式合同要求的逐资产绑定清单。

**可用范围与待补证据**

当前可以报告：这批固定 100 样本的覆盖、输入配对、保存结果统计、4 个任务的独立评分匹配，以及上述源码和 cache 记录事实。它们可作为审查证据交付，**尚不授权作为已验收实验结果进入科学结论消费**。

解除 pending 至少需要：

- 完成与冻结评分口径一致的官方 scorer 复算，特别是 GovReport 的 60 个 ROUGE 分数，并绑定依赖版本。
- 补充独立参考与边界证据，尤其 near-only 跨页老化迁移；旧 screen 未覆盖这一行为。
- 补齐模型权重、评分及运行依赖的来源绑定，以及适用的独立生成复现证据。
- 固定正式 validation plan、完整结果资产绑定及宿主认证的本次独立审查记录。

本轮没有执行上述补实验，也没有替全量 1300 样本签发结论。