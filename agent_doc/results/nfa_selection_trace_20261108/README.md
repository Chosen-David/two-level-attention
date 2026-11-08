# NFA 真实选择轨迹（M1）

## 目的与固定范围

本数据包用于回答审稿矩阵 **M1：真实 L1/L2/Gather 选择轨迹**。它不是质量或性能实验，不测量 TPOT，也不扩展为全数据集 trace。

预先固定的轨迹范围：

- 模型：Qwen3-8B，最终 attention 使用完整 K/V；
- 输入：原生 RULER 32K `niah_single_1` sample 0；
- 执行：dense prefill 后的第一个 eager decode step；
- 层：第 0、17、35 层；
- NFA：sink=4、SWA=128、K1=128、K2=1024（含保护区）、near pages=32、near length=4096、page=16、tail32，Near/Far 为 Avg/MinMax；
- gate：dynamic-frozen，epsilon=0.05，物理 KV retain，无旧首步 seed/core-token 补偿。

原生输入文件 SHA256：

```text
e851c310be0cc7d1fa32469c9443f8fb2dd65ab6b3b28d8f2d700db86399c2db
```

sample 0 为 32,612 个 token；按 little-endian int64 token IDs 计算的 SHA256：

```text
db96ef119fc1815ce43912b85b17180f6b33a5ae2eca2dbae6056643c2008999
```

## 文件契约

- `export_selection_trace.py`：真实模型导出器；外置包装生产 selector，在返回后快照持久 workspace，不修改生产热路径。
- `validate_selection_trace.py`：独立 NPZ 校验器；不加载模型，不导入生产 selector 或导出器。
- `selection_trace.npz`：真实轨迹原始数组。
- `manifest.json`：输入、模型、源码、配置、环境、坐标和逐数组 shape/dtype。
- `receipt.json`：导出完成回执及 NPZ/manifest SHA256。
- `validation.json`：独立校验结果，包含生产工作点与数组语义两类硬门。
- `metadata.json`：实际执行环境、共卡/锁卡状态、单位和适用边界。
- `export.log` / `validation.log`：真实导出与独立校验原始日志。
- `SHA256SUMS`：发布文件身份。

只有真实生成 `receipt.json` 且独立校验输出 `NFA_SELECTION_TRACE_VALIDATION_PASS`，本包才属于已完成证据；脚本或 plan-only 输出本身不代表 trace 已完成。

## 复现

默认命令只做计划检查，不启动 GPU：

```bash
python3 agent_doc/results/nfa_selection_trace_20261108/export_selection_trace.py
```

真实执行需要一张空闲 CUDA 卡、本地 Qwen3-8B 权重和上述原生 RULER 文件。运行前必须确认没有其他计算进程或他方 reservation：

```bash
CUDA_VISIBLE_DEVICES=<空闲卡> python3 \
  agent_doc/results/nfa_selection_trace_20261108/export_selection_trace.py \
  --execute \
  --model-path /path/to/Qwen3-8B \
  --data-root /path/to/RULER \
  --layers 0,17,35
```

`--data-root` 下应有 `32768/niah_single_1.jsonl`。导出器在模型加载前显式固定 page=16，并在加载后核验全部 36 层 indexer，避免通用 helper 的 page=64 默认值静默改变工作点。`nfa_enable_layer_skip=false` 只关闭旧静态 D′ 掩码；本包 strict-skip 来自 dynamic-frozen `GateConfig`。导出器拒绝覆盖已有 NPZ、manifest 或 receipt；失败时应保留日志并修复，不能生成占位结果。

独立校验（正式 M1 必须带 `--require-production-scope`）：

```bash
python3 agent_doc/results/nfa_selection_trace_20261108/validate_selection_trace.py \
  --package-dir agent_doc/results/nfa_selection_trace_20261108 \
  --require-production-scope
```

## 数组坐标与内容

每层使用 `layer_XXX/` 前缀。逻辑 token ID 从 0 开始；第一个 decode step 的 `planned_length = prompt_length + 1`。物理地址定义为：

```text
physical_slot = physical_page * page_size + page_offset
```

保存的主要数组包括：

1. 当前 post-RoPE Q、L1/L2 缩放后的 Q 和实际 tail32 坐标；
2. 页 min/max/sum、页统计长度、L1 far/near scores 和 Top-K pages；
3. 联合 candidate IDs、有效掩码、边界和配额；
4. candidate 来源：L1 far、L1 near、sink/SWA boundary bypass、sink/SWA protected；
5. packed L2 keys、解码 token、BF16 原始位及以 FP32 无损表示的 BF16 分数；
6. middle IDs/counts、最终逻辑 IDs、sink/far/near/SWA 区域标签；
7. 最终 Gather 物理 slots、每 KV head visible count 和逻辑页到物理页表；
8. 实际 paged-pool Gather K/V，以及从原生 prefill K/V 加当前 decode K/V 独立重建的 K/V，二者均保存 uint16 BF16 payload。

无效 candidate、middle 和 final 尾槽的区域标签均为 0，避免 padded token 0 被误标为 sink。

## 独立硬门

校验器逐项检查：

- 正式模式下的固定输入 SHA、token IDs SHA、单请求首步 eager 范围、第 0/17/35 层、dynamic-frozen gate 和完整 NFA 工作点（包括 page=16）；
- NPZ 与 manifest 的 SHA256、数组清单、shape 和 dtype；
- 最终逻辑 ID 的范围、唯一性、严格递增和无效尾槽；
- 每个有效物理 slot 经 block table 唯一反解到逻辑 token，地址回算一致；
- 最终集合严格等于 `valid middle ∪ sink/SWA protected`，冻结区域不重新进入；
- signed-BF16 ordered packed key 的 token、BF16 bits 和分数解码；
- 有效 L2 分数只引用有效 candidate，candidate 来源和区域符合 virtual-slot 布局；
- Gather K/V 与原生逻辑 K/V 的 BF16 payload 逐位一致。

## 适用边界

本包只证明指定真实请求、指定三层、第一个 eager decode step 上的选择集合、物理地址和 K/V 对应关系。它不外推为所有请求/层/步的正确性，不提供任务质量结论，也不提供 eager/Graph TPOT 或显存结论。both-skip 的 `protected_only` 轻路径没有真实 L1/L2 工作；若目标层走该路径，导出器会 fail-closed，而不会伪造 L1/L2 轨迹。
