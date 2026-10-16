# TL-E152-TIMING-20261016：请求计时边界修复（非提速结果）

## 范围与根因

- 基线 main：`9e0310077902038f6886f531900900efffe5db23`，本轮重新 fetch 确认。
- 来源：用户要求真实入口、计时、汇总的完整成本可信；不改变算法、预算、
  chunk2048、selector/kernel 或人类 `agent_doc/guide/`。
- 旧 `exp/bench/e152_e2e_latency.py:88–120`：`request_wall_ms` 取
  `prefill_ms + sum(step_ms)`，而非连续起止差。首个 argmax、段间 CPU
  开销和同步边界被漏计；没有 TTFT 字段。`max_gen=1` 对空列表除零。
- 旧同文件 `137–140`：decode-only 的 assert 在 `python -O` 下消失。
  公共 parser 还允许 tia、SparQ 等，但本采集器非 TLI 一律打 native dense
  标签。TLI 的 K1/K2/grp/tail 标签也是常量，不随命令行实际配置改变。

## 采用方案与合同

修复位于同文件 `89–154`、`171–175`、`215–253`；精确源文件与测试 hash
见 [validation.json](../results/e152-timing-20261016/validation.json)。

| 字段 | v2 定义 |
| --- | --- |
| request_wall_ms | 输入 tensor 已搬运并同步后、prompt forward 前，到最后 greedy token argmax 完成并同步；连续 host perf_counter 差 |
| ttft_ms | 同一请求起点到首 token argmax 完成并同步 |
| decode_after_first_wall_ms | 首输出到末输出的连续 wall，包含中间 Python 与同步开销 |
| tpot_ms | 上一行除以实际剩余输出数；只有一个输出时 null，终端 N/A |
| prefill_ms / decode_ms | 原分段诊断（forward 段 / decode step 和），不可再合成 request wall |
| timestamps / per_token_wall_ms | perf_counter 秒时间戳与相邻输出间隔 ms，保留可复算边界 |

显式拒绝非法计数、未指定或非 none/tli 的方法；TLI 只接受
`TLI_SPARSE_PREFILL=0`（未设置沿用 0），不依赖 assert。命令行错误在
模型加载前退出，正常与 -O 行为相同。记录冻结 treatment manifest，
不再凭硬编码标签声称特定预算。

方案比较：保留分段和仅改字段名仍不能提供请求 wall，故不采用；以连续
时间戳闭合边界，保留逐步同步作诊断，属于最小修复。默认移除全部逐步
同步虽可能更快，但会改变原诊断协议，且无 GPU 不能验证，留作单独候选。
时间 O(G)、辅助空间 O(G)（G 为输出数），新增 token 间隔记录 O(G)；
无 SIMD、CUDA/PTX 或跨流修改。沿用本项目已读的请求计量合同与已有
LongBench capture 测试，不为本次 harness 修复另装 profiler。

## 修前失败与修后证据

原始短输出及两版实际函数的可控时钟记录在上述 validation JSON。
测试通过 AST 执行仓库内真实函数，排除重量级 loader import；没有修改
全局 torch 模块或 sys.modules。不是独立 GPU 模拟器，也不是模型测速。

固定输入 `[11,22,33]`，int64 little-endian SHA256：
`9357e9bb2a76ed5872f09c0e11cb677a0af2b01af70950c388dcac6e10563592`。
可控时钟中 forward=10ms、sync=1ms、argmax=7ms、unsqueeze=2ms：

```text
修前：Ran 6 tests; FAILED (failures=2, errors=14)
python -O 修前：Ran 6 tests; FAILED (failures=1, errors=15)
AssertionError: 31.000000000000004 != 42
ZeroDivisionError: division by zero
```

两版执行总时钟均为 43ms，剔除起点前初始 sync 后为 42ms。旧记录
request wall 却为 31ms，新记录为 42ms、TTFT=21ms、decode wall=21ms。
**这些全是设定时钟下的正确性证据，不能解释为真实性能变快或变慢。**
单 token 修前除零，修后 TTFT=request wall、TPOT=null。修后 E152 专项
8 项、既有 LongBench capture 11 项均在普通和 -O 下通过；E152 包含
实际 main 的两臂 warmup/measured JSON 写出、旧 v1 记录保留与 N/A 打印。
编译检查、diff whitespace 检查通过。

环境：Python 3.12.14，x86_64 / AMD EPYC 9V74；既有 LongBench 回归使用
Torch 2.8.0+cpu。E152 专项本身不需要 Torch/Transformers。GPU 架构、
驱动、CUDA 编译器和真实模型后端未验证；没有启动未知 GPU 工作。
质量、速度、真实显存、kernel 数量、host 同步实测均未新增。

## 历史影响与后续门禁

- schema 升为 `e152-e2e-latency-v2`。本轮 tracked 文件中未发现 E152
  实测结果；外部在飞/未入库结果状态未知，不声称均已重采或停止。
  v1 如已用于 E2，不可补写 TTFT 或把旧分段和改名作新 wall；保留原件，
  重跑同输入 hash、预算、dtype 的成对两臂。也不能把 v1/v2 的 TPOT 混合。
- v2 明确 `instrumentation_mode=per-step-device-synchronized` 和
  `backend_verified=false`。逐步同步有扰动，非自然 serving 时延；输入
  H2D、分词、模型加载、清理、文本解码和传输均不在请求 wall 内。
  必须另做真实路由核验、无逐步同步的低扰动计时及 CUDA/实模校验后，
  才能作为 E2 科学验收依据。none/TLI 的 dense prefill 意图不等于
  已证明 SDPA/FlashAttention 的实际派发。
- JSON 追加仍要求编排器串行写同一文件；未验并发 writer。理论 KV
  字节仍是声明的请求长度公式，不等于实际 KV cache/allocated 峰值。
- 后续性能候选：核验本部署 Qwen3 `model(...)` 是否物化全 prompt logits，
  以及支持的 last-logit 参数和峰值生命周期；当前没有 Transformers/实模
  证据，不修改模型调用，也不将此假设认定为 1.x prefill 的已证根因。
- 紧凑索引、候选内 L2、fused indexed attention 与 overlap 仍需授权的
  可用 GPU、同输入全成本对照；本次测量修复没有代替这些优化。

CODE-T001（本轮）契约→复现→修复→CPU验收已完成；发布后远端核验由
本轮私有 publication 检查点记录。CODE-T002（待办）依赖本修复，在有
已授权设备时验证真实路由及低扰动 E2；失败则保留诊断，不进入论文主表。
