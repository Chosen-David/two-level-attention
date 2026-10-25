# TL-PREFILL-SCORE-BUDGET-20261019：GQA score 预算漏算查询头

## 结论与边界

审计基线 `bcd5b58c75628af45a0bd1cc840647f583cc4d7b`。
`sparse_attn/ops/eager_prefill.py` SHA256：
`bd05914d4e7f6471f2287956b665ba4138d140663e4a83654eafc98be9752352`。

显式选择 `TLI_PREFILL_RUNTIME=candidate`，进入共享 KV 选集的 eager sparse
路径时，行子块预算使用 Hkv，但实际 fp32 score 有 Hq=Hkv×G 个头。
默认 runtime 是 reference，本问题不能泛化到全部默认请求。
无组模式不会进入要求 `sel_list` 且 `grp>0` 的 fused 分流。
本次仅提交源代码绑定的形状复算及审计记录，**没有生产修复**。

## 证据链与最小形状反例

- `eager_prefill.py:50-62`：单 score 预算为 `1<<27` 元素，即 fp32 512 MiB。
- `:95-96`：`rhg=H if per_qh else Hkv`，这是 gather 索引头数。
- `:174-177`：共享选择将这个 Hkv 传入 `_row_block_size`。
- `:199-209`：无组 W 为静态容量与前缀长度的较小值，pad 列也实际占据 score 空间。
- `:242-255`：`einsum(... -> r h g w)` 产生 `[R,Hkv,G,W]`，应按 Hq 计元素。
- `:386-395`：普通 top-k 的容量为 K2+2×block+sink+SWA。

固定外层 chunk2048，输入长度4096，检查第二个 chunk `[2048,4096)`；
普通 top-k、非 per-query-head、无组，K2=1024、block64、sink128、SWA128、D128，
容量1408。以下来自实际 helper 的 AST 提取执行和调用者形状公式，不分配张量：

| Hq/Hkv/G | helper R | 实际行数 | W | score 元素数 | fp32 逻辑载荷 MiB |
|---|---:|---:|---:|---:|---:|
| 32/8/4 | 3347 | 2048 | 3712 | 243269632 | 928 |
| 64/8/8 | 3347 | 2048 | 3712 | 486539264 | 1856 |
| 32/32/1（MHA 对照） | 1378 | 1378 | 3042 | 134140032 | 511.70361328125 |

Hq32/Hkv8 的头配置也见 `exp/bench/e138_grp_bitdiff.py:24`，但该历史脚本使用组模式，
不能将上述无组反例当成它的实际运行结果。公共组模式调用在 `:562` 传 `w_sel_cap=0`，
按完整 chunk 的实测选集并集计宽；**未确认“组并集被单组选集上限低估”**。
`max(grp, ...)` 可能把最小组抬到预算以上属于另一个待核验边界，不作为本次已证实缺陷。

## 复现、环境与原始短日志

运行 `python agent_doc/results/prefill-score-budget-20261019/probe.py`。
加 `--require-compliance` 应返回1：这是现存预算反例，不是通过的 GPU 回归。
脚本固定源文件 SHA256，源码变化即拒绝继续；输出完整形状配置及配置 SHA256。
不存在张量输入 hash，因为没有生成或读取张量输入。

实跑配置 SHA256：`6b97415b790272011eab1162ada51d3bc3c45319714d13d7b8ab90322e48aa6a`。
普通 Python 与 `python -O` 各执行普通模式和预算门禁，共四次输出一致；
门禁失败是预期的未修复反例，不是跳过失败。原始短日志：

```text
mode= [] compliance= False exit= 0 payload_MiB= [928.0, 1856.0, 511.70361328125]
mode= [] compliance= True exit= 1 payload_MiB= [928.0, 1856.0, 511.70361328125]
mode= ['-O'] compliance= False exit= 0 payload_MiB= [928.0, 1856.0, 511.70361328125]
mode= ['-O'] compliance= True exit= 1 payload_MiB= [928.0, 1856.0, 511.70361328125]
PASS: four runs match; two expected violations and one MHA control
```

环境 Python3.12.14；Torch 未安装、`nvidia-smi` 不存在；未接入 GPU，设备架构、
驱动、CUDA/Triton 及生产 kernel 配置不可验证。只有标准库计算，无模型推理、
JIT、warmup、kernel 发射或 GPU 计时。独立只读复核确认上述缺少 G 因子的证据链，
并纠正了组模式容量推断。质量/速度/实测峰值内存变化均未测。

**928/1856 MiB 是一个 score 张量的逻辑载荷，不是实测进程峰值、OOM 或性能退化。**
softmax、gather、其他张量和 allocator 还会占内存；即使修好该预算也不代表限制总显存。

## 待验证修复与历史重测范围

区分 index_heads 与 score_heads：前者保持 Hkv 的 gather/mask 维度，后者按 Hq
确定执行行 tile；不可全局替换 `rhg`。保持外层 chunk2048、算法粒度、选集、sink/SWA
及预算不变。处理单行或最小组本身超预算的边界，不以放大预算解决。

先用实际张量验证共享/per-query、MHA/GQA/MQA、强制子分块、组对齐、sigma、
dense/fused 分流及尾块/stride，再在授权 GPU 上对照语义参考、检查数值误差。
行 tile 改动可能改变数值执行路径，故本环境不直接合入生产修改。
之后同输入 hash、dtype、预算、计时口径测完整选择/打包/attention 成本和峰值显存，
不能只测单 kernel。历史 candidate eager 共享选择的 GQA/MQA 数据需要在修复后重测；
本次不改历史数据，也未证明过去运行失败。详见任务检查点。
