# TL-PREFILL-SPEEDUP-DIRECTION-156：E139 对 dense 的加速判决方向颠倒

## 结论与范围

- 状态：**confirmed，P1（论文/验收结论错误，非已复现的 kernel 输出错误）**。
- 核验主线：`e87ba7133c1670bd59a63456e044eb92b21532e7`；实验对照：`e138-ablation@39ab56c43b2abe7c9fbec9239a6673d319b464eb`。
- 本次只提交这份纠偏记录到 main；未修改原始实验数据，未修改实验分支，未将实验实现合入 main。以下融合态数字不是 main 的性能。
- 新增范围：实验分支 `b0f40cc..39ab56c` 的融合 attention、验证/性能脚本和判决。主线源码未新增变化；此前 153/155 修复不重复报告。

**64K 融合态尚未比 dense 快。** 日志的 `dense / fused = 0.73` 正确，判决却将其倒数 `fused / dense = 1.37` 称为净加速。倒数在这里是耗时倍数，而不是加速比。不能据此宣称达到“64K 对 dense 净加速”的目标。

## 定位与触发

以下均是上述实验 SHA 下的 `exp/trace/results/` 文件：

| 文件 / 行 | 问题 |
| --- | --- |
| `e139_gang4_wall_lsweep.json:16–24` | 539.6 ms 对 395.3 ms 被解释为 1.37×净加速，并据此升级论文主张、推断 48–56K 交叉点 |
| `e139_gang4_k2_64k.json:12–13` | 将 0.69–0.73×解释为全预算档 1.37–1.45×净加速，误判 K2=2048 质量/性能双达标 |
| `e139_selchain_seg_bench.json:12–13` | 继续使用“当前 1.37×净加速”的错误基线解释后续优化空间 |

对应 wall 脚本输出的是 `td/t_fus`，原日志与 JSON 数值字段方向一致；错误发生在自然语言判决，不是此处计时比值公式。未证明原计时本身有效或无其他问题。

## 按原始数值重算

固定定义：`speedup = dense_ms / candidate_ms`；仅 `speedup > 1` 为更快。`candidate_ms / dense_ms - 1` 为相对耗时增幅。

| 条件 | dense ms | fused ms | speedup | fused 耗时增幅 |
| --- | ---: | ---: | ---: | ---: |
| 64K，L 曲线汇总 | 395.3 | 539.6 | 0.732580× | +36.504% |
| 64K K2=1024，原日志 GPU6 | 394.8 | 541.3 | 0.729355× | +37.107% |
| 64K K2=1536，原日志 GPU6 | 394.8 | 567.1 | 0.696174× | +43.642% |
| 64K K2=2048，原日志 GPU6 | 394.8 | 569.3 | 0.693483× | +44.200% |
| 64K K2=1024，原记录 GPU7 | 398.2 | 545.7 | 0.729705× | +37.042% |
| 64K K2=1536，原记录 GPU7 | 398.2 | 573.9 | 0.693849× | +44.124% |
| 64K K2=2048，原记录 GPU7 | 398.2 | 575.5 | 0.691920× | +44.525% |

8K/16K/32K 的 L 曲线分别为 0.125230×、0.219949×、0.401121×，也都慢于 dense。**所测四种长度均无交叉，48–56K 交叉点不受现有数据支持**；不推断未测长度永远不能获益。

同一文件的增量也应纠正：`539.6 - 249.8 = 289.8 ms`，不是“32K→64K 增量 144 ms”。dense 对应增量 `395.3 - 100.2 = 295.1 ms`。近似相等的两个增量不能用于证明已经越过 dense。

仍成立的比较必须保留：64K 相对旧逐组路径 `2302.2 / 539.6 = 4.266×`，相对批量但未融合路径 `1495.1 / 539.6 = 2.771×`。这是相对旧稀疏实现的显著改善，不等于优于 dense。

## 实际复核与最小复现

本次实际执行 Python 3.12.14 标准库，直接从指定 git SHA 读取 JSON，重算四个长度及两设备×三个预算，共 10 个条件；全部 `fused > dense`。并阅读 `e139_g4lsweep_g4.log`、`e139_g4k2_64k_g6.log` 的计时输出，与相应 JSON 数值一致。

```python
import json, subprocess
rev = "39ab56c43b2abe7c9fbec9239a6673d319b464eb"
def read(name):
    return json.loads(subprocess.check_output([
        "git", "show", f"{rev}:exp/trace/results/{name}.json"
    ]))
rows = read("e139_gang4_wall_lsweep")["results"]["lsweep_gpu45"]
for name, row in rows.items():
    if name.startswith("L"):
        d, t = row["dense"], row["fused"]
        print(name, round(d/t, 6), "faster" if t < d else "slower")
        assert t > d
rows = read("e139_gang4_k2_64k")["results"]
for gpu in ("gpu6", "gpu7"):
    d = rows["dense_sdpa"][gpu]
    for key in ("fused_k1024", "fused_k1536", "fused_k2048"):
        t = rows[key][gpu]
        print(gpu, key, round(d/t, 6), "slower")
        assert t > d
```

短输出：

```text
L65536 speedup=0.732580 latency_multiplier=1.365039 latency_increase_pct=36.504
gpu6 fused_k2048 speedup=0.693483 latency_increase_pct=44.200
gpu7 fused_k2048 speedup=0.691920 latency_increase_pct=44.525
CHECK: 4 length cases + 6 budget/device cases are slower than dense; no tested crossover
```

输入 SHA256：

- `e139_gang4_wall_lsweep.json`：`de3e226e032259c2be6272bd47e361fc90a53581012c87ef0339044399e3e52f`
- `e139_gang4_k2_64k.json`：`71e26b81626ff40bbd946d7372a1bff485214532da5e556df7369a386a893a30`
- `e139_selchain_seg_bench.json`：`93dceef307244711a76218b995738942895593a8e1330fc849f04c8846520b29`

## 修正、历史影响与下一门禁

1. 本记录替代上述三个实验判决中的“已对 dense 净加速”“64K 全预算质量/性能双达标”“48–56K 交叉”结论。生成图表、合并实验分支或写论文时必须应用纠偏；不改写测量值使其符合目标。实验分支原判决文字尚未由本任务修改。
2. 将 summary 生成改成由同一数值函数产生 `speedup`、`latency_change_pct`、`faster/slower`，并以这组实际负例回归；不要手工把倒数填入“加速比”。这属于后续结果生产器门禁；当前仓库记录未展示生成这些判决文字的可修函数，本次不虚构已修复生产器。
3. 既有输出精度、相对旧稀疏实现的提速不因此失效；但“优于 dense”的论文性能表、目标达成状态及派生叙述必须复查。未据此判断其他 decode 结果。
4. 后续选择链优化应同输入、同预算、同精度、同设备重测整段 wall 与 dense。将选择链压到 100 ms 是假设，不能当已获得 1.6×的证据。`total - standalone_selector` 是残余成本估计，包含 attention、其它选择/胶水/打包等，不能无 trace 就称为纯 attention 时间。
5. 维持外层 chunk2048；本次未修改 kernel、选集语义、预算或数据。上一轮 host 标量读回消除候选仍待可靠性能验证，不随本报告发布。

### 未验证边界

无本次可用 GPU 实测；未重跑 CUDA/Triton、内存/竞态检查、真实模型质量、TTFT 或完整请求；不声称 kernel 修复或新提速。实验记录描述单流、fresh indexer、2 warmup、7 次中位数和设备编号，但本文没有独立核实设备架构、驱动/编译器版本、实际 SDPA 派发、kernel trace 与输入张量 hash。原记录的 15/15 对拍属于远端报告，非本次执行。上述算术纠偏不依赖这些尚待核实的 GPU 条件。
