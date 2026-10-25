# E140 增量审查：整数配额阈值与 mass 探针 oracle

## 范围与权限

审查 main `a28f05225dabcce5539bb89240e622815d9a429f`，增量为
`6a8da9d..a28f052`；实验分支 `e138-ablation` 仍为
`e1edf9c316fc4190dd0c29bc5d47477136f8bdab`。本次新增的是 E140 探针、
汇总和图生成逻辑，没有生产 selector/kernel 改动。

沿真实代码核对了 decode 的 nb_near/nt_near/far_budget、prefill 的对应预算，
以及新增探针的 reference → L1 → L2 → retained_mass 链路。未重新审查全库。
`agent_doc/README.md:29` 将 guide 目录定义为 AI 只读，人类 GUIDE.md 亦明确禁止
AI 修改。因此保留 `guide/paper_figs/` 内的脚本和图，以及既有实验数据；本报告
提供可运行复现与修正方案，**未声称原脚本或图已修复**。本轮不以旁路复制脚本
绕过编辑边界，不修改生产算法来迎合图中结论。

## 新增发现

| ID / 状态 | 定位（上述 SHA） | 违反的契约及影响 |
| --- | --- | --- |
| TL-E140-QUOTA-ROUNDING-20261015 / confirmed / P2 | `agent_doc/guide/paper_figs/figs/fig_a_core_ablation.py:117,120–122,204–215,233–235`；`exp/trace/results/e140_far_quota_mechanism_verdict.json` 的 `mechanism.far_zero_threshold_K1` | 把连续近似当作带 round 的精确整数阈值，图中的三条 iff 标注均有边界反例。另有 `e140e_f_sinkswa_cliff_verdict.json` 把 K2=896 写成总召回1152，与同文件/生产代码的 K2_mid=640 冲突。 |
| TL-E140-MASS-ORACLE-20261015 / confirmed / P1 机制证据 | `agent_doc/guide/paper_figs/trace/e140_l1_dilution_probe.py:3–6,97–128` | “exact”臂按跨 head 原始 logit 求和选 token，却以逐 head softmax 后平均的 P 评估。即便没有量化、没有子空间误差，候选超集扩张也不能保证该 P 的 retained mass 单调。文首用于判决的单调性前提不成立。 |

## 1. 配额：连续交点不是精确整数边界

生产代码 `sparse_attn/indexer/tli_indexer.py` 的 E64 分支使用
`nb_near=max(1,int(round(K1*beta)))`、`nt_near=min(int(nb_near*bs*gamma),K2_mid)`；
`sparse_attn/ops/batch_select.py` 的批量预算沿用该顺序。
默认 beta=.125、bs=64、gamma=.625、sink=swa=128，长前缀且池宽充足时：

| K2 | 当前图例的 far>0 条件 | 实际首次 far=0 的整数 K1 | 阈值前一档 far 配额 |
| --- | --- | --- | --- |
| 1024 | K1<154 | **156** | K1=155：8 |
| 896 | K1<128 | **124** | K1=123：40 |
| 768 | K1<100 | **101** | K1=100：32 |

特别是 K1=154/155 时 round(K1/8)=19，nt_near=760，far=768−760=8，
并未归零。153.6 可以作为忽略舍入的连续交点，但不能标为代码精确阈值。
图的稀疏采样点大多没有踩到这些边界，因此曲线点值正确也不能验证 iff 文本。

可采用的修正（在有权编辑图源时）：

```python
def far_budget_exact(k1, k2, beta=.125, bs=64, gamma=.625, sink=128, swa=128):
    mid = max(0, k2 - sink - swa)  # 此处限定长前缀的配额，不代表所有实际入选数
    near_pages = max(1, int(round(k1 * beta)))
    return max(0, mid - min(int(near_pages * bs * gamma), mid))

for k2 in (1024, 896, 768):
    first_zero = next(k for k in range(1, 1025) if far_budget_exact(k, k2) == 0)
    print(k2, first_zero, far_budget_exact(first_zero - 1, k2))
# 1024 156 8
# 896 124 40
# 768 101 32
```

阈值、阴影、图例和 TSV 应从同一整数函数生成，并增加阈值±1 的回归。
一般 gamma 下还需保留 `int` 截断；短前缀、保护区重叠和池宽截断另行处理，
不能把预算上限当成所有样本必然入选的 token 数。

同一预算身份下，K2=896 已包含 sink128+swa128，mid=640；不能再加256称为1152。
需勘误 `e140e_f_sinkswa_cliff_verdict.json` 的 protocol、e140f 和 quest_caliber
叙述；保留其原始分数，不重写历史运行身份。K1=256/512 时 far 配额为0、
K1=128 时 K2<=896 会使 far 配额为0的结论仍成立，不因边界勘误撤销。

## 2. 探针：“不量化”不等于 mass oracle

reference 由 `group_mean_probs` 计算 `mean_h softmax_t(score_h,t)`。
而 probe_group 的 exact 路径用 `einsum("hd,td->t",q_s,k)`，即先对 head 求和，
再对 token 排序。二者一般不同；即使后来给 sum-logit 加 softmax，也不等价于
先逐 head softmax 再平均。候选集合的包含关系不能改变这一事实。

### 实际执行原函数的最小反例

在仓根运行下列代码。AST 仅提取仓库原函数以避免启动模型/写输出；只有设备和
输入规模常量缩小。两个维度均保留，没有子空间降维损失；本例 q4 不改变用于
点积的数值，因此可隔离出 oracle 问题。

```python
import ast, hashlib
from pathlib import Path
import torch
p = Path('agent_doc/guide/paper_figs/trace/e140_l1_dilution_probe.py')
names = {'group_mean_probs', 'probe_group', 'quant4_rows'}
ns = dict(torch=torch, DEV='cpu', SINK=0, SWA=0, BS=2, K2MID=1, SUB_IDX=[0, 1])
nodes = [n for n in ast.parse(p.read_text()).body
         if isinstance(n, ast.FunctionDef) and n.name in names]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(p), 'exec'), ns)
q = torch.tensor([[1.,0.], [1.,0.], [0.,1.], [0.,1.]])
k = torch.tensor([[10.,-10.], [-1.,-1.], [0.,1.], [-100.,-100.], [-100.,-100.]])
P = ns['group_mean_probs'](q, k[:, None, :], 1.)[0]
print(hashlib.sha256(q.numpy().tobytes() + k.numpy().tobytes()).hexdigest())
for mode in ('exact', 'q4'):
    for k1 in (1, 2):
        print(mode, k1, ns['probe_group'](P, q, k, 1., k1, mode))
```

原始短输出（返回 overlap、retained_mass、page_recall）：

```text
d522155a005ef114ba69ffe67235119b014e33f21ed8f1fbd8bdcbc47902bb36
exact 1 (1.0, 1.0, 1.0)
exact 2 (0.0, 0.8808712363243103, 1.0)
q4 1 (1.0, 1.0, 1.0)
q4 2 (0.0, 0.8808712363243103, 1.0)
```

L1 第一次只选第0页，再扩张为第0/1页，严格嵌套；full-reference 最佳 token0
概率约0.499976，新增 token2 概率约0.440415。但 sum-logit 从token0的0换成
token2的2，导致 retained_mass 下降。page_recall 始终1，不能发现这个排序目标差异。

### 正确的实验拆分与验收

- 单调性 oracle 应直接在同一个固定 full-reference `P` 的嵌套候选集合内 top-k，
  用相同 P 计算 retained mass；此时固定预算的最优质量才有超集单调性。
- 保留现有臂时应改名为 subspace sum-logit unquantized，撤销其单调保证。
  若目标是贴近生产，再增加“逐 Q-head softmax → GQA mean”的子空间未量化臂，
  明确 softmax 是全前缀还是 L1 掩码后的归一化；归一化域变化本身也是因素。
- full-P oracle、子空间影响、head 聚合顺序、量化影响应分臂隔离。q4 对照使用同一
  Q/K、候选集合、聚合和归一化规则；不能把多个差别归于量化。
- 先加入本反例及随机嵌套候选测试，再重跑真实样本。现有探针结果保留为该代理
  算子的观测，不作为“生产量化稀释假说被严格排除”的充分证据。

## 验证与历史影响

实际环境：Linux x86_64、Python 3.12.14、torch 2.8.0+cpu；
`PYTHONPATH=/tmp/twolevel-cpu-deps:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`。
执行了原绘图函数的 K1=1..1024 三档整数阈值枚举，以及上述原探针函数反例。
没有 CUDA 设备实测、GPU profiler、模型/数据运行或性能结果；驱动/目标 GPU 架构/
编译器/kernel 配置未验证。本报告不宣称 prefill 提速，也不将 CPU 反例幅度外推到
Qwen3/LongBench 实际损失。

受影响：A 图 E/F 的精确边界标注、相关判决预算文字、E140g 的机制证据解释。
E140a/b/e/f 已记录的模型分数没有被本次测试直接证伪，不改原始实验数据。
生产 far=0 配额机制有独立代码证据；需要降级的是“精确边界”和“该 mass 探针
已排除其他机制”的强表述。K1页数减半也不能直接称完整请求性能更优；需要保持
同预算/质量及真实派发的同机全链测速。

下一检查点：有权编辑 guide 图源的维护者落实上述修正并重新导图；先修正 oracle
再重测 E140g，不直接重跑原错误判据。现有 GPU blocklist 候选、Graph schema
GPU 对拍与实际派发验证仍待有授权的空闲设备，本次未更改其验证状态。
