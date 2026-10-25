# E139 Graph PoC：先修 near 派发与硬门，再认定生产收益

审查时间：2026-10-15。main `4059a93c25c4fc3888edf7dc6a3aba1a16210351`；实验分支 `e138-ablation` `8f1ac8c6b090672aee3a8be70ced28a159892859`（本次新增四个脚本/结果文件）。main 实现未变。本报告不合并实验分支，不修改历史实验数据，不声称生产修复或 GPU 验证完成。

## 新发现

| ID / 状态 | 位置（上述实验 SHA） | 影响与修复要求 |
|---|---|---|
| TL-E139-GRAPH-ORACLE-20261015 / confirmed / P1 实验证据 | `exp/bench/e139_l2_graph_poc.py:111–125,263–279,439–453` | 三份 L1 计算都无条件调用 `_fast_score_coarse`，near 也用 minmax。实际 `make_args("mavg_tail")` 继承 `e131_arm_bench.py:132–148` 的 far=minmax、near=avg；生产 `batch_select.py` 按方法派发 near 到 `_fast_score_fine(k_avg,...)`。PoC 的 eager 与 graph 共享此错误，0/1008 只说明二者自洽，不证明等价于生产。 |
| TL-E139-GRAPH-GATE-20261015 / confirmed / P2 测试可靠性 | 同文件 `247–250,298–301,387–390,475–476`；`e139_l2_subseg.py` 末尾 split_selfcheck | 差异计数/布尔值只打印，非零仍会继续计时并输出 DONE；zip 也会忽略尾部长度差异。应显式核对列表长度、shape/dtype 和逐位相等，失败抛异常、非零退出，且在计时前执行。不要只加会被 `python -O` 删除的 assert。当前提交日志确实显示零差异，不能据此断言已发生错误结果发布；缺陷是门禁不执行。 |

### 根因与受影响结果

Graph 脚本没有读取/注入 `k_avg`。原 near=avg 在初始 L1、第二 seed 的候选生成、L1 graph 静态输入三个位置均变成 minmax。相对地，本次 `e139_l2_subseg.py` 已保留 `far_use_mm/near_use_mm` 派发。因此两份测量并不是同一 selector 算法；图内 L2 的候选也来源于错误的 near L1，不能把 subseg 的 eager 数字直接接到 graph 的数字上。

需重新验证 `e139_l2_graph_poc_verdict.json` 的 L1 9.5x、L2 1.64x、选择链约206ms与整层约324ms投影。不是证明 Graph 无效，也不是否定既有未使用该 PoC 的质量实验。原日志与 JSON 的 eager/graph（314.0/192.3 vs315.5/192.2）亦并非完全一致，后续须绑定具体 run 的原始样本；不要直接覆盖历史数值。

## 实际执行的最小 CPU 复现

环境：Python 3.12.14，torch 2.8.0+cpu，Linux x86_64；无 CUDA/Triton/nvidia-smi。下面从固定 Git blob 提取真实 `l1_loop` 函数，不执行 GPU 顶层脚本。两页输入第一维分别可由 `[-10,10]` 和 `[1,1]` 产生，故 avg=[0,1]、max=[10,1]；q=+1。mock 仅实现这一标量情形的独立打分公式，验证 Python 派发，不替代 Triton/生产模型测试。

```python
import ast, subprocess, torch, types
rev = "8f1ac8c6b090672aee3a8be70ced28a159892859"
def source(p):
    return subprocess.check_output(
        ["git", "show", rev + ":exp/bench/" + p]).decode()
a = source("e139_l2_graph_poc.py")
b = source("e139_l2_subseg.py")
class Scores:
    def _fast_score_coarse(self, lo, hi, q, dtype):
        return hi[:, 0, 0].reshape(1, -1)
    def _fast_score_fine(self, avg, q, dtype):
        return avg[:, 0, 0].reshape(1, -1)
ctx = dict(torch=torch, N=1, H=1, Tb=2, qdtype=torch.float32,
    dev="cpu", b_q=torch.ones(1,1,1), nb_l=[0], sl_l=[2],
    kf=0, kn=1, far_lo=0,
    k_min=torch.tensor([[[-10.]], [[1.]]]),
    k_max=torch.tensor([[[10.]], [[1.]]]),
    k_avg=torch.tensor([[[0.]], [[1.]]]), idx=Scores(),
    far_use_mm=True, near_use_mm=False)
for name, src in [("graph",a), ("production_dispatch_copy",b)]:
    f = next(n for n in ast.parse(src).body
             if isinstance(n,ast.FunctionDef) and n.name=="l1_loop")
    env = dict(ctx)
    exec(compile(ast.Module(body=[f],type_ignores=[]),name,"exec"),env)
    out = env["l1_loop"]()[1]
    print(name, out.tolist(), out.argmax(-1).tolist())
# 执行实际硬门的赋值与打印节点，注入一行错误结果。
nodes = ast.parse(a).body
i = next(i for i,n in enumerate(nodes) if isinstance(n,ast.Assign)
         and any(isinstance(t,ast.Name) and t.id=="diff" for t in n.targets))
env = dict(torch=types.SimpleNamespace(equal=lambda x,y:x==y),
           masks_static=[0], masks_eager=[1], N=1)
exec(compile(ast.Module(body=nodes[i:i+2],type_ignores=[]),"gate","exec"),env)
print("continued_after_mismatch",env["diff"]==1)
```

原始短输出：

```text
graph [[[10.0, 1.0]]] [[0]]
production_dispatch_copy [[[0.0, 1.0]]] [[1]]
bitexact same-input replay vs eager: diff_rows = 1/1
continued_after_mismatch True
```

固定复现输入为上述字面量（无随机输入）。源码 SHA256：graph=`c7d964ae94098999537197ce337b115b55657be08af45fd23f4ad5fa1b7e8126`；subseg=`16736cff0682e4b0a81173421b0b8048a63fb7e3ee7151901166fd825554cfc6`。这是源码函数抽取验证，不是完整 GPU 程序跑通，不证明实际1008行 mask 有多少不同。

## 性能归因也须同步更正

新 verdict 的 `reading.direction_reversal` / `verdict` 将 B 段216.5ms称为“70%都是dispatch”，但记录 graph B 仍158.7ms。按记录算，相对减少仅 `(216.5-158.7)/216.5=26.70%`；即使不考虑上述算法不一致，也不支持整个 B 段都是 CPU 开销。此差值也不能精确分离 CPU 成本，跨运行噪声、分配和图执行方式均可能参与。32K→64K两点增长约2.2x不足以证伪某类GPU工作占主导；需要实际时间线/同口径测量。

`158.7-47.2=111.5ms` 是两个不同 capture 的残余差，包含 topk、mask、不同输出生命周期/内存布局等，不能直接认定“纯 topk kernel 111.5ms”，更不能据此断言批量化不可消。保留它作定位线索，下一步用同一实际路由的 trace/阶段计时验证。CUDA Graph 值得继续测试，但不应据此排除批量化、紧凑候选等路线。

## 有界修复与重测顺序

1. 在实验侧修三处 L1：读取 `index_dict.get("k_avg")`，复用生产的 `far_use_mm/near_use_mm`；第二 seed 同样处理；L1 graph 显式传入静态 `k_avg_s` 并在输入更新时 copy，不能 capture 旧平均统计后只替换 min/max。优先抽取共同 L1 函数，避免再次复制漂移。
2. 增加真正 fail-closed 的门禁，测试单值翻转、列表截短、dtype/shape错配，普通和 `-O` 都必须失败；通过才允许输出性能结论。
3. 对照真实 `batched_group_sels`，而不只 PoC 自身 eager：比较 L1分数/候选/L2 mask；覆盖同/新seed、near avg与minmax、GQA、短前缀、非整块 fallback、ties/零概率补位。先固定生产支持域，再扩配置。
4. 重测同机器同输入的 eager与graph，记录输入hash、设备/驱动/torch/Triton版本、源码hash、raw repeats、capture和首次构建成本、replay+所有输入注入、峰值/缓存常驻显存。先明确固定shape复用次数，再讨论跨prompt缓存，不能只报告热replay。
5. 通过后再测完整 selector+attention、dense强基线和TTFT；保持chunk2048与预算/数值契约。当前不将未经GPU验证的实现移入main。

本轮仅发布新证据与明确修复范围；实验分支不是写入目标，没有修改其中脚本。main既有153/155修复不受本发现影响，156速度方向修正保持有效。待验证readback补丁仍保留原位，不重复无收益CPU测速。
