# E139 Step2：提交的探针不能复现“补位全过”判决

2026-10-15，main `4a9b29caf3898660280c9428ddf36bf36a101be2`，实验分支 `1ac889799499b27cd0066a43d4f62bc58e630534`。仅新增 `exp/bench/e139_step2_tie_probe.py` 与 `exp/trace/results/e139_step2_topk_cand_feasibility.json`；没有新增生产实现。本轮不修改实验分支，不把其代码移入main。

## TL-E139-TOPK-PROBE-IDENTITY-20261015 / confirmed / P2

结果 JSON 的 `results.probe_bug_note` 明确承认初版 k2<C/负值错误，称已修正重跑；但本次提交的脚本第26–27、42、50行仍是该初版：

```python
vals = (torch.randn(rows, C, device=DEV) * 8).round() / tie_levels
vals = vals.to(torch.bfloat16)
k2 = min(k + C // 2, W)
comp = mask.nonzero().squeeze(-1)[: k2 - C]
```

因此问题不是“作者不知道初版有bug”，而是**修正版脚本、对应运行日志及内容hash未随判决提供，现存脚本与全过结论不匹配**。不能据此断言作者另行运行的GPU结果是假的，也不能将其标为已独立复现。

### 无需GPU即可证明的不匹配

全部内置形状实际都满足 k2<C。候选位置由argsort截取，无重复；`sc`包含全部C候选与非候选负切片，两个集合不交。其集合大小如下，必不等于topk返回的k2个索引：

| W / C / k | 实际k2 | sc集合大小 | 结论 |
|---|---:|---:|---|
| 4096 / 1024 / 256 | 768 | 3840 | 必不相等 |
| 32768 / 8192 / 512 | 4608 | 29184 | 必不相等 |
| 65536 / 8192 / 512 | 4608 | 61952 | 必不相等 |

这是Python切片及集合大小的确定性结论，与CUDA topk实现/tie顺序无关；按提交脚本执行补位项应为每格320/320不匹配，而非JSON写的0/320。

另：randn生成负值，不是报告所写的softmax型非负值；对已round整数再除64/8/2只是二进制幂缩放，在本探针正常数值范围内不构成三个不同tie密度。修正探针时应真正改变离散级数，并单列严格正值与候选零概率。

## 本次执行证据

Python3.12.14，torch2.8.0+cpu，Linux x86_64，OMP_NUM_THREADS=1；无GPU执行。实际抽取固定Git源码的 `probe` 函数，唯一设备适配为其全局DEV设为cpu；W=4096、C=1024、k=256、tie_levels=64、rows=2、reps=1、seed=0，输出 `(2,2,2)`，其中第二项为补位错误行数。CPU的第一项不能代替CUDA tie测试；本文核心结论来自上述设备无关的集合大小矛盾。

```python
import ast, subprocess, torch, hashlib
rev="1ac889799499b27cd0066a43d4f62bc58e630534"
s=subprocess.check_output(["git","show",
    rev+":exp/bench/e139_step2_tie_probe.py"]).decode()
f=next(n for n in ast.parse(s).body
       if isinstance(n,ast.FunctionDef) and n.name=="probe")
e=dict(torch=torch,DEV="cpu")
exec(compile(ast.Module(body=[f],type_ignores=[]),"probe","exec"),e)
torch.manual_seed(0)
print(e["probe"](4096,1024,256,64,rows=2,reps=1))
for W,C,k in [(4096,1024,256),(32768,8192,512),(65536,8192,512)]:
    k2=min(k+C//2,W)
    print(W,C,k,k2,C+len(range(W-C)[:k2-C]))
print(hashlib.sha256(s.encode()).hexdigest())
```

源码hash：`d9e0eaee49ef9a45e8ba3fa555c7be0b5084b85dcb03b65148b89dbe46286cde`。集合大小验证输入即上表整数，不依赖随机tensor；未获取GPU随机输入hash、驱动/架构和原始重跑日志。

## 修复与重测范围

1. 提交实际使用的修正版与原始日志，关联脚本hash、torch/CUDA/驱动/设备、seed、shape、dtype、stride；保留初版为历史失败，不能手改原始结果装成新跑。
2. 补位档使用独立pad参数，显式检验 `0 < C < k2 <= W`（例如 `k2=C+pad`）；正常候选必须严格正值，另设零值/下溢档与全零档。计数非零立即显式抛异常，普通及 `-O` 都不能输出成功结论。新增负值输入仅作为非法生产输入反例，不混入概率等价验收。
3. 对照真实生产L2产生的 `p/ct_j/cand_valid`，覆盖双池边界、候选无效槽、重复位置、GQA和k附近ties；topk最终集合与全宽reference比较。值gather无ULP不自动保证选集相同。
4. PyTorch2.8官方topk文档明确不保证并列元素索引稳定。因而“低index优先”不是API契约；即便某GPU版本/形状探针通过，也须限定派发/版本/shape，并保留全宽fallback。不得擅自引入新的确定性tie规则改变既有选集语义。来源（2026-10-15核验）：https://docs.pytorch.org/docs/2.8/generated/torch.topk.html 。
5. 在上述证据绑定前，Step2维持“待验收候选”，不要以“全格通过、边界分析完备”晋升。若已有修正版只漏提交，补齐并复验即可，不必重跑模型；真实GPU探针仍需在授权可用资源上执行。

历史影响仅限本次Step2可行性判决的可复现性/适用边界，不据此否定此前固定配置Graph热态计时或已有质量数据。缓存键缺陷及旧Graph PoC门禁问题未见新代码修复，保持原状态，不重复新立问题。下一检查点：修正版探针及其输出身份、真正fail-closed门禁；本轮无实现合入、无GPU性能提升声明。
