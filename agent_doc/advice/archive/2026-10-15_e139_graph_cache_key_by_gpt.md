# E139 Graph 集成：缓存键缺失被 capture 固定的配置

日期：2026-10-15。main `7393ed620054fa03b443799f77189ab962d8a324`；实验分支 `e138-ablation` `30edd7c08cf6d199fa8f01a3bc1a7230a62bc453`，新增集成来自 `9c85a6a`。本次范围是新增 `batch_select.py`、验收脚本与性能判决；未修改实验分支，未合并实现到 main。

## TL-E139-GRAPH-CACHE-20261015

状态：**confirmed（缓存键/闭包复用逻辑）；真实 CUDA replay 差分待验证**。严重度：P1，条件为同进程开启 `TLI_PREFILL_SELECT_GRAPH=1` 后使用相同长度/分组/K2、但不同 α、γ 或 sink/SWA 等配置。默认 GRAPH=0 不受此缓存问题影响。

位置：`sparse_attn/ops/batch_select.py:68–108,276–310,444–455`。缓存键只有：

```python
(L, N, kf, kn, K2arg, tuple(group_ends), far_use_mm, near_use_mm)
```

但 `_body` 捕获并在 graph 中固定的内容还包括 `nb_l/sl_l`、`m_far/m_near`、`k2_far_l/k2_near_l`、sink/SWA、head/GQA 数量、dtype/device 等。命中时 `_graph_cached_run` 只更新五个输入张量并 replay 旧图，不使用新 `_body`。保留首个闭包引用确实解决了旧几何张量被回收的问题，却不能保证新配置的几何与它相同。

例如 α 从0.25改为0.5：同一组末2048，sink=SWA=64，near 起始块从23变成15，缓存键仍相等。新层虽然算出了新几何，却在命中后把它丢掉。输入 shape 不变，因此不能期待 copy_ 报错触发 eager fallback。γ 改变预算、sink/SWA改变保护范围也是同类风险。`_gate_ok` 允许这些参数邻域，未固定为首次 capture 的值。

另有 device/dtype 缺失风险：`copy_` 可以转换 dtype/跨设备复制，不保证报错；旧图的输出 dtype/device 不会因更新输入而改变。多流共享静态输出则需要独立生命周期协议，不能只以配置键修复竞态。本轮未运行这些 GPU 情形，不将它们标为已实测错误。

## 实际验证与复现

环境：Python3.12.14、torch2.8.0+cpu，Linux x86_64；未调用GPU、无CUDA/Triton实测。使用固定实验源码的真实 `batched_group_sels` AST，CPU独立打分替身，模拟“保留首次捕获闭包”的语义。**这不是 CUDA Graph 模拟器或生产GPU验收**；用于证明遗漏配置能改变实际选择体的结果，且键无法区分它们。

```python
import ast, subprocess, torch, types, os
import torch.nn.functional as F
rev = "30edd7c08cf6d199fa8f01a3bc1a7230a62bc453"
src = subprocess.check_output(
    ["git","show",rev+":sparse_attn/ops/batch_select.py"]).decode()
f = next(n for n in ast.parse(src).body
         if isinstance(n,ast.FunctionDef) and n.name=="batched_group_sels")
cache, keys = {}, []
def frozen(key, body, inputs):
    keys.append(key)
    return cache.setdefault(key,body)(**inputs)
env = dict(torch=torch,F=F,os=os,_gate_ok=lambda *x:True,
           _graph_cached_run=frozen)  # 显式绕过CUDA门，仅测试CPU选择体
exec(compile(ast.Module(body=[f],type_ignores=[]),"selector","exec"),env)
class Index:
    def __init__(self, alpha):
        self.args=types.SimpleNamespace(tia_block_size=64,
            tia_level1_topk=8,tia_level2_topk=256)
        self.alpha=alpha; self.beta=.25; self.gamma=.625
        self.sink_tokens=64; self.sliding_window_size=64
        self._sink_pool_lo_blk=1
        self.far_method="minmax"; self.near_method="avg"
    def _subspace_indices(self,dev):
        return torch.arange(32,device=dev)
    def prepare_index(self,k,cu):
        x=k[...,:32].contiguous(); p=x.reshape(1,32,64,1,32)
        return dict(k_min=p.amin(2),k_max=p.amax(2),
                    k_avg=p.mean(2),k_qat=x)
    def _fast_score_fine(self,k,q,dtype):
        return torch.einsum("thd,hgd->hgt",k,q.reshape(1,2,32)).reshape(2,-1).to(dtype)
    def _fast_score_coarse(self,lo,hi,q,dtype):
        return (self._fast_score_fine(hi,q.clamp(min=0),dtype)
                +self._fast_score_fine(lo,q.clamp(max=0),dtype))
g=torch.Generator().manual_seed(157)
q=torch.randn(1,2048,2,128,generator=g)
k=torch.randn(1,2048,1,128,generator=g)
run=lambda a:env["batched_group_sels"](Index(a),q,k,128**-.5,[2048],1024)
os.environ["TLI_PREFILL_SELECT_GRAPH"]="1"
a=run(.25); stale=run(.5)
os.environ["TLI_PREFILL_SELECT_GRAPH"]="0"
fresh=run(.5)
print(keys[0]==keys[1],torch.equal(stale[0],a[0]))
print(int((stale[0]!=fresh[0]).sum()),int(stale[0].sum()),int(fresh[0].sum()))
```

实际输出：`True True`；`36 256 256`。预算总数相同，但36个布尔位置不同，不能用“都选256个”作为正确性证明。

另从真实源码抽取 `_graph_cached_run`，预填一个缓存项（mock graph 仅记录 replay；静态输入为CPU tensor），传入会抛异常的新 body：实际返回旧输出，更新静态输入1→2，调用 replay 一次且未调用新 body。该测试验证真实命中控制流，不执行CUDA。源码SHA256：`753c7bb1d5eb864a2cd43120e34906188c77c8afe79b17d61315c76ef12aeed4`；输入 `q.numpy().tobytes()+k.numpy().tobytes()` SHA256：`b25f9f3204e41d6fa854a6fffdc56b0ea2e96392a458fac4521bcc23db15f902`。GPU上具体差异数仍未知。

## 修复方案：保留跨层复用，不退回实例ID

1. 从 capture 实际依赖构建完整 signature。复用已有host列表（避免新增 `.item()` 同步）：`tuple(nb_l), tuple(sl_l), tuple(k2_far_l), tuple(k2_near_l)`，加bs/sink/SWA、H/Hkv/G、组边界、各输入是否None及shape/dtype/device、有效打分后端/实现版本。或纳入所有决定上述量的原始配置并证明完备。几何相同的跨层新实例仍可共享，无需简单加入 `id(indexer)` 牺牲复用。
2. 命中时再校验静态输入schema，不能让 `copy_` 的隐式转换替代校验。不支持的组合明确回退；异常不应被测试误记成graph成功。
3. 单stream使用明确作为支持边界；不同stream/并发请求需分池或ready/free事件、输出消费完成保护。仅将stream放进key不能替代完整并发/释放协议。此项未实测，不直接合入高风险异步改动。
4. 必测同一进程配置A→B→A（α/γ/sink/SWA），每项与清缓存的eager逐行mask比较；dtype、device、head配置及不同实例分开测试。加入graph命中/miss/fallback可观察计数，确认真正执行replay。
5. 继续验证冷capture+首请求、热replay、8项容量溢出后的重新capture，以及峰值/常驻显存。当前 `e139_gang4_wall_lsweep_graph.py` 明确在warmup排除capture，`fresh_idx()`并不清除模块全局缓存。已有约1.21x应称“同shape缓存命中后的单层稀疏prefill wall收益”，不是完整TTFT；生产首次请求与不同长度分布收益仍未验证。无需否定有效热态数据，但不能外推。

## 上次问题与本次数据的边界

集成后的 `_body` 保留了正确near avg派发，并显式注入k_avg；因此不能把旧PoC的near错误直接判为集成版仍有。旧PoC脚本本身未改，旧7393ed6发现仍保持原状态。新增验收脚本仍只打印 `diff_rows/allclose`，未强制失败退出，而且可被内部fallback掩盖；属于上一条门禁问题的延续，不新增重复ID。集成版同一输入两次eager只能证明重复性，不能单独证明重构相对旧版零扰动。

Step3保留tile表、未将1.8%收益强行并入生产是合理决策；但不据此确认“唯一瓶颈纯topk111.5ms”。该数仍来自旧PoC不同capture相减，既有归因限制未解除。本轮不改任何原始日志，也不声称推翻固定配置的已有GPU输出结果。

本次只发布新增缓存缺陷证据及修复验收要求，实验实现没有移入main。未做GPU编译、memcheck、racecheck、真实模型质量或端到端测速；GPU/驱动/编译配置未独立核验。下一检查点：实验缓存signature、fail-closed测试、配置切换GPU差分和冷/热成本；main既有153/155修复与待验证CPU同步规避候选保持不变。
