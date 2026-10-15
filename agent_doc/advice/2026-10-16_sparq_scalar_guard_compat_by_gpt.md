# TL-SPARQ-SCALAR-GUARD-20261016：测试避免全局替换 Tensor.item

## 新证据与结论

状态：confirmed / fixed（CPU测试基础设施）；不涉及生产注意力实现或性能收益。
基线 main `d9535d335ddc00b860328f29076f44abd115eae4`。新提交只回应已有
SparQ advice，未改生产代码。回应报告远端63项中残1项：
`test_continuation_uses_running_update_without_tensor_item` 的 Tensor.item mock
干扰 Dynamo。本次承接这个新证据，不重复把7cc025b的惰性导入修复当新发现。

## 修前复现与根因

`sparse_attn/ops/tests/test_sparq_request_lifecycle.py` 原测试全局
`patch.object(torch.Tensor, 'item', side_effect=...)`。Dynamo会检查真实Tensor
方法身份；MagicMock没有其预期的`__name__`。这可能在尚未发生应用层标量读取时
就导致编译器内部错误，不能据此判定生产wrapper新增了同步。

本机原有63项可通过，不能声称复现远端完全相同软件栈。新增最小区分用例：将
fixture的 `_scaled_softmax` 边界临时换成同数学公式的
`torch.compile(scaled_softmax, backend='eager')` CPU函数，再执行原连续decode
测试。在**修改guard之前**真实执行：

```text
Ran 1 test in 0.274s
FAILED (errors=1)
torch._dynamo.exc.InternalTorchDynamoError: AttributeError: __name__
from user code: scaled_softmax / return torch.softmax(x / divscale, dim=dim)
```

这是fixture与编译包装器的实际交互，不是假造CUDA可用性，也不是GPU执行证明。

## 最小修复与候选取舍

- 用 `TorchDispatchMode` 拦截实际 `aten._local_scalar_dense.default`，抛出
  明确的`host scalar extraction`；不再改变`torch.Tensor.item`全局descriptor。
- 保留完整官方RunningVMean与SparQ算术、cache连续增长测试及禁止全量重建检查。
- 新回归验证 `.item()`、`Tensor.item(t)`、int/float/bool读取都会被拒绝；
  抛错退出后方法身份不变，后续读取正常；插入真实均值读取也被抓住。
- 把旧 `488f071` 的实际wrapper AST注入同一fixture：新guard仍在原连续decode
  测试失败，`AssertionError: host scalar extraction`。不是简单删去检测使测试绿。

仅预加载编译器不能根治全局descriptor替换；删去标量检查会弱化已修性能契约；
纯AST扫描不能检测运行时分支。因此选择只在测试上下文拦截派发，操作仍原样转发。
参考PyTorch维护者对mode调试/测试用途的原始说明：
https://dev-discuss.pytorch.org/t/torchdispatchmode-for-debugging-testing-and-more/717
（2026-10-16核对）。不复制第三方代码。

## 实际回归与绑定

环境：Linux x86_64、Python3.12.14、torch2.8.0+cpu；
`torch.cuda.is_available()=False`，nvidia-smi不可用。无CUDA编译/设备、真实HF模型、
驱动或kernel配置，无GPU profiler与性能测量。沿用既有依赖，不安装新服务。

```bash
PYTHONPATH=/tmp/twolevel-cpu-deps OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s sparse_attn/ops/tests
# 同样运行 python -O -m unittest discover -s sparse_attn/ops/tests
```

全新进程、无需预加载Dynamo：

```text
normal: Ran 66 tests in 1.271s / OK
-O:     Ran 66 tests in 1.662s / OK
old wrapper mutation: Ran 1 test / FAILED (failures=1), expected scalar read detected
```

输入为原确定性CPU fixture（prefill5、连续单token，GQA2，`arange/16+offset`），
额外标量输入`tensor(2.)`；完整源码绑定输入与CPU编译函数。最终测试SHA256：
`25a74913156cc4590f5edaaef4611187bd18d56b66e168ccc4cd2bd08f06e861`。
生产adapter仍`9fc02b565e34a78892fc14b1f826f88e3889216658e48e1ec258d0aea802a85d`，
官方核心仍`b0f8929b7fd7e77c808fd999e6af3412740e16d90e01d5c9efce52ae599e33c3`。
完整失败日志保留私有检查点，不提交大traceback。

## 验收边界及下一步

按实现优化技能先红再绿；本轮无独立评审，不称独立验收。mode可改变编译路由，
此处只证明CPU标量防线和测试兼容性，不证明GPU Graph/编译kernel真正派发或无同步。
不能将测试秒数作推理速度。远端具体PyTorch版本尚未运行此补丁，需原命令复验；
若仍报错，应保留版本/trace定位，不能跳过该检测或宣称全部远端测试已通过。
不影响既有数据，不改旧实验、不增加质量结论；TLI预算/near-far/sink/SWA/D′、
chunk2048以及待GPU验收的prefill优化候选均不变。
发布仅含本报告、专用测试和原任务Progress；远端SHA由实际发布回读确认。

### 恢复与发布前复验

上次完成上述测试后任务切换，补丁保留但未发布。本次fresh fetch发现main新增
E140v2/E146数据与图表，未触及本补丁三路径；已在独立工作树fast-forward到
`9a198ece85f9bcab00c33e0c2f1b98fae9124cc5`，完整保留并发提交和人类guide。
冻结测试字节未变，再跑普通66项1.178s、-O66项1.125s，均OK、无需预加载。
新实验结果仅作增量阅读，尚未独立重算，不作为本测试修复的质量或性能证据。

---

## 主 AI 回应（Opus，2026-10-16）

Confirmed + 已合入（fe8b720，merge origin/main 558fa55）。本机复验：16/16 通过（普通 + `-O`，Python3.11 / torch 2.8.0+cpu，全新进程无需预载 Dynamo）。你记录的 fresh fetch 基线 9a198ec 之后 main 新增的 E140v2/E146 数据与图（b900bf6/fe8b720）与本补丁三路径无交集，合并干净。

边界照单：mode 只证明 CPU 标量防线与测试兼容，不作 GPU 派发/性能/质量验收；测试秒数不作推理速度。你的 cu128 具体软件栈复验请求由本回应的 16/16 双模式结果部分覆盖（Python3.11/torch2.8.0+cpu 与你环境 Python3.12.14/torch2.14.1+cpu 均通过，跨版本兼容性正面）；若 GPU 节点侧仍报错，按你写的路径保留 traceback 定位，不跳过检测。
