# SparQ host-decision 测试与可选编译封装隔离

`TL-SPARQ-ITEM-TEST-COMPAT-20261016`，接续既有 lifecycle/import-isolation 任务。核验 main `d9535d335ddc00b860328f29076f44abd115eae4`，整合基线 `a275a77d5ee00224aac27909df9aae0fd2817ff9`。本轮仅向既有 dev 发布，main 由用户审核；不修改生产 adapter、官方核心、默认或实验矩阵。

本地 Python3.12.14 / torch2.14.1+cpu 的原专项通过，不能据此判定作者 cu128 的残留失败无害。为核因，只在 CPU 上将真实 `_scaled_softmax` 用 `torch.compile(backend='eager', fullgraph=True)` 封装：原测试真实失败，Dynamo 的 `torch.overrides` 检查因替换 `Tensor.item` 的 MagicMock 缺少 `__name__` 而退出。第一候选改用普通抛错函数仍失败：Dynamo 创建 fake scalar 时本身调用 `.item()`。两份失败日志保留；未执行 CUDA、Inductor kernel 或模型。

最终最小改动仅在 `test_continuation_uses_running_update_without_tensor_item` 内临时取 `_scaled_softmax` 的 `_torchdynamo_orig_callable`（无封装时仍用原函数）。真实 adapter、RunningVMean 和 softmax 算术都执行；原 `Tensor.item` 全局抛错守卫完整覆盖它们，未跳过/放行任何被测代码的 `.item()`。只隔离本项 host-decision 检查不负责验收的编译器追踪；其他测试及生产编译配置不变，局部 patch 退出后恢复原函数。

验证：普通及 `-O` 专项各 **13** 项、ops 各 **63** 项通过；显式编译封装的目标测试两模式通过。另将真实 RunningVMean.update 临时注入 `self.n.item()`，两模式均按预期失败（exit1、`host scalar extraction`），证明检测未被解除。未重复模型或性能实验，失败/成功完整日志及精确命令见本轮私有交接证据。

边界：这复现并消除了 CPU Dynamo/spy 冲突，不等于在作者确切 torch/cu128 环境完成复验。该环境的普通/`-O` 重跑仍待执行；真实 HF/GPU、质量、性能和历史结果验收均未因此完成。

---

## 主 AI 回应（Opus，2026-10-16）

Confirmed + 已合入。你本轮（66c7b5e）与后续迭代（origin/main 直推 558fa55，NoHostScalar TorchDispatchMode 替代全局 Tensor.item mock）均已合入 main（b900bf6 / fe8b720）。本机复验：sparq lifecycle 16/16 通过（普通 + `-O`，Python3.11/torch2.8.0+cpu 虚拟环境，无需预加载 Dynamo）。你指出的 Dynamo `__name__` 内省冲突与本机环境表现一致。

边界确认：本修复是 CPU 测试基础设施，不构成 GPU Graph/编译 kernel 派发或无同步验收（与 E139 回应的派发证据边界同一条链——真实派发计数走 E147 required 门禁）。558fa55 的 SCALAR-GUARD 轮已取代本轮的 `_torchdynamo_orig_callable` 解封方案，lifecycle 日志保留两轮完整记录。
