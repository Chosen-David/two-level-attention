# TL-PARTITION-RATIO-20261019：拒绝非法 alpha/beta 比例

## 版本、触发与根因

基线 `36c04140fbcc1d030fb48af9472b9ee56ceecda9`（本轮重新 fetch）。
实际实验分支 `origin/e138-ablation` 为 `2ed6a1feb78a7b5bcf3cac922b4510c3968714a5`，
只核对来源与 main 差异，没有合并或采用其测试结果。

`sparse_attn/arguments.py:286–291` 将 alpha/beta 定义为 near 长度/页预算的比例，
但 CLI 的 float 类型及 `sparse_attn/indexer/tli_indexer.py:355–356`
只做浮点转换，没有有限性和 [0,1] 检查。
`sparse_attn/patches/patch.py` 的 indexer 派发最终进入此构造器。
消费端 `compute_mask` 的分区条件为 alpha>0 且 beta>0；near 区宽按 alpha 推导，
near 页配额按 `max(1, round(K1*beta))` 推导。
负值/NaN 可令分区静默关闭；正无穷可推迟到 int/round 时失败；beta>1 在池容量
足够时可要求超过 K1 的 near 页。上述消费影响为源码分析，不是 GPU 运行结果。

## 修复与兼容性

在浮点归一后增加 5 行构造期守卫，非法比例抛 ValueError，保留参数名诊断。
使用显式条件而非 assert，普通与 `python -O` 行为一致。
合法端点 0/1、中间值、默认 0/0、gamma 数值/off、显式 nb/nt/nl 模式均保留。
不裁剪参数，不改 topk、mask、预算、chunk2048、kernel 或计时路径。
不顺带改变 gamma 合同。候选比较：静默 clamp 会改变实验意图，拒绝；构造期
fail-closed 与此前显式配额守卫一致，采用。没有性能改进主张。

## 实际验证

新测试 `tests/test_partition_ratio_contract.py` 使用现有完整构造器继承链 AST loader，
执行 Indexer/TIAIndexer/TLIIndexer 的完整 __init__；只有 metrics factory 为占位，
任何 Torch 运算都会失败。不是 Tensor 模拟，也不能替代全依赖集成。
输入由测试文件内固定网格生成，其 SHA256：
`f5d4082633d314341b12e9a4b42790d7875f1937219e9222f4be96eae14e72f0`。

修前（普通及 -O 各一次）原始短日志：

```text
Ran 3 tests
FAILED (failures=28)
AssertionError: ValueError not raised
```

修后命令（每条另以 python -O 执行一次）：

```text
python -m unittest discover -s tests -p test_partition_ratio_contract.py
Ran 3 tests / OK
python -m unittest discover -s tests -p 'test_explicit*budget.py'
Ran 7 tests / OK
python -m unittest discover -s sparse_attn/ops/tests -p test_batch_explicit_quota_guard.py
Ran 6 tests / OK
```

普通/-O 各 16 项测试通过。另尝试 `tests/test_e151c_quota*.py`，两模式均为
3 个加载错误：`ModuleNotFoundError: No module named 'torch'`，未计入通过。
`git diff --check` 通过。
环境：本地 x86_64，Python 3.12.14，NumPy 可导入；Torch/Triton 不存在，
nvidia-smi 不在 PATH。未运行 CUDA 编译器/kernel，无设备/驱动性能记录。

独立上下文复核（twolevel_ratio53_review）另执行新测试普通/-O各3项及页预算4项，
全部通过；独立完整构造器差分641例：563合法legacy、22合法显式几何全实例状态
一致，56非法比例旧版接受/新版拒绝。独立loader不提供torch全局，只有metrics
工厂占位。完整生产模块AST移除唯一新守卫后与基线一致。
这只是host合同与源变更范围验证，不是GPU输出等价证明。

## 历史影响与待验

已有合法比例实验不因本修复失效；此前使用非法比例的结果不能继续作为其声明
配置的证据，需要定位具体原始配置后重测，不修改或删除旧数据。
当前未测实际选集/输出、真实模型质量、GPU 正确性、峰值显存、吞吐/时延；
构造期拒绝契约的验收不外推这些项目。
候选内 L2、紧凑绝对索引、fused attention 与 overlap 仍需授权可用 GPU，
保持既有待办，不以 CPU 构造器测试替代。
