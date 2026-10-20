# TL-EXPLICIT-FREE-PROTECTED-20261019：nt=off 不能绕过保护区预算

## 问题与契约

基线 `836281d3edb46cf0f36807eb91c042ddb2ca6e2b`。E155d 显式几何的
`nt=off` 归一为 `gamma=None`，只是取消 L2 near/far 分额，不取消 K2
总预算或 sink/SWA 必选语义。只读人类 guide `GUIDE.md:155-168,205-217`
及既有 gamma-off advice 均明确保护区计入总预算。

`sparse_attn/indexer/tli_indexer.py:435-450` 原校验把剩余预算非负检查
放在 `if not _nt_off` 下，故 off 模式允许 `K2 < sink+SWA`。
同文件基线 `:2439,2509-2516` 的消费端将负 mid 预算截到 0，再强制
保留两个保护区；截零不能使保护 token 消失。

例如 bs=64、K1=4、nb=2、nl=512、nt=off、sink=SWA=64、K2=127、
S=4096：构造成功，但仅必选区 `[0,64)` 与 `[4032,4096)` 就有128个
互异 token，大于127。此计数是契约/源码推导，不是真实张量选集测量。
短输入可能让保护区重叠，不消除同一配置在后续长输入上的必然矛盾。
既有数值 nt 模式本来就构造期拒绝此配置。

静态入口：`patches/patch.py` 由 `indexer_type_dict['tli']` 实例化；
`ops/reference_prefill.py:200`、`ops/eager_prefill.py:462` 及
`patches/qwen3_attn_patch.py:275` / `llama3_attn_patch.py:105`
调用 `Indexer.prepare_mask`，继而 `compute_mask`。未运行实模派发。

## 修复与不变量

只在显式几何 `_geo` 的构造校验中增加 `K2 >= sink+SWA`，位置
`tli_indexer.py:435-440`；保留原数值 nt 的额外 near 配额检查。
新拒绝域仅为此前接受的 off 模式保护预算不足配置。
`K2 == sink+SWA`（mid预算0）合法；子块sink和零sink合法。
不改变合法配置的实例状态、selector、topk ties、L2归一化、GQA、
保护区、chunk2048、tile、缓存、量化或任何kernel。
legacy 无显式几何配置保持原行为；本修复不宣称解决所有模式的预算校验。

候选取舍：截断保护区违背必选合同；提高K2是扩大预算；延迟到每次选择再
拒绝会重复检查并延迟暴露无效配置。采用现有构造期门禁，常数次host运算，
不是性能优化。依研究实现优化技能先红后绿、独立差分，未引入新依赖。

## 实测证据

`tests/test_explicit_free_budget.py` 使用既有完整构造AST加载器：
Indexer→TIAIndexer→TLIIndexer，仅metrics工厂替身，Torch任意访问失败。
不是完整模块导入、真实selector、张量或GPU验证。
新增3个测试方法，覆盖18个非法off、27个合法off及9个数值模式子用例。
block32/64/128，sink0/4/128，预算不足1/整块、恰好及有剩余。

```text
python tests/test_explicit_free_budget.py           # 修前
Ran 3 tests; FAILED (failures=18)
python -O tests/test_explicit_free_budget.py        # 修前
Ran 3 tests; FAILED (failures=18)

python tests/test_explicit_free_budget.py           # 修后3项 OK
python tests/test_explicit_near_page_budget.py      # 修后4项 OK
python sparse_attn/ops/tests/test_batch_explicit_quota_guard.py # 6项 OK
# 上面三条均另以 python -O 执行，各自相同通过数。
git diff --check                                  # 通过
```

独立只读复核额外比较基线/候选1080个host构造场景：144个只有预期的
非法off从接受变为拒绝，612个接受实例字典完全相同，324个两侧均拒绝；
没有发现合法构造变化。独立专项/相邻普通及-O各7项亦通过。
这些计数不能替代prefill→decode选集/输出的设备回归。

测试输入矩阵SHA256（测试文件）：
`c05f23d64a6d4e70e19ce2ff4d0a28694eacc2562cc2af58a98831e000945e6c`。
无张量输入，不伪造张量hash。Linux x86_64，容器AMD EPYC 9V74、9逻辑CPU，
Python3.12.14 / Clang22.1.3。Torch未安装，nvidia-smi不可用；GPU架构、
驱动、CUDA/Triton/kernel配置无本轮验证值。速度、峰值内存、kernel数、
host同步数、模型质量和TTFT均未测量，无提升数字。

## 历史影响与待验

只应筛查显式nt=off且K2低于保护区和的历史配置；本轮未认定任何已报告
实验实际触发，不修改旧数据。该非法配置应纠正后重跑，不能据本补丁为
旧结果重新背书。合法配置仍须按既有GPU门禁验证，不扩展CPU结果的范围。
待全依赖导入、真实选择/输出、短长输入与prefill到decode、模型质量/性能验证。
实验分支e138-ablation最新核验为`2ed6a1feb78a7b5bcf3cac922b4510c3968714a5`，
仅用于身份对照，未合并或借用其结果。映射sglang人类文件本机不可用，
以当前用户授权及本仓只读指南约束；未访问其他机器或新增资源。
