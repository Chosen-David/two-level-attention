# TL-EXPLICIT-L1-BUDGET-20261019：near 显式页配额不能超过 K1

## 触发与根因

基线 `6f006b1527c122ca942db4860a81047d027b8715`。
`sparse_attn/indexer/tli_indexer.py:418-441` 的 E155d 构造校验允许
`tli_near_pages > tia_level1_topk`，只检查非负、区域容量及 L2 配额。
但同文件 `:369-375` 和只读人类 guide `:166-168,205-217` 的契约是
near/far 竞争页共用 K1。消费端 `:2151-2158,2224-2234` 保留 near 配额，
把负 far 配额截成 0，并未限制 near，因而总竞争页可以大于 K1。

对齐反例：K1=4、near_pages=5、block=64、near_len=320、near_tokens=16、
K2=1024、sink=SWA=128、S=4096。构造检查原先全通过；near 池
`[3648,3968)` 有5个整页，far 名额0，L1竞争页5 > K1=4。
保护区均对齐，无边界保送页干扰。`nt=off` 只取消 L2 分区，同样有此问题。
这是源码几何推导，不是实际 GPU 选集测量，也不表示最终 L2 token 数超 K2。

静态调用链：patch 构造 TLIIndexer；reference prefill `reference_prefill.py:200`
及 candidate 逐组 `eager_prefill.py:462` 经 `prepare_mask` 到 `compute_mask`；
decode 同样消费该 indexer。上一修复使显式配额不能走 batch 旧公式，但不能
替代构造期总预算校验。实际设备派发、本轮模型调用未验证。

## 有界修复与取舍

只在 `tli_indexer.py:424-429` 增加 `nb > K1` 的显式 ValueError，数值 nt
与 off 一视同仁；不静默截断用户配额，不修改任何合法配置的 selector/kernel。
nb=0、nb=K1、内部值及 legacy 无几何参数配置保留。边界保送仍按既有合同
不占竞争页预算，不能把含保送页的全部候选强行截成 K1。
未改 chunk2048、选择粒度、mask、GQA、量化、D′、重叠调度或缓存。

独立只读复核确认契约、上面的5页反例及数值/off双模式，支持构造期拒绝；
没有把静态复核计入 GPU 通过。

## 修前失败与修后验证

新增 `tests/test_explicit_near_page_budget.py`，用标准库原样编译并执行
Indexer→TIAIndexer→TLIIndexer 的完整构造函数 AST。仅 metrics 工厂替身，
Torch 任意属性访问硬失败；关闭外部层掩码读取、oracle表和投影加载。
不是完整模块导入测试，不是张量运算或 GPU 验证。

覆盖 K1=0/1/4/128，超额1/3页，nt=0/32/off；合法几何覆盖block32/64/128、
nb零/内部/上界；另查负数、容量不足、保护区预算、缺参、比例/几何互斥，
以及 legacy 缺省/数值/off 归一。

```text
python tests/test_explicit_near_page_budget.py        # 修前
Ran 4 tests
FAILED (failures=24)  # 24个超额子用例均未抛错，其余用例通过

python tests/test_explicit_near_page_budget.py        # 修后
Ran 4 tests
OK
python -O tests/test_explicit_near_page_budget.py
Ran 4 tests
OK
python sparse_attn/ops/tests/test_batch_explicit_quota_guard.py
Ran 6 tests
OK
python -O sparse_attn/ops/tests/test_batch_explicit_quota_guard.py
Ran 6 tests
OK
git diff --check                                    # 通过，无输出
```

完整输入矩阵位于新增测试；该文件SHA256
`36415e8bada18e7e77195b925f4a8ed68be4525f0824c8f4bbd5c46dcae8632e`。
无张量输入，故不提供虚构张量hash。环境：Linux x86_64，容器可见
AMD EPYC 9V74、9个逻辑CPU；Python3.12.14（Clang22.1.3）。Torch未安装，
nvidia-smi不可用；GPU型号/驱动/CUDA/Triton/JIT/kernel配置无本轮验证值。

## 历史影响与剩余工作

仅 `near_pages>K1` 的显式几何实验可能以更大 L1 候选预算执行，需要核实
原配置后在纠正预算下重测；不能将其质量/速度与同 K1 合法配置等价比较。
未认定任何具体历史实验已触发，未修改历史数据。合法配置不因本问题自动失效。
CPU host 契约验收完成；全依赖导入、真实选择/输出、prefill→decode、模型质量、
性能/峰值显存/kernel数/host同步均未实测，无提速或质量改善数值声明。

本轮复用实现优化技能：先契约反例与修前失败，再最小修复/普通及-O回归，
无GPU不晋升kernel候选。没有为小型host校验更换工具/依赖；成熟基线就是现有
构造与合法配置。已fetch main及实验分支 `e138-ablation`
`2ed6a1feb78a7b5bcf3cac922b4510c3968714a5`；后者仅对照，不合并或借用结果。
