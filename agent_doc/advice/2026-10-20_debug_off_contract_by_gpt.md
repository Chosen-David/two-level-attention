# TL-DEBUG-OFF-20261020：关闭调试仍触发读回/打印

基线 `0b661c477fdc2fb98aad56b4ee314ca5f312e251`，本轮重新 fetch main 未变。
使用独立 detached worktree、本机审计锁，保留其他执行者的无关字体临时文件。
实验分支 `e138-ablation` 只核验已有引用，不合并、不借用实验结果。

## 问题、路径与契约

`sparse_attn/indexer/tli_indexer.py` 的 L1 调试点原本已将 strip/lower 后
的空串、`0`、`false` 解释为关闭。但六个相邻点只判断环境字符串是否非空：
基线行 581/657/673（静态 pair、动态 D′ 标定/冻结）、2019（MoBA）、
2418（sigma）、2622（普通 L2）。因此 `TLI_DEBUG=0` 会关闭 L1 却开启 L2。

固定配额入口 `__init__` 将 nb/nt/nl 激活为 E64 分区；普通非簇 far L2
分支使用显式 nt 分额，随后到达最后一个调试点。该处额外执行
`(far_p > -inf).sum(...).max().item()` 和 flush 打印；不必开启 sigma/MoBA
也能触发。静态 pair 调试还含 `.tolist()`。这里只确认源码路由和 Python
调试条件，不冒称运行过真实 selector 或测得 CUDA 同步次数。

修复只将六处条件与现有 L1 的布尔约定对齐，保留各自 layer 过滤条件及
原始调试体。空环境维持关闭；`1/true/yes` 等非关闭字符串维持开启；
为兼容原 L1，字符串 `off` 本轮仍开启，关闭请用 `0/false` 或不设置。
每次仍读取环境，未改为构造期缓存，运行中开关变化仍可生效。

候选：复用现有内联条件（采用，最小六处修改）或新增全局 helper（本次不选，
会额外改变现有 AST 审计加载器的依赖）。不是算法/近似/配额优化。
预算、L2 归一化顺序、GQA、ties、保护区、D′ 状态、chunk2048、kernel、
数据和输出身份均未改。整个模块除七个调试条件外的 AST 完全一致
（七个条件中 L1 实际未改），这是源码隔离证据，不是张量等价实测。

## 修前复现和修后验证

新增 `tests/test_tli_debug_contract.py` 从生产 AST 获取七个调试点。
关闭用例直接执行原始 if 及其原始 body，故意不提供调试体需要的 tensor/
scalar 局部变量；错误进入调试体即报错。不是模拟 tensor 运算的数学原型。
8种关闭输入（含未设置）、3个层号，共168次条件/体检查；6种开启输入、
3层号，共126次条件检查。关闭误入修前54、修后0，开启和层号条件不变。

```text
python -m unittest discover -s tests -p test_tli_debug_contract.py # 修改生产前
Ran 3 tests; FAILED (errors=54)
python [-O] -m unittest discover -s tests -p test_tli_debug_contract.py
Ran 3 tests; OK
python [-O] -m unittest discover -s tests -p 'test_explicit*.py'
Ran 10 tests; OK
python [-O] -m unittest discover -s tests -p test_partition_ratio_contract.py
Ran 3 tests; OK
python [-O] -m unittest discover -s tests -p test_tli_enum_contract.py
Ran 2 tests; OK
python [-O] sparse_attn/ops/tests/test_batch_explicit_quota_guard.py
Ran 6 tests; OK
python [-O] agent_doc/results/debug-off-20261020/verify.py
off_guard_evaluations=168; erroneous_debug_entries_before=54; after=0
enabled_guard_controls=126; enabled_controls_unchanged=true
entire_module_ast_except_seven_debug_predicates_unchanged=true
```

普通/-O各24项host测试通过，py_compile及diff检查通过。verify从git固定基线
直接加载源码，raw.json/raw-O.json保存每个输入的七处判定；均由本执行者
执行，不称独立代理验收。完整依赖导入、GPU及全库测试未运行。

输入hash `c83559541fb844947cdfadb76c2e1d8238db2b6b122225fdadc01e78bd4a5bc8`；
候选生产源码hash `6a97bb8568caa00a7932caf01e761f5be3728a11b8236271ad3b2efbb583b823`；
专项测试hash `7d6cbbb733bd432fb23e770eea9d3033b9ba7df0ef9be23ec6bce827d5a7d18a`。
修前原始日志短摘要如上，完整本地日志SHA256
`f4c60fe830b1651e0be7afe4ed2f948878259cb3805790df8ed17ed6630409df`。

环境：Python3.12.14（Clang22.1.3），Linux6.18.55 x86_64，glibc2.39。
Torch/Triton不可导入、nvidia-smi不可用；GPU架构/驱动/JIT无本次验证值。
没有TTFT、wall/device时间、峰值显存或模型质量新数据，不宣称提速百分比。

## 性能依据、历史影响与后续

本轮沿用 research-implement-optimize 的复现/最小修复/证据边界要求。
2026-10-20核查 [PyTorch官方调优指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization)：
CUDA tensor的item属于应避免的不必要CPU/GPU同步来源。这是官方机制说明，
不等于本轮在目标硬件测到了加速。修复移除关闭状态下的误执行，未删开启
状态的诊断功能，也未移出计时区来伪造收益。

历史重测范围仅：环境显式设为0/false或其空白大小写变体，并到达上述六处。
不宣称既有实验实际受影响，不改旧数据。未设置TLI_DEBUG的正常运行不应
获得该修复带来的GPU收益；启用debug的计时也不与关闭计时混比。

待有授权空闲GPU：同SHA前后、同输入hash、固定nb/nt/nl和相同dtype，
比unset/0/1三个环境，核对mask/输出、真实派发和完整选择+attention开销；
分别统计读回/日志、wall/device时间，保留dense强基线及负结果。
固定nl<bs、候选内L2、紧凑打包和overlap仍未验，本轮不晋升其生产补丁。
