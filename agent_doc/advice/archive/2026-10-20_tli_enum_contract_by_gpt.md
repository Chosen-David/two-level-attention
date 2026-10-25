# TL-ENUM-CONTRACT-20261020：优化解释器不得取消选择配置校验

基线/main `bb173c7300a01cbe720adbc705795a578da5ae75`，实验分支
`e138-ablation=2ed6a1feb78a7b5bcf3cac922b4510c3968714a5`；仅核验分支身份，
未合并其代码或借用结果。新独立检出和本机审计锁；保留无关临时字体文件，
不改guide或其他仓库。

## 触发、根因、修复

直接Python构造TLIIndexer时，`tli_subspace`、`tli_l2_subspace`、
`tli_near_select`、`tli_sim_dims` 四个枚举只靠assert校验
（`sparse_attn/indexer/tli_indexer.py:93,104,155,159`）。`python -O`
会删除assert，因此拼错/空值可以构造成功。固定nb=2/nt=32/nl=512和
legacy模式均可复现。CLI本身有choices，不声称合法CLI输入会触发。

静态消费链：patch构造indexer；prepare_index/compute_score使用子空间，
prepare_mask/compute_mask消费选择模式；`_subspace_indices:700–722`
的最后else及`_sim_dims_indices:1386–1395`的默认尾维分支可接收未校验值，
不是报错出口。L2非full值会使用same路径；near非cluster/sim_greedy会落普通
路径。此为源码路径判断，未运行张量或测得质量损失。

仅把上述四个assert替换为显式if/ValueError，成员集合和字段名不变。
非法输入普通模式从AssertionError改为ValueError；优化模式从静默接受改为拒绝。
合法配置、固定配额、sink/SWA、D′、预算、topk ties、GQA、chunk2048、
kernel、缓存和stream均不变。配置期常数次检查，不宣称性能优化。

## 验证证据

`tests/test_tli_enum_contract.py`复用完整三层生产构造体AST加载器；metrics
工厂替身，任何Torch访问硬失败。不是完整依赖导入，也不是GPU等价测试。
2模式×4字段×4非法值=32个子例；所有合法枚举逐项保留。

```text
python -O -m unittest discover -s tests -p test_tli_enum_contract.py # 修前
Ran 2 tests; FAILED (failures=32)
python [-O] -m unittest discover -s tests -p test_tli_enum_contract.py # 修后
Ran 2 tests; OK
python [-O] -m unittest discover -s tests -p 'test_explicit*.py'
Ran 10 tests; OK
python [-O] -m unittest discover -s tests -p test_partition_ratio_contract.py
Ran 3 tests; OK
python [-O] sparse_attn/ops/tests/test_batch_explicit_quota_guard.py
Ran 6 tests; OK
python -O agent_doc/results/tli-enum-contract-20261020/verify.py
identical_controls=336; accepted_before_rejected_after=32
git diff --check # 无输出
```

普通/-O各21测试通过。差分从git固定基线加载构造体，枚举四参数完整笛卡尔积
及两种配额模式，共336个对象全部属性相等；32反例修前接受、修后拒绝。
配置矩阵hash `eb16b6c0bda7ca9078a96d5b7dc27341231b4a7253ee947c9ce3c05d77426d8d`；
候选源码hash `c76897ee5596ac28385970aa73abd72ae359f3fd69d76371469182b0e78700a3`。
以上均为本执行者验证，不称独立代理复核。

环境Python3.12.14，Linux x86_64；未安装Torch/Triton，nvidia-smi不可用。
额外尝试E151c三测试模块因缺Torch全部导入错误；未运行其-O、完整依赖、GPU、
模型、性能、显存、kernel数或host同步测量。驱动/架构/JIT配置无本轮验证值。
测试失败缺口不以host测试替代。此补丁只修复可在host验证的构造期校验。

## 历史影响与下一步

只需排查直接Python调用、优化解释器、四字段非法输入同时成立的历史任务；
未证明任何既有实验实际触发，原始数据不改。非法配置纠正后必须重测，
合法CLI实验不因本修复自动失效。

仍未解决：固定nl<bs的旧整页保底、显式配额批量选择、全前缀L2和索引打包
浪费、跨tile overlap收益。没有GPU证据不晋升这些候选。本次按
research-implement-optimize技能执行修前复现→最小host修复→合同差分，
不以本项输入修复冒充算法或性能创新。
