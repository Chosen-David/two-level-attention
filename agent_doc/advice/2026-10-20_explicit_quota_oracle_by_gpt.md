# TL-EXPLICIT-ORACLE-20261020：固定配额禁止混入 prefill oracle 覆盖

基线 main：`50abf698a16c506f30fe72a3fbb1f0a4df080d15`。
本轮在干净独立检出上取得并发锁并 fetch；main 未变化。只修改本仓库，
不修改人类 guide、实验分支或历史实验数据。沿用实现优化技能的证据边界：
本次是 host 配置契约修复，不是 GPU 优化验收。

## 触发与根因

基线 `sparse_attn/indexer/tli_indexer.py:403–494` 接受固定 nb/nt/nl，
但未禁止环境变量 `TLI_E151C_ORACLE_QUOTA` 加载的逐层 oracle 表。
原 E151c 兼容性检查只要求 alpha/beta>0、numeric gamma 和普通 top-k；
固定配额内部归一化为 alpha=beta=gamma=1，恰好绕过该检查。

静态调用链：`sparse_attn/ops/eager_prefill.py:456–467`、
`reference_prefill.py:196–205` 在选择前启用 `_e151c_on` 并在 finally 清除；
基线 indexer:2035–2045 取层配额，2588–2590 与2623–2625 在簇分/普通
两条 L2 far 路径覆盖刚计算的 `K2_mid-nt`。decode 不启用此覆盖。

最小配置：bs64/K1=4/K2=256/sink64/swa64/nb2/nt32/nl512，另设置 oracle
表 `[32]`。若 far 分支执行，声明的 far budget 应为128−32=96，却会被
替换为32；near 获得剩余预算。此数字是源码算术说明，非实际张量选集实测。
若表恰好等于预算、far 为空或 D′ 跳过 far，可能不改变该次选择，但不能
据此将两套合同混为一种固定配额模式。nt=off 原本已被 E151c 门禁拒绝。

## 最小修复与边界

构造器在几何齐备性检查后（候选429–435行）拒绝固定几何与 oracle 表混用，
错误明确指向 `TLI_E151C_ORACLE_QUOTA`。不改变配额算法、kernel、归一化、
ties、SWA/sink、chunk2048 或 D′。选择 fail-closed 而非静默忽略用户配置。
单独使用 legacy alpha/beta/gamma 的 E151c 实验仍被支持；无 oracle 的固定
配额保持完整构造状态不变。没有新增依赖或性能后端。

## 实际证据

环境：x86_64，Python3.12.14；当前 Python 无 Torch，设备目录无 NVIDIA
节点。未运行 GPU、CUDA/Triton 编译、驱动/架构查询、完整依赖导入或模型。
完整生产构造器 AST（含基类）执行，只替代 metrics factory，任何 Torch
访问会失败。测试和差分核验均由同一执行者完成，不宣称独立代理复核。

原始短日志：`agent_doc/results/explicit-oracle68/before.txt`、`after.txt`。

```text
python -m unittest discover -s tests -p test_explicit_quota_oracle.py
修前：Ran 4 tests; FAILED (failures=9)
python [-O] -m unittest discover -s tests -p 'test_explicit*.py'
修后：普通/-O各17 tests; OK
test_partition_ratio_contract.py / test_tli_enum_contract.py / test_tli_debug_contract.py
普通/-O各3 / 2 / 3 tests; OK（每模式合计25项）
python [-O] tests/check_explicit_quota_oracle.py
各 legal_state_equal=212; baseline_conflicts_accepted=9;
candidate_conflicts_rejected=9
git diff --check：通过
```

差分脚本从冻结基线 git 对象加载旧构造器，对比完整实例状态；覆盖4种
block size、nb边界、nt=0/16/off、量化/边界保送开关，以及独立 oracle 的
不同 gamma、far selector、空环境及 nearest-even 表解析。9个冲突配置
由修前错误接受变为修后明确拒绝。

输入 SHA256：`de61e1761991f99525e821bdae4c32bcf5dc0bad13bda55af4bf53ae5ebb581e`。
候选 indexer SHA256：`b0a85860117b4e51878228b81c800bb5f35f3213cff43e356a5eeee200d5d70b`。
仅构造期增加常量级判断；无速度、显存或质量收益实测，不报告性能提升。

## 历史影响与后续

未证实历史命令实际同时使用固定几何及 oracle 环境变量，不宣布旧分数失效。
只需核查该组合的实验：固定配额应清除 oracle 环境变量；oracle 实验继续
使用其独立 legacy 配置。命中混配的历史结果保留并按正确单一合同重测。

待办仍为 TL-EXPLICIT-NEAR-WIDTH-20261019 的实际张量/GPU验证、候选内L2、
紧凑 indexed attention 的同语义公平测量；本修复不替代这些验收。
发布须再次fetch并校验远端SHA/tree；此有界修复不结束持续任务。
