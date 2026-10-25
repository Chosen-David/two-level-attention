# TL-EXPLICIT-MODES-20261020：固定配额禁止被其他选择模式旁路

基线 main `927b7279bd66efcf2c90d23b2c5c4bbaf9ddc8e1`。本轮只修改
`Chosen-David/two-level-attention`；实验分支 `e138-ablation` 核验为
`2ed6a1f`，不合并，也不把其性能当 main 结果。

## 触发与根因

`sparse_attn/arguments.py:297–307` 将 nb/nt/nl 声明为固定几何；
`sparse_attn/indexer/tli_indexer.py:403–478` 原构造器却允许与
`tli_sinkswa_only=True`、`tli_moba=True`、`tli_sigma_select=far/near/mid`
同传。compute_mask 的对应分支在固定 L2 配额逻辑前返回（基线约
2067、2383、2387行），分别采用仅保护区、块展开、阈值可变基数合同。
因此声明的固定 token 配额没有被执行；nt=off 也只表示固定总预算内
自由竞争，并不等于 top-sigma 可变预算。

最小配置：bs=64、K1=4、K2=256、sink=swa=64、nb=2、nt=32、nl=512，
再开启上述任一模式。修前完整生产构造器均成功，不能据此接受为固定配额实验。
默认普通固定配额不受影响。未发现证据证明历史跑批实际使用这些冲突组合，
不能将所有既有结果宣布失效。

## 修复与选择

构造器在几何齐备性检查之后拒绝上述组合（候选约429–434行）。复用既有
E151c 的模式兼容性原则，避免静默忽略配置；不擅自决定哪套算法覆盖另一套。
只新增常量级配置判断；无 kernel、预算计算、归一化、ties 或选集修改。
没有新增依赖或后端。合法 legacy 模式及普通 nb/nt/nl 仍可使用。

## 实际验证

host x86_64，Python 3.12.14；Torch/Triton不可导入，PATH无nvidia-smi。
候选tli_indexer.py SHA256：
`deb0e38fc0a171d262c7c4e550d7cf24400a4f5b16c4758faab5aece2671d67f`。
未执行 CUDA/驱动/编译器/kernel 测试，未占用远端资源。
已有测试加载完整生产构造器继承链，仅替代 metrics factory，任何 Torch
调用会失败；这不是完整依赖导入或张量选择测试。

原始日志在 `agent_doc/results/explicit-modes65/`：

```text
python -m unittest discover -s tests -p test_explicit_quota_modes.py
修前：Ran 3 tests; FAILED (failures=15)
python [-O] -m unittest discover -s tests -p 'test_explicit*.py'
修后：各 Ran 13 tests; OK
test_partition_ratio_contract.py / test_tli_enum_contract.py / test_tli_debug_contract.py
普通与-O分别：3 / 2 / 3 tests; OK
python [-O] tests/check_explicit_quota_modes.py
各 legal_state_equal=181; baseline_conflicts_accepted=15;
candidate_conflicts_rejected=15
git diff --check：通过
```

差分脚本直接从冻结基线 git 对象加载构造器，对比完整实例状态，使用显式异常
而非会被-O取消的assert。覆盖bs=16/32/64/128、nb=0/1/2/4、nt=0/16/off、
边界保送和量化开关，另保留5种无几何legacy模式。输入hash见differential.json。
测试与核验由同一执行者完成，不宣称独立上下文复核。合计每模式21项host
unittest；未跑需要Torch的全库测试，也没有GPU、质量、速度或显存改善数字。

## 后续与边界

历史重测只针对实际同时启用几何和上述模式的命令；应改成单一明确合同后重跑，
保留旧结果。固定配额子页nl的整页保底缺陷仍为
TL-EXPLICIT-NEAR-WIDTH-20261019，原候选补丁继续等待生产GPU验证。
候选内L2、批量显式选择与overlap未在本轮实现，不能由构造期修复推导提速。
发布以最终fetch、普通main更新和远端SHA/tree回读为准。
