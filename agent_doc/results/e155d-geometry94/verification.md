# TL-E155D-GEOMETRY-20261020 验证

基线 `8d541d4001b184c2c14062e5e7303dce9b7b12a2`；独立worktree、独占锁、
fresh fetch一致。无适用AGENTS；根TASK/AI索引/相关advice及guide安全段只读，
人类目录、collector、原数据和远端作业未改。实现Skill用于基线→修复→实测边界；
沿用标准库哈希与现有生产manifest，未增加依赖或第二套配置协议。
实验分支另行fresh fetch：e138-ablation=`2ed6a1feb78a7b5bcf3cac922b4510c3968714a5`，
prefill-compact-block-traversal=`93e4a364ea3be4897a076ef62feddb60145eccef`；均未合并，
本项不采用实验分支性能作为main结果。

## 固定证据

| 对象 | SHA-256 |
| --- | --- |
| 基线分析器 | `c719089b3c4e279abc1a42430d116802b901929054f936ec52100abbd4f44658` |
| 候选分析器 | `85ae9d442a09fc19fec9cd564929b52a81cf5cff11da2b611bc57d75655c4e2f` |
| 新geometry单测 | `8db3506b6dc9815fb8b06c1c1308d0db15a0c23885c47195ac0e010caf693fae` |
| 实际producer检查脚本 | `18f4b4e61819c40cf55389b5b9613390d9a61af1702d947a745135d2b20a308d` |
| 修前最终日志 | `28c1c35ac89e23757e8b8de5a0b03582819134da4983b4b12b200ded635fab26` |
| producer完整manifest输入串 | `70b6b0f0f99cf7f3e213cd98169540a648ca7e5b28db6e3a6eae8f9575dbd241` |
| 全入口差分预测输入 | `3aaf366af9095b401cb7517f1767b986d99093e03091adf1b3be7d9f3125f8f9` |

修前最小8方法：7失败、0error、1合法87格控制通过；失败直接观察错误几何/
缺失或错误manifest仍产出avg5，非fixture加载错误。更早before.log的清理错误
单独保留，不充当生产缺陷证据。新增16方法覆盖nb/nt/nl/块大小、类型、CTRL、
缺失/非对象/非TLI、filenamehash、评分中sidecar变更、main排名过滤和合法网格。

```text
before-v2.log: Ran 8 tests in 0.325s; FAILED (failures=7)
python -m unittest discover -s tests -p 'test_e155d*.py' -v
after-resume.log: Ran 57 tests in 0.815s; OK
python -O -m unittest discover -s tests -p 'test_e155d*.py' -v
after-resume-opt.log: Ran 57 tests in 0.768s; OK
python tests/check_e155d_geometry.py  # 普通/-O相同
PASS: 87 real producer geometry manifests; 51 wrong labels rejected
python tests/check_e155d_snapshot.py  # 普通/-O相同
PASS: 24 full-entry baseline differential cases; input_sha256=3aaf366af9095b401cb7517f1767b986d99093e03091adf1b3be7d9f3125f8f9
python tests/check_e155d_score_units.py  # 普通/-O相同
PASS: 24 baseline differential cases; actual aggregate scorer scale
```

相邻host专项`PYTHONPATH=tests python [-O] -m unittest test_vendor_frozen_manifest
test_explicit_quota_modes test_explicit_quota_oracle test_explicit_quota_integer
test_tli_enum_contract test_e152_result_store test_e152_request_timing -v`：各35项
通过（0.459/0.504s）；既有181/212合法构造状态差分保持，仅回归不作本补丁收益。
旧snapshot fixture仅补真实格式manifest/文件名，原断言未放宽；全入口差分仅
排除路径/新增provenance字段，任何evaluation_errors仍报错。

## 验收范围

Python3.12.14、Linux6.18.55/x86_64/glibc2.39；Torch/Transformers/Triton缺失，
无nvidia-smi或可见NVIDIA设备，无架构/驱动/编译kernel配置。没有运行GPU或
未知训练任务。实际info.py仅按既有load_info移除torch import，任何tensor访问
均报错；完整resolve/name/manifest代码运行，不是另抄公式。评分与逐例metric
使用合成替身；未运行真实LongBench scorer/模型，未覆盖张量batch/head/stride。
测试耗时不是性能基准，无质量/速度/显存/kernel数/host同步收益主张。

[独立结果复核](independent.md)确认同源码/测试SHA无阻断；另复演直接基线8项
7失败0error、对直接基线各24组全入口差分及9类畸形/删除隔离负控。独审报告
SHA256为`15f28920298bc096daca1c230b5fb97adbda70df05c9d96d5417f176d44a7def`。
不得仅据上述绿色日志发布高风险kernel。本次只涉及
CPU汇总准入，非几何超参/跨任务完整配置、跨主机sidecar一致性、运行真实性、
共享staging/ABA和真实T2历史影响未闭合。必要后续见唯一TASK详情，不另建任务表。
