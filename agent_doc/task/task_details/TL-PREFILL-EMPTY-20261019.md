# TL-PREFILL-EMPTY-20261019

- owner: GPT; scope: reference prefill 空行兜底正确性。
- baseline: 7eb4241；e138-ablation=2ed6a1feb78a7b5bcf3cac922b4510c3968714a5 仅对照。
- contract: 仅选择集与因果相交为空的行/head补对角线；其他行选集不变。
  外层chunk2048、组选择、预算、sink/SWA、L2与kernel均不变。
- DAG: 同步/既有问题核对 → 修前失败 → 局部修复 → 普通/-O回归 → 独立边界复核 → 原子发布/回读。
- acceptance: 混合空/非空行及共享/per-query head不互相污染；与显式绝对索引oracle一致。
- resources: 本地CPU单线程；无CUDA设备，不动远端作业，不报告GPU或真实模型收益。
- on_failure: 保留失败证据，暂停发布受影响补丁。

## Progress

独立工作树保留原检出中不相关字体临时文件。main相对上轮新增论文与E147结果，
未修改本次reference函数。没有发现仓库AGENTS.md或Git锁；进程列表能力不作为外部无并发证明。
本轮只改reference空行兜底、新回归及证据文档；不改历史实验或人类guide。

修前6个runtime/head场景中reference 3个失败，candidate 3个通过。
修后ops测试普通及-O各67项通过，E151c相邻21项通过。
独立上下文1324个CPU边界场景通过并接受限定范围修复，报告/脚本已归档。
发布前再次fetch，main仍为7eb4241；普通原子发布及远端回读由主控完成。
证据与影响范围见对应advice；后续GPU/真实模型/性能验证继续保留。
