# Task-ID: TL-PARTITION-RATIO-20261019

## Plan

来源：用户持续审查并修复独立 TwoLevel 仓库授权。
范围：alpha/beta 比例有限性与 [0,1] 构造期校验；合法选择算法不变。
修前失败→最小守卫→普通/-O host 回归→独立复核→fetch/普通发布/回读。
不动人类 guide，不合并实验分支，不使用未知 GPU，不更改历史结果。

## Progress

- 基线 36c0414，修前普通/-O 各 28 个拒绝子用例失败。
- 新增 5 行 host 校验；修后每模式专项 3 项、预算 7 项、batch 守卫 6 项通过。
- 依赖 Torch 的 3 个 oracle 测试模块无法加载；GPU/模型/性能未验。
- 证据与原始短日志：[advice](../../advice/2026-10-19_partition_ratio_validation_by_gpt.md)。
- 独立复核通过：563合法legacy +22显式几何全实例状态不变；56非法配置新增拒绝。
- 完整模块AST移除唯一新守卫后等于基线；生产host契约修复验收完成，远端发布回读单独确认。
