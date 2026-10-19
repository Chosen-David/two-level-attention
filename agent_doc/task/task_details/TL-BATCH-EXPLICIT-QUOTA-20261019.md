# TL-BATCH-EXPLICIT-QUOTA-20261019

## Plan

按用户当前独立仓库授权审查最新main增量，保留他人改动。基线
`a2ef9163796e822379b3f592b1aeeea431039436`，干净独立twolevel-audit-47检出，
本地flock串行锁；不声称跨机器排他。已读TASK、人类guide及相关advice；
guide不编辑，不扩大到SGLang/agent。使用实现优化技能的先复现后修复流程。

范围：E155显式near配额未同步batch旧比例公式；只加host入口守卫，保持
chunk2048及selector/kernel不变。先修前失败、普通/-O测试与独立只读复核，
再fetch、仅暂存本次文件、原子普通main发布与回读。无GPU/付费资源占用。

## Progress

三个显式字段（含0）改为回退逐组；旧缺字段/全None配置保留。发布前main
推进到1814ea932641afe3e930e6339677bd51472a2033，已重放并复查E155d改动，
新增合法归一配置覆盖；最终6项测试在最新修前函数上25子用例失败，修后
普通/-O各6项通过。AST执行生产host函数，不是Torch
端到端或GPU验证。独立复核支持最小门禁，未扩大boundary/quant门禁。
完整证据见 `../../advice/2026-10-19_batch_explicit_quota_guard_by_gpt.md`。

## Pending

真实CUDA选集/输出、prefill→decode交接、模型质量和完整耗时/显存未验证；
回退可能较慢，无性能收益声明。批量显式配额支持及历史受影响实验重测待办。
本项代码入口修复与持续性能目标分开；发布由主控最后fetch及远端回读确认。
