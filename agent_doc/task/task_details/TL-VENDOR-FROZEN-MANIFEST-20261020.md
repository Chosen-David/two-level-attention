# TL-VENDOR-FROZEN-MANIFEST-20261020

## Plan

审查最新main新增E155e元数据路径；修复冻结runtime内容与后读manifest不一致，保持固定配额、kernel和有效未变源身份。根TASK及guide只读。用户授权本独立仓库实现/测试/文档和直接main发布。

## Progress

- 基线f622c907；源替换/删除4个反例修前失败，修后扩展7项普通/-O通过；无Torch host源码执行，未声称GPU验证。
- 普通/-O固定配额17项、batch guard6项均通过；E155e既有7项host检查通过。
- 详情与原始日志：agent_doc/advice/2026-10-20_vendor_frozen_manifest_by_gpt.md、agent_doc/results/vendor-frozen80/。独立复核/远端回读后发布完成；near宽度及T7仍按既有待验任务推进。
- 独立tl80_vendor_review复跑7项普通/-O，17项base/current身份比较和32组固定配额配置的完整名字/manifest对照均通过；源冻结同时发生symlink retarget的竞态仍在本修复范围外。依据独立报告准许有界host修复发布；远端树须与候选一致。
