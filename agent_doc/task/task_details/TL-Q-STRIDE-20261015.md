# TL-Q-STRIDE-20261015

## Plan

延续用户授权的TwoLevel主库优化，核验上轮未决Q特征stride与kernel ABI。基线main e87537a，保持kernel数学、预算、chunk2048与guide只读。不操作其他仓库或未知GPU任务。

依赖：源码/派发 → 修前CPU ABI复现 → 最小host operand整理 → 生产回归 → 独立复核 → fetch/普通发布与回读。主执行者唯一写入；独立上下文只读验证。缺GPU只接受有界host修复，不宣称GPU/性能验收。

## Progress

- 真实wrapper对转置Q保留qs stride(1,H)，与kernel +offs_d不符；新增3测试修前6子项失败。条件连续化位于现有低精度缩放/升精度之后，标准路径不复制。
- 普通及-O全ops44项通过；独立120个CPU ABI矩阵、96个禁止额外复制场景通过；不代表实际GPU数值/速度。
- [报告](../../advice/2026-10-15_eager_query_feature_stride_fix_by_gpt.md)记录输入/源码hash、环境、取舍及历史影响未知。下一步：实际GPU转置/标准布局对拍、memcheck、同机完整成本测量；长期优化不因本项完成停止。
