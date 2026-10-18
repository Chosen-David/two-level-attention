# TL-BATCH-SINK-ZERO-20261019

## Plan

目标：修复零sink进入使用token0保护哨兵的批量prefill路径；仅修改入口守卫，不改kernel、预算或chunk2048。
基线ffaaf9fe51c86608a45b4a809890a1e065dd55a4，独立工作树twolevel-audit-0450；旧0355检查点与其他工作保留。无适用AGENTS文件，已读取人类GUIDE及相关advice，guide禁止编辑。当前仅root执行；本地flock串行锁已取得，ps不可用，不声称跨机器排他。
依赖：重新fetch/read→修前失败→守卫→普通/-O回归→中文简报→再次fetch→普通main发布及回读。
范围：一个host条件、5项回归、合成复现及必要文档；CPU单线程，无付费/新机器/GPU任务。真实CUDA、模型质量与性能缺口单独保留。

## Progress

修前5项中3失败；旧CPU胶水选集丢token0且误选token1，预算数不变。修后新增测试通过；普通/-O每模式75项ops+17项相邻unittest+8个sink场景通过。未测GPU、真实模型及性能，不称提速。
详见 `../../advice/2026-10-19_batch_zero_sink_guard_by_gpt.md` 及 `../../results/tl-batch-sink-zero-20261019/`。守卫确保无保护sink回退，required Graph明确拒绝。历史数据未改。
发布前再次核对远端，发布后主控回读SHA/代码；代码修复与持续性能目标分开。待完成GPU回退/状态交接及历史受影响格重测；不关闭持续任务。
