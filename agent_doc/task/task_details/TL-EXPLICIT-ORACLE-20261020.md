# TL-EXPLICIT-ORACLE-20261020

- 目标：固定nb/nt/nl不被E151c逐层prefill oracle静默覆盖。
- 基线：50abf698a16c506f30fe72a3fbb1f0a4df080d15。
- 已完成：修前9冲突子用例失败；构造期fail-closed；普通/-O各25项host测试与212合法状态差分通过。
- 证据：[中文简报](../../advice/2026-10-20_explicit_quota_oracle_by_gpt.md)、[原始日志](../../results/explicit-oracle68/)。
- 边界：仅host构造验证；GPU、完整依赖、实际选集、模型质量与性能未测；不改选择算法。
- 下一步：发布远端回读；追踪混配历史命令，若实际使用则按单一合同重测；继续既有GPU待验任务。
