# TL-EXPLICIT-INTEGER-20261019

- 来源：用户持续代码审查授权、固定配额关注点。
- stage：host correctness；owner：主执行者；depends_on：无。
- inputs：main `bf9b07994b187cdebfd5070d94b80d27292e73ec`，nb/nt/nl整数计数合同。
- action：复现小数截断绕过检查；构造期严格整数协议，保留整数串/nt=off。
- outputs：[中文证据](../../advice/2026-10-19_explicit_quota_integer_by_gpt.md)、
  `tests/test_explicit_quota_integer.py`、`agent_doc/results/explicit-quota-integer-20261019/`。
- done_when：修前失败保留；普通/-O每模式19项回归与108对照/6反例差分符合预期。
- status：host有界修复验收完成；发布后由主控回读远端SHA/树另存发布回执。
- on_failure：不发布；保留日志，核查类型兼容或并发main变化后重新验证。
- resource_budget：本地标准库CPU测试，未新增依赖/设备/费用。
- 未验：完整模块导入、真实selector/GPU、模型质量及性能；未更改kernel。
- 后续：有环境时补设备集成；排查直接Python传非整数的历史配置，不修改旧实验数据。
