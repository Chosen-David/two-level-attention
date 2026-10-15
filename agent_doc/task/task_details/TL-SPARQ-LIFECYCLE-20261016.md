# TL-SPARQ-LIFECYCLE-20261016

## Plan

- 来源：用户持续审查、修复并验证后发布 main 的授权。
- 目标：修复实际 B 轨 SparQ 适配层跨请求 V_mean 串用；不替换官方算法，
  不改 TLI/预算/chunk2048，不将 CPU 结果解释为 GPU 性能。
- 基线：main `488f071ab9b8462d827d768d02bb0c618c996244`。
- 链路：真实 benchmark 入口→固定源码及官方均值契约→修前失败→host
  cache owner/length 修复→CPU 回归/正常请求对照→fetch/非强推/远端核验。
- 作者：本次 GPT，隔离检出；允许文件只限 adapter、专用测试、此详情、
  任务索引和报告；人类 guide 只读。
- 验收：跨请求恢复与 fresh adapter 一致；正常请求保留官方增量算术；
  不 pin 请求 cache、不新增设备同步；保留失败和未验证边界。

## Progress

- 修前4项失败：第二请求输出、交错 cache、同对象重填、每步 item。
- 已实现 weakref owner + host length 绑定；prefill 前失效，无 cache/不连续
  owner 重建均值，正常追加仍增量。官方核心未动。
- 专项12项通过；ops62项普通/-O通过；正常36场景144 decode输出与修前逐位
  一致。普通 discover 的18个惰性导入错误和 probe 哈希错误均保留并说明。
- 证据：[复现、绑定与重测范围](../../advice/2026-10-16_sparq_request_lifecycle_by_gpt.md)。
- 状态：CPU适配层修复完成，发布前待最终 fetch/回读；真实模型/GPU与历史
  SparQ跑批影响核验未完成，无速度/精度提升数字。
- 下一步：有授权空闲GPU及真实Qwen后做双请求smoke，再以新postfix重跑相关
  SparQ质量/整请求成本；不修改旧数据或占用未知训练作业。

- 发布闭环：适配层修复已在 `44a98682538209e3fcd44cd12b703012f277b58b`
  发布 main，fetch 回读 tree 及全部5文件字节一致；此前“待回读”为当时快照。
- 后续 TL-SPARQ-TEST-IMPORT-20261016：定位并修复本 fixture 全量恢复
  sys.modules 导致的惰性导入/重复注册问题，生产 adapter 不变。普通 discover
  修前62项18错误；修后63项普通/-O均无需预加载通过。仅改精确 alias 恢复。
  [证据与限制](../../advice/2026-10-16_sparq_test_import_isolation_by_gpt.md)。
  GPU/真实模型与历史质量待办不因此完成。

- 2026-10-16续审 d9535d3 的新回应：残留Tensor.item全局mock与Dynamo冲突。
  TL-SPARQ-SCALAR-GUARD-20261016在CPU编译边界先复现InternalTorchDynamoError，
  改为实际标量派发拦截；66项普通/-O通过，旧wrapper真实item仍被检出。
  生产adapter/官方核心均未变；远端具体软件栈复验、GPU/质量待办仍未完成。
  [修复与边界](../../advice/2026-10-16_sparq_scalar_guard_compat_by_gpt.md)。
