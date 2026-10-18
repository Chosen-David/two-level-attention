# TL-DECODE-CU-STRIDE-20261019

来源：用户对独立 two-level-attention 仓库的 bug 审查与修复授权。

- 已完成：真实 fused host 链路 cu stride ABI 最小复现、按需布局修复、
  修前2错误+2失败→修后专项3方法通过；ops70项普通/-O通过；独立30布局复核。
- 基线：bb14bc8a0347f1e440c78d397a3e2019e4a16c82；仅host wrapper修改，无kernel变更。
- 证据：[中文简报](../../advice/2026-10-19_decode_cu_layout_by_gpt.md)，
  [原始日志/复现](../../results/decode-cu-layout-20261019/)。
- 待办：合法连续/间隔cu GPU对拍、非法逻辑长度flag3、实际派发/请求生命周期，
  完整性能成本与相关自定义cu历史实验重测。缺GPU，不得标记设备验收完成。
- 未完成优化：候选内L2、紧凑绝对索引、fused prefill、overlap的实测门禁。
