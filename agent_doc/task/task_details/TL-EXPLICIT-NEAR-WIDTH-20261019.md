# TL-EXPLICIT-NEAR-WIDTH-20261019

## Plan / Progress

用户手动触发固定配额审查；基线 d7d71009cb8962490b9468c63a66f44d196e4c01。
确认合法显式 nl<bs 在三个消费点被旧整页保底放大，与 D′ 标定不一致。
210配置/840源码绑定host分支检查：原版普通/-O各270错误；三行候选各0错误，
相邻预算回归普通/-O各7项通过。报告、复现脚本和待验证patch留档。

## Pending

生产源码未应用补丁；缺 Torch/CUDA，未验证实际tensor选集/输出、GPU、模型
质量或性能。必须完成报告内GPU矩阵后再发布生产修复。全前缀L2与冗余clone
只登记静态成本，未宣称提速。详见
[中文证据](../../advice/2026-10-19_explicit_near_width_manual_audit_by_gpt.md)。
