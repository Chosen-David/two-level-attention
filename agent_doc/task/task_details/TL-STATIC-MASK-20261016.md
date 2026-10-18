# TL-STATIC-MASK-20261016

Task-ID: TL-STATIC-MASK-20261016
Date: 2026-10-16；恢复并复验：2026-10-18
Owner: GPT

## Plan

用户于 2026-10-18 恢复本独立仓库修改及直推 main 授权；保持原定时任务启用。
复用暂停补丁的有界正确性方案，基于新 main 重现后发布；禁止把历史验证、CPU 数学或静态推断当 GPU 验收。
人类 guide 只读，外层 chunk2048、选择预算、已有实验数据不变。

- CODE-T001：当前源码/指南/检查点核验 → 当前 baseline 复现 → mask 与缓存类型修复 → 指定 CPU 回归 → 发布。
- GPU 后续依赖当前可用且已授权资源；验证失败保留证据，不能改门槛或直接认定提速。

## Progress

2026-10-18：基于 main `4828e4d` 核实生产源文件与旧待验证补丁基线完全一致。
实际 e138-ablation `2ed6a1f` 仅用于对照。同步 `784c1a4` 的并发论文增量。
重建 Torch 2.8.0+cpu / Transformers 4.56.2 环境；当前修前 3 项失败，修后专项 12 项、ops 66 项、E152 16 项、捕获 11 项，普通与 `-O` 均通过。
512 个 bool mask 穷举只接受精确下三角；当前 CPU SDPA 合法 static/dynamic 输出逐位一致。
CPU profiler 仅确认一个标量读取消除，未减少该输入的 contiguous/clone，未测速度。

报告：[恢复审查与验证](../../advice/2026-10-18_static_cache_mask_resume_by_gpt.md)。
结果：[验证记录](../../results/static-mask-resume-20261018/validation.json)。

状态：CPU 正确性修复可用；GPU/真实模型、Graph、完整请求时延与历史实验影响仍未验。
待核 TL-E151C-ROUTE-20261018 配额路由及 compact indexed prefill 性能路线见报告。
定时任务持续推进；不因本次修复结束而关闭。
