# TL-DECODE-INPUT-20261015

## Plan

用户授权的独立主库代码优化，延续未完成的 decode 输入契约；main基线bb3991f，e138-ablation仅对照不合并。保持chunk2048、预算、选择器和kernel数学。主执行者维护本文件与唯一索引；独立上下文核验host守卫与CPU证据。原始输入/复现/版本及边界见[报告](../../advice/2026-10-15_eager_decode_input_contract_fix_by_gpt.md)。

## Progress

- CPU确认公共ops接受多个query却只计算首行；Q/K宽度不同可越过host门至fused封装，GPU越界后果仅静态风险。
- 先保存修前14失败/14错误，再新增rank/单query/batch/mask及QK宽度守卫。41项ops普通与-O通过，48个合法路径CPU输出与基线逐位相同。
- 不修改其他执行者、guide、数据；当前上层已有限制，尚无历史数据污染证据。真实GPU、速度、MLA decode padding/bias仍待验证；持续优化任务保持进行中。
