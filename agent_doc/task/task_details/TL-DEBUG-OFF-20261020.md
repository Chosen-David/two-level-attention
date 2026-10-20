# TL-DEBUG-OFF-20261020

Date: 2026-10-20

## Plan

用户要求固定配额真实bug及无效host同步审查。基线0b661c4；仅修复六处
TLI_DEBUG关闭字符串被非空真值误开启，与已有L1语义一致。不改算法/预算。

## Progress

修前54次误入调试体，修后0；普通/-O各24项host测试通过。
168关闭+126开启条件差分及非调试AST隔离通过。源码、输入hash、原始短日志
和重放入口见[简报](../../advice/2026-10-20_debug_off_contract_by_gpt.md)。
发布前fetch和远端SHA/内容回读由主控单独完成。

## Pending

无Torch/Triton/GPU，未运行实际selector/模型/全库，不宣称提速。
同输入同预算GPU计时和上述环境历史任务重测仍待执行。
