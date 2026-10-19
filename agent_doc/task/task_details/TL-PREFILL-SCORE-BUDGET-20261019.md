# TL-PREFILL-SCORE-BUDGET-20261019

## Plan

来源：用户要求保留 chunk2048、分离算法选择粒度与执行 tile、审查真实内存/性能浪费。
目标：candidate eager 共享 KV 选择的 score 行块预算按 Hq 而非 Hkv 计数，
保持 gather/mask 形状、选集及保护区契约。
验收：实际张量回归覆盖多种头配置和分块/分流；授权 GPU 验证数值与实际显存，
同条件全路径计时后才能报告性能收益。

## Progress

- 基线 main：`bcd5b58c75628af45a0bd1cc840647f583cc4d7b`；干净独立检出。
- 已确认缺少 GQA 因子的源码证据和标准库形状反例；Hq32/Hkv8 时单 score
  逻辑载荷928MiB，超出代码512MiB预算。不是 GPU 峰值/OOM 实测。
- 独立只读复核确认；公共组模式使用完整 chunk 并集容量，未发现其单组 cap 低估。
- 本次仅审计与复现脚本，未修改生产路径，未调整外层 chunk2048。
- 阻塞：当前未安装 Torch、无可用 GPU 验证入口；生产修复和性能实验待完成。

## Evidence / Next

- [审计与重测范围](../../advice/2026-10-19_prefill_score_budget_audit_by_gpt.md)
- [源码绑定复现](../../results/prefill-score-budget-20261019/probe.py)
- 下一步：分离 score/index 头数，用实际张量测试强制行分块和所有受影响路由，
  然后验证 GPU 正确性和完整路径内存/时间；单行/最小组超预算需明确处理。
