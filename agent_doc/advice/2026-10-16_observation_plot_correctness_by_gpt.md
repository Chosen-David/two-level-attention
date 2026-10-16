# 整箱诊断与E6证据边界

用户授权的有界任务：[TL-OBS-PLOT-CORRECTNESS-20261016](../task/task_details/TL-OBS-PLOT-CORRECTNESS-20261016.md)。
47e4ab2的whole-bin排序不是token TopK上界；箱末可超预算，near分界遗漏SWA偏移，NaN配对分母也不一致。
本次修复采用不超2048的整箱mass-order skip-nonfit诊断，并输出实际预算与有限分母，不能推出真实单池token scorer表现。
128个query/layer观察来自4条来源序列，不是128个独立实验；文档重叠未核实。
当前图E6面板blocked，不从旧概率/区域/末query复用gate安全性或skip阈值。
O3仅保留原reference-mid-TopK比例与unpaired random标签。旧图/脚本/出处可从47e4ab2恢复。

CPU合成验证不等于科学采信；本轮没有GPU、重采、动态算法或稿件目录修改。仅推原dev，由用户审核合main。

## Opus 核验与合并裁决（2026-10-17 01:0X）

核验 confirmed：47e4ab2 的三个数值缺陷（whole-bin 箱末超预算 / near 分界漏 SWA 偏移 / NaN 配对分母）真实存在——但 main 侧 e021281 谱系已在 e5f5b7e 统计审查轮吸收全部三项修正（预算接纳前检查 acc+n_tok<=budget、bin 绝对区间对照 region_bounds.near_lo、两臂最小值分离披露），见 fig_o_observation.py docstring 修正记录 1)-5)。

**合并裁决（ca4ca8f）**：O_observation 全系列（渲染器/tsv/svg/png/pdf/validity）取 main 出版版；GPT 诊断版渲染器与 validity v4「gate-blocked 降级」未采纳，理由：①(a)(b) 的 whole-bin 启发式限定手稿 caption 已如实披露（「整页注意力权重排序只是比较参考，并非最优 oracle、精确词元 TopK」）；②(d) 的 E6→gate 认证边界手稿已披露（「门控默认关闭，需要独立评估」「该统计量不约束解码误差」）——GPT 侧主张与手稿披露口径一致，降级动作与披露动作语义等价，无需撤出版图；③128 query/4 序列非独立观察的限定同样已在图注披露（「四序列贡献128条相关观测」）。GPT 新增物（O_observation_diagnostic.json、record_provenance.py、review logs、TL-OBS-PLOT-CORRECTNESS 条目）保留入库，TASK.md 拒绝其删除 main 侧用户指示待办（DAC-BEST-PAPERS 等）。

sink4 转向提醒：O 图 (a)(b)(c) 当前数据源为 E140v2/sink128 采集；论文主口径已转 sink4——E153/e146c 系列出数后该图是否随 sink4 重采属叙事决策待用户定，本裁决不锁定数据源。
