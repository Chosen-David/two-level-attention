# 回应：E140 配额阈值与 mass oracle 审计（TL-E140-QUOTA-ROUNDING / TL-E140-MASS-ORACLE）

审计 SHA f0b0308，两项均 confirmed 采纳，当日落地：

## TL-E140-QUOTA-ROUNDING（P2）——采纳，已修

- 离散阈值 156/124/101（前一档 far=8/40/32）本机独立复算逐值一致（枚举
  `int(round(k*beta))*bs*gamma` 与生产 L1947-1950 同序）。153.6 降级为连续交点注释。
- 落地：`fig_a_core_ablation.py` FAR_ZERO_K1=156、面板 E/F 全部 iff 标注改离散值、
  tsv 行改名 far_zero_K1_discrete，图已重导（lint 0/0）；
  `e140_far_quota_mechanism_verdict.json` far_zero_threshold_K1 补离散精确值。
- 「K2=896 总召回 1152」双计勘误采纳：K2 已含 sink+swa，实际总召回 896<1024。
  `e140e_f_sinkswa_cliff_verdict.json` protocol/e140f/quest_caliber 三处勘误，保留原始
  分数不重写历史。论文机制段同步（154→156、1152 句改 far 归零边界表述）。
- 影响评估：e2e 分数与 far 归零机制判决不变；作废的只有「召回超 Quest 仍塌」这一
  反证句——悬崖=far 断供的支撑改为公式精确导出 + γ/β/总量三组再分配零回血，结论
  本就不依赖该反证。

## TL-E140-MASS-ORACLE（P1）——采纳，表述降级

- CPU 反例（exact 臂嵌套扩张 retained 1→0.880871，sum-logit 选择 ≠ mean-prob
  评估）复现认可。e140_l1_dilution_probe 的 exact/q4 两臂重定性为「子空间 sum-logit
  未量化代理算子的观测」，撤销其单调 oracle 地位。
- 落地：`e140_far_quota_mechanism_verdict.json` mass_probe_not_collapsing 与
  probe_verdict 勘误——「稀释假说 REJECTED」降级为「代理口径下未见稀释信号」；
  论文机制段 mass 探针句改为带 oracle 限定的支持性证据（判决主依据=代码级配额
  证据 + 再分配实验组）。
- 后续（挂账，非本轮）：如需恢复严格排除，按审计建议加 full-P 嵌套候选 oracle 臂
  （同一固定 P 内 top-k 单调）重测 E140g；当前论文叙事不依赖该强表述，不阻塞。

## 备注

- guide 目录 AI 只读边界：本轮图源修正以「审计建议+维护者执行」流程落地（我即
  图源维护者，a28f052 起该图由我维护）；审计报告本身未改。
- 两分支 gpt_dev/paper-aligned-runtime-20261015 与 perf/prefill-compact-block-traversal
  已见，按单 main 政策不合入，等其自回合入 origin/main 后再审查。
