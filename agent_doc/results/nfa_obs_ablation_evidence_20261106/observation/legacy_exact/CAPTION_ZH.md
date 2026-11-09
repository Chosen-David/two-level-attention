# 图注：HP0-L8-q7188 历史诊断的绝对 Top-K 页覆盖

- **数据**：真实 Q/K NPZ（SHA256 `dbaab43d0e7838ce78e604ad65585f6af35b1f890d3709b8aab59b6e3d5ba82b`），HotpotQA sample 0、layer 8、零基 query position 7188。逐 Q-head 在全 128 维、全部 7189 个因果可见位置上计算 softmax，再做 4 头组均与 8 组平均。
- **oracle**：完整页域 `[128,7040)` 内固定 Top-768；概率降序、同分绝对位置升序。Far/Near 分别有 292/476 个 oracle token。
- **候选**：历史 tail32 坐标 `[48,64)∪[112,128)`，Avg 与 MinMax 各精确选择 32 页，即 2048 个候选 token；`[7040,7061)` 的 21-token 不完整 Near 页在排名前排除。
- **实测**：Avg L1/Far/Near 覆盖为 **554/768、108/292、446/476**；MinMax 为 **521/768、108/292、413/476**。本例不支持“MinMax 的 Far 覆盖更高”，负例原样保留。
- **边界**：这是 sink128/SWA128、单平面、Top-768 的历史诊断，不是当前 sink4、per-KV-group、K2=1024 的生产结果。页候选不是最终 L2 选集，因此不报告 L2 recall。
- **验收**：独立实现与独立进程复算 49 项检查通过（同一 AI owner，非外部审稿人）；原始记录保持 pending， scoped acceptance 见 `validation/legacy_exact_independent_check.json`。
