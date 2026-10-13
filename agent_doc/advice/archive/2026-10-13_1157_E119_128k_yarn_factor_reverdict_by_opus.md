# E119 128K yarn factor 重判：round18「三臂作废」撤回（Opus 报备）

> 作者：Opus · 2026-10-13 11:57 · GPU 监督器收割轮（用户 13:2X 改制后并入监督）
> 致：用户 + DeepSeek（fullkv 128K 收割链交接）

## 结论（一段话）

round18 判定「fullkv/mavg/aavg 128K 因 yarn_factor=2.0 PE 溢出精度受损而作废」是**误判**：那个 2.0 是 **score_ruler_formal 打分时的 operator 声明值**（057 correction sidecar 早已标注 operator_declared_not_effective），不是 producer 实际配置。e119 批 canonical 派单脚本 run_ruler_e109.sh 在 128K 档只传 `--yarn` → auto 表 131072→**4.0**，三臂实际全程 effective 4.0。**128K 锚点（mavg 47.49 / aavg 42.56 / fullkv 46.43）恢复有效，mavg/aavg 128K 重跑取消**，主表 §3 128K 行与脚注 4 无需改动（本就正确）。

## 证据链（五条独立线索全部指向同一结论）

1. **逐位对拍（最硬）**：DeepSeek 11:45 重打分用的 fullkv 128K 新产物（producer_receipt 验证 effective 4.0）与旧产物（Oct-9/10）逐位一致：4 任务（multivalue/single_1/fwe/cwe）× 100 行 × **0 差异**。若旧产物真是 2.0（81920<131072 溢出），free-form 生成逐位一致不可能。
2. **派单脚本**：run_ruler_e109.sh 128K 档只传 `--yarn`，YARN_FACTOR_AUTO={65536:2.0, **131072:4.0**}。三臂共用同一脚本。
3. **057 标注**：yarn_correction sidecar 早写明 yarn_factor_status=**operator_declared_not_effective**、producer_native_evidence=missing——「2.0」从头到尾只是打分侧声明。
4. **e116b 独立批交叉印证**（10-11，声明 4.0，新口径独立生成）：fullkv **46.43 同值** / aavg **42.56 同值** / mavg 47.2 vs 47.49（Δ0.29 为 E116b near/SWA 修复的生成差异）。
5. **重打分逐值不变**：DeepSeek run 20261013114511 per-task 与旧打分完全相同 → AVG 46.43 不变；主表脚注 4「128K factor 4.0」早已写对。

## 处置

- **mavg/aavg 128K 重跑取消**（原登记转 Opus）：与 fullkv 同脚本同派单模式，effective=4.0 已被直接证明，重跑 22 格只会复现同分。省 ~11 卡×2h+。
- **fullkv 11 格重跑不算白跑**：产出 receipt 验证的正式发布（46.43，producer_receipt provenance），永久补上 legacy manifest 的元数据缺口。
- **quest 128K 代表任务**继续按 DeepSeek 待办推进（对手 baseline 范围）。
- 判决 JSON：`exp/trace/results/e119_128k_yarn_factor_reverdict.json`。

## round18 教训修正（替代原教训）

原教训「跨档位打分必须核对 producer yarn identity」方向正确但用反了：**legacy manifest（identity_mode=legacy-partial）的 yarn_factor 字段只代表打分时 operator 声明，不可反推生产配置**。判定生产配置须用 producer 侧证据（057 receipt / 派单脚本 auto 表推演），这正是 057 机制存在的原因。minf 49.53 > fullkv 46.43 的 sparse>dense 现象与 factor 无关，归对手 baseline 领域待解释，不影响 TLI 锚点。
