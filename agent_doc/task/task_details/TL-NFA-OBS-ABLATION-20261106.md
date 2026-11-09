# TL-NFA-OBS-ABLATION-20261106

## 状态与范围

- 状态：**完成，`usable-with-scope`**。
- 本任务收官的是 O-GATE 固定轨迹机制回放和 B-REPLAY 固定 Q/K 预算回放。
- **不是新增质量分数、性能实验或 production-Triton bit-exact 验收**；不得据此宣称任务精度保持、等价或加速。
- 生产源身份：`f5f3df3c4fe4c4af39c9074db0baaa5e00770eab`；模型为 Qwen3-8B。
- 原始大产物保留在 pc02 `/home/wangyuanshuo02/tla-nfa-evidence-20261106/agent_doc/results/nfa_obs_ablation_evidence_20261106/`；git 只交付下述两个精简 JSON，不包含 NPZ、逐请求 JSONL 或逐文档大 JSON。

## 精简结果

- [O-GATE 精简结果](../../results/nfa_obs_ablation_evidence_20261106/gate_replay_summary.json)，SHA256 `e216ae56d51f210d6e1f2fe85631eb1ae75149909c4e675d1899582bbe5ddbca`。
- [B-REPLAY 精简结果](../../results/nfa_obs_ablation_evidence_20261106/budget_replay_summary.json)，SHA256 `3d384bf2d3814c14d1999977a593f7de2516169c46a18b6008246840e591b6c7`。

## O-GATE：12 plans / 240 请求

三数据集各执行 `gate-full/gate-far/gate-near/gate-dual` 四臂，每臂 20 请求。独立验收覆盖 3,624 条 trace 和 60 份 dual raw dense-oracle Q/K 输入。

| 数据集 | 每臂请求 | 每臂 trace | dual raw Q/K |
|---|---:|---:|---:|
| hotpotqa | 20 | 206 | 20 |
| gov_report | 20 | 420 | 20 |
| niah_multikey_2 | 20 | 280 | 20 |

正式独立总门：`status=usable-with-scope`，`producer_or_runner_imported=false`，SHA256 `e523591a472d4fa56ad78c7d5d635b791c6c960641c2989848e7f7536fc930d6`。raw oracle 验收只确认 Q/K 输入身份，不表示 O-GATE 已计算 dense-oracle attention。

## B-REPLAY：12 plans / 240 文档 / 69,952 cells

固定选择臂为 `regional-nearAvg-farMinMax`，几何为 K1=128（far96+near32）、low32、page64、sink4/SWA128、dense prefill。HotpotQA/GovReport 使用 O40 prefill Q/K 与 O-GATE dual decode Q/K；NIAH 仅使用 O-GATE dual decode Q/K，明确排除 O40。

下表均为**逐文档均值的描述性机制指标**，不是任务质量分。`fixed` 指 dense-oracle Top `min(892, eligible)`；`budget` 指按当前 K2 扣除 sink/SWA 后的预算 oracle。

| 数据集 | K2 | cells | fixed L1 recall | fixed final recall | budget final recall | fixed capacity ceiling | middle mass fixed/budget |
|---|---:|---:|---:|---:|---:|---:|---:|
| hotpotqa | 256 | 6768 | 0.7824 | 0.1381 | 0.4247 | 0.1563 | 0.4784/0.6385 |
| hotpotqa | 512 | 6768 | 0.7824 | 0.3530 | 0.5092 | 0.4563 | 0.6289/0.7019 |
| hotpotqa | 1024 | 6768 | 0.7824 | 0.5867 | 0.5867 | 1.0000 | 0.7554/0.7554 |
| hotpotqa | 2048 | 6768 | 0.7824 | 0.7356 | 0.6640 | 1.0000 | 0.8470/0.7910 |
| gov_report | 256 | 8480 | 0.7570 | 0.1378 | 0.4348 | 0.1531 | 0.4513/0.6225 |
| gov_report | 512 | 8480 | 0.7570 | 0.3549 | 0.5108 | 0.4511 | 0.6045/0.6802 |
| gov_report | 1024 | 8480 | 0.7570 | 0.5872 | 0.5872 | 1.0000 | 0.7311/0.7311 |
| gov_report | 2048 | 8480 | 0.7570 | 0.7191 | 0.6643 | 1.0000 | 0.8219/0.7666 |
| niah_multikey_2 | 256 | 2240 | 0.0950 | 0.0425 | 0.0911 | 0.1390 | 0.1568/0.2371 |
| niah_multikey_2 | 512 | 2240 | 0.0950 | 0.0660 | 0.0819 | 0.4260 | 0.1789/0.2120 |
| niah_multikey_2 | 1024 | 2240 | 0.0950 | 0.0798 | 0.0798 | 1.0000 | 0.1962/0.1962 |
| niah_multikey_2 | 2048 | 2240 | 0.0950 | 0.0886 | 0.0814 | 1.0000 | 0.2112/0.1845 |

结果只支持以下有界观察：固定 Top892 与预算匹配 oracle 在低 K2 下不可混用；三个数据集的 fixed L1 recall 不随 K2 改变；NIAH 的本组 recall 明显低于另外两个数据集。上述数字不能外推为端到端精度结论。

### 轨迹对照与边界披露

K2=1024 仅做 decode canonical logical support set 对照，**不是逐位对照**：HotpotQA 1,648、GovReport 3,360、NIAH 2,240 cells，合计 7,248 cells，`equal_cells=0`、`mismatch_cells=7248`。该比较是描述性记录而非相等门禁；它明确排除了以本回放证明生产选集一致或 bit-exact 的用途。

全部 12 plans 的 `cpu_zero_padding_count=0`、`fallback_triggered_cells=0`；生产 trace 未记录可见支持集之外的 padding，故保留 `production_padding_status=not-recorded-by-production-trace`，不伪造计数。首 decode seed 并集的增加支持数、超名义 K2 单元数及完整均值均保留在精简 JSON。

## 独立验收与回归

- 每个 B-REPLAY plan 均在 producer 结束后由独立实现逐文件重算并验收；12/12 为 `usable-with-scope`，`producer_or_o40_replay_imported=false`。
- combined 总门：12 plans、240 documents、69,952 cells，SHA256 `3c4b55ccca4c60a57ef7207a8837b28725e73bf706a24473f7c123a990250fce`。
- O40 capture 独立验收 SHA256：`a6f88cf30704d886b55c3661f39df4c0acdaa4d32ff960639ac665fb7a706f22`。
- 冻结测试在源 HEAD 工作树中普通模式 11/11、`python -O` 11/11 通过。
- producer SHA256 `4f8da37df4b68add7a098df9d58db8ca658a3cb0a959a88858e5ca88d910f707`；独立 verifier SHA256 `9b7823bc74e3e53afbcd51ffaa4abe1f45c9a497e987ef77c909ff418b981ca1`。
- 全部阶段 `performance_authorized=false`，未扩大冻结网格，未新增质量生成或性能测试。
