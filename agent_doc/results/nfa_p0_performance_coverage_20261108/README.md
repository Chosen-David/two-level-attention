# P0 完整 decode TPOT 覆盖审计

本包只做 CPU 证据找回、源码谱系核验和协议审计，没有执行 GPU，也没有修改生产代码、论文、图或 Overleaf。目标是判定既有 raw 能否回答正式问题：**Qwen3-8B、NFA vs dense FA2、32K/64K/128K × B1/B4 × eager/Graph、512-token decode、5 个固定输入 × 3 次独立重复**。

## 重生成

在研究仓根目录运行：

```bash
python3 agent_doc/results/nfa_p0_performance_coverage_20261108/build_coverage_audit.py
sha256sum -c agent_doc/results/nfa_p0_performance_coverage_20261108/SHA256SUMS
```

脚本不需要 PyTorch，不启动模型。它从已归档的 9 个 ABBA manifest、68 个逐 run JSON/log 和监控文件重建结果，独立检查：

- manifest 均为 `complete`、`gpu_hygiene=clean`，监控的 `unknown_apps` 全为空；
- 每个 execution 都按 baseline A → candidate A → candidate B → baseline B 排列，68/68 returncode=0；
- 逐 run 的输入 SHA、sample0、B1/B4、eager/Graph、512 tokens、3 warmup、latency-only 和完整 generated IDs；
- tail32、K1=128、K2=1024、sink4、SWA128、near pages32、near length4096、dynamic gate epsilon=0.05；
- 每步 TPOT 的 steady median、ABBA 汇总 median、repeat 和 old/new NFA token identity；
- 五个源码锚相对冻结审计基线的祖先关系，以及限定生产路径的实际 diff。

## 文件

- `coverage.csv` / `coverage.jsonl`：24 个目标“方法×长度×batch×execution”单元；每行记录目标协议、已有证据范围、源码、输入/repeat、raw 路径与 SHA，以及 `rerun` 原因。raw 路径均相对研究仓根目录。
- `run_inventory.jsonl`：68 个历史 NFA ABBA run 的逐 run 索引；其中 raw 路径相对本包目录，并另存 JSON、log、监控和完整 token 数组 SHA。
- `source_lineage.json`：源码祖先关系及生产路径 diff，不沿用无输出的链式命令推断。
- `audit_summary.json`：机器可读的覆盖计数、排除项与限制。
- `manifest.json` / `metadata.json`：正式协议、源码锚、上游证据 SHA、raw 来源和统计口径。
- `raw/`：9 组原目录的 manifest、逐 run JSON/log、`gpu_monitor.jsonl`，不是重新生成的占位数据。
- `SHA256SUMS`：本包除自身外全部文件的完整性清单。

## 审计结论

1. **正式可复用单元为 0/24，24/24 仍需补测。** 这不是说没有 raw，而是现有 raw 不满足完整的同输入 NFA/FA2 ABBA 与 5×3 协议。
2. **12/12 个 NFA 目标单元都有 512-token、sample0 的单输入历史 raw。** 其中 6 个与当前生产语义一致：32K B1、64K B4、128K B4 的 eager/Graph；其余 6 个只有 iter7 selector 语义。32K B1 eager 另有一次独立复跑，因此该单元保存 4 个当前候选测量，而不是擅自合并成正式 3-repeat 结果。
3. `c5be4f69` 到 `4bc6aa98` 只改变直接 Top-K benchmark 和记录；在 `sparse_attn`、完整 e2e runner 与 NFA/FA2 driver 的限定路径上无生产 diff。`4bc6aa98` 到审计基线同样无这些路径的 diff。`cc7d6aa4 → c5be4f69` 改了生产 selector，因此 iter7 结果只作历史 sanity。
4. **FA2 的既有全矩阵只有 128-token 汇总。** `nfa_decode_main_verdict.json` 的 hygiene 和 eager/Graph token identity 可保留为历史参考，但不能与 512-token NFA 数字拼成新倍率。
5. **FlashInfer 不是 dense FA2。** `nfa_flashinfer_20261104/e2e/ruler32k.json` 是真实 512-token ABBA 基础设施样板，但方法不符，不能填入本表。NFA-old vs NFA-new 的优化对照同理不能充当 NFA/FA2。
6. 下一阶段只执行 `coverage.*` 中 `classification=C` 的正式单元。每个固定输入至少 3 次独立重复，ABBA 两腿和重复轮次分别保存；新实验必须重新记录实际 HEAD+diff、输入和模型身份、逐步 TPOT、完整 generated IDs 与 `gpu_hygiene`。既有 sanity raw 用于正确性对照和异常排查，不用于减少正式配对要求。
