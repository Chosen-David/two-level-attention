# Q0 三臂输入冻结与复用审计

本包只做 CPU 证据审计和官方重评分，没有执行 GPU，也没有修改论文、图或 Overleaf。目标是固定 HotpotQA 200、GovReport 200、RULER 64K `niah_multikey_2` 100 的输入身份，再判定 dense FullKV、旧 seed-on NFA、当前 strict-skip NFA 哪些 raw 可以复用。

## 重生成

在研究仓根目录运行：

```bash
/home/wangyuanshuo02/two-level-attention/.venv/bin/python \
  agent_doc/results/nfa_q0_reuse_audit_20261108/build_reuse_audit.py
```

脚本会校验三个原始数据文件的 SHA256、500 行唯一键、FullKV 的逐行 answers/length（RULER 另含 `_id`）、5 个 smoke 输入 SHA，以及官方重评分均值与原汇总的一致性。所有仓库内路径均相对研究仓根目录。

## 文件

- `target_inputs.jsonl`：500 个冻结目标；包含 dataset/row、原始记录 SHA、answers SHA、任务生成预算和可获得的输入 token 身份。RULER 另存输入文本 SHA；LongBench 的 `b_final_*` 字段只用于绑定输入身份。
- `fullkv_quality.jsonl`：既有 dense FullKV 的 500 条预测与逐样本官方重评分；保留 raw 文件路径和 SHA。`generated_token_ids=null` 是真实缺口，不用文本反推 token。
- `reuse_decisions.jsonl`：500 行逐目标、逐臂的 `reuse / rescore / rerun` 判定及原因。
- `audit_summary.json`：机器可读的动作计数、重评分均值和限制。
- `manifest.json`：目标三臂协议、生成预算和全部输入/scorer/manifest 的 SHA256。
- `SHA256SUMS`：本目录派生文件与脚本的完整性清单。

## 目标协议

- HotpotQA `max_new_tokens=32`，GovReport 512，RULER MultiKey-2 128；来自现有任务配置，不把 B_final 的 GovReport 32-token 区域评分协议改称为正式 Q0 质量协议。
- 当前 strict-skip：sink4、SWA128、K1=128、K2=1024、near pages=32、near length=4096、page=16；第一次 decode forward 起生效并冻结，物理 KV retain，不含旧首步 seed/core-token 补偿。
- 旧 seed-on 臂固定到 `f5f3df3c4fe4c4af39c9074db0baaa5e00770eab`；当前 strict-skip smoke 固定到 `2f702c222186bdf17de996e981c52af467957ee5`。正式补测仍须另存实际 HEAD、diff 和环境。

## 审计结论与边界

1. **FullKV：500/500 可重评分，0/500 可直接用于完整 token 桥接。** 400 条 LongBench 的 answers/length 和 100 条 RULER 的 `_id`/answers/length 均与原始数据一致，官方均值为 HotpotQA 53.4784767875、GovReport 33.1737066802、MultiKey-2 50.0。旧 raw 没有完整 generated token IDs，也没有闭合 effective cap/model revision；LongBench 还没有逐样本 input SHA。因此 `fullkv_quality.jsonl` 是 legacy-partial 质量证据，不能恢复 token 级首个分叉。
2. **seed-on 与 strict-skip：各可复用 5 条 HotpotQA smoke，剩余各 495 条需补。** 这 5 行的输入 SHA 与冻结目标一致，使用 `dual_0.05` 臂并保存旧/新生成 token；不能外推为全量质量结论。
3. **B_final 不重复充当 Q0 输出。** 它在此只提供 LongBench 输入身份；其 `gate=full`、旧 `reclaim=true` 和 GovReport 32-token 结果均不改称当前 strict-skip 或正式 Q0 输出。
4. 下一阶段只补 `reuse_decisions.jsonl` 标记为 `rerun` 的单元。任何新补测必须保存完整 generated IDs、预测、逐题分数、输入身份和实际生成协议；不得通过文本截断伪造 token 对齐。
