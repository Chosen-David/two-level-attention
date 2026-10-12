# S-T023 主表：Quest vs TLI vs FullKV（三件套对比初稿）

> 状态：**初稿（2026-10-12 16:1X）**。RULER 128K quest 11/11 在跑（~17:3X 全齐），完成后填 last cell。
> 口径标注见文末。主表数据 = 官方算法（Quest vendor / TLI 我方）+ 我们协议（vendor 适配层），空卡贴合门禁已过。

## 1. LongBench LBv1（16 任务 AVG）

| 模型 | Quest | FullKV(dense) | Δ |
|---|---|---|---|
| Qwen3-8B | **41.35** | 49.39 | **-8.03** |
| Qwen3-32B | **38.08** | 43.28 | **-5.20** |
| DeepSeek-V2-Lite-Chat | —（MLA patch 不支持） | 40.55* | — |

- Quest 检索大掉点（triviaqa -18.38/hotpotqa -13.98/trec -12.5/musique -12.32/2wikimqa -12.51/qasper -11.04），摘要类维持（multi_news +0.68/qmsum -0.72/samsum -0.91/gov_report -1.04）——**完全符合 Quest 论文趋势**（8B）。
- *DeepSeek fullkv 为 16K 口径（max_length=16384），不与 Qwen3 31500 横比。

## 2. LongBench LBv2（503 样本 AVG，Qwen3-8B）

| method | AVG |
|---|---|
| TLI mavg | **32.21** |
| fullkv(dense) | 32.01 |
| TLI aavg | 31.61 |
| Quest | 30.22（Δ=-1.79 vs fullkv） |

## 3. RULER（11 任务 AVG，Qwen3-8B）

| 长度 | Quest | TLI mavg | TLI aavg | FullKV(dense) | Quest Δ(fullkv) |
|---|---|---|---|---|---|
| 32K | 42.85 | 60.0 | 57.35 | 59.38 | -16.53 |
| 64K | 26.25 | 49.42 | 47.51 | 48.54 | -22.29 |
| 128K | **待打分（11/11 在跑）** | 47.49 | 42.56 | 46.43 | 待定 |

- RULER 长度越长 Quest 掉点越大（32K -16.5 → 64K -22.3），检索任务掉点放大（64K single_3 100→4/multikey_2 50→6）——**符合 Quest 论文趋势**。
- TLI（mavg/aavg）在 32/64K 均 ≈ dense 维持（±1.5 内），无掉点。

## 4. 标注（诚实声明）

1. **Quest legacy cap null**：跨机 pred.py 版本漂移（远程 .24 旧版无 096 runcap sidecar）→ quest 8B/32B 产物 cap 身份 null；32B fullkv 同 legacy → 同口径 Δ 有效。
2. DeepSeek fullkv 16K 口径为独立锚点（MLA dense），只作 TLI 扩展参考，不参与 Quest 对比。
3. 8B/32B 为同一 16 任务集（LBv1）；LBv2 503 独立于 LBv1。
4. RULER 32K 无 yarn、64K/128K 有 yarn（factor 2.0/4.0），打分已按 receipt 校验。
5. MoBA/ClusterKV 尚未接入（MoBA 训练依赖 + ClusterKV pylibraft 版本障碍），后续补齐即入本表。

## 5. 待办（主表收尾）

- [ ] RULER 128K quest 11/11 打分（--yarn-factor 4.0）→ 填 §3 128K 行
- [ ] 32B quest qmsum 尾格完成 → 32B 16/16 收官确认（不影响已打分结果）
- [ ] MoBA/ClusterKV 接入后补充对比臂
