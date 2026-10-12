# twolevel 基线任务监督 round8（by deepseek）

- 时间：2026-10-12 13:35
- 范围：round8 例行审查（RULER quest 128K 补派推进 + 64K 补派进度确认）
- 结论：**.187 GPU7 新空闲 → 派 RULER quest 128K multiquery → 128K quest 7/11 在跑；.24 64K 补派 59-62/100 正常（预计 ~14:0X 完成）；无失败任务。**

## 1. S-T023 三件套进度总览

| 组件 | Quest | FullKV（锚点） | 状态 |
|---|---|---|---|
| 8B LBv1 16 任务 | **AVG 41.35** | AVG 49.39 | ✓ 已上主表资格（sanity 8 任务全预算≈dense 过） |
| 8B LBv2 503 | quest 30.22 | fullkv 32.01（mavg 32.21/aavg 31.61） | ✓ Quest Δ=-1.79 |
| 32B LBv1 16 任务 | **AVG 38.08** | AVG 43.28 | ✓ Quest Δ=-5.20 |
| DeepSeek LBv1 16 任务 | —（MLA patch 不支持） | AVG 40.55（16K 口径） | ✓ fullkv 锚点 |
| RULER 32K 11 任务 | AVG 42.85 | fullkv 59.38 | ✓ Δ≈-16.5 |
| RULER 64K 11 任务 | 补派 8/11 任务 62/100 | fullkv AVG 48.6（11 格） | ~14:0X 完成 → 打分 |
| RULER 128K 11 任务 | **7/11 在跑** | fullkv AVG 46.8（11 格） | ~16:3X 全齐 → 打分 |

## 2. 本轮派单

| 目标 | 卡 | 任务 | TS | 备注 |
|---|---|---|---|---|
| .187 GPU7（新空闲） | 7 | niah_multiquery | 20261012-133052 | --yarn factor 4.0，已验证 100% util 53GB |

- 背景：e119 128K 锚点 11 格全格，128K quest 需补全 11 格才公平对拍。
- 128K quest 覆盖 7/11：single_1/2（.251）+ single_3/multikey_1（本地）+ multikey_2/3（.187 GPU1/4）+ multiquery（.187 GPU7）。
  剩余 4（multivalue/cwe/fwe/vt）已备好 .24 派单脚本（/tmp/ruler_quest_128k_24_full.sh），
  等 .24 64K 补派完成（~14:0X）立即派。

## 3. 在飞任务核实（无失败）

- .24 64K 补派 8 任务 59-62/100（12:48 起，~41s/it，预计 ~14:0X）。
- .251 128K single_1/2 66/100（11:29 起，预计 ~14:35）。
- 本地 128K single_3/multikey_1 5/100（13:17 起，128K 慢 ~2.4min/it，预计 ~17:1X）。
- .187 32B qmsum（035844 尾格）在跑（GPU3 93%），完成后 32B quest 产物全齐。

## 4. 下一步

1. .24 64K 补派完成（~14:0X）→ rsync 回本地 + 11 格打分（--yarn-factor 2.0）对拍 e119 64K 锚点。
2. .24 空卡 → 派 128K 剩余 4 任务（multivalue/cwe/fwe/vt）。
3. .251 128K 完成（~14:35）→ 2 卡可再补（若仍有缺）。
4. 128K 11 格全齐（~16:3X）→ rsync + 打分（--yarn-factor 4.0）对拍 e119 128K 锚点。

— by deepseek（128K quest 7/11 在跑 + 64K 补派进度确认）
