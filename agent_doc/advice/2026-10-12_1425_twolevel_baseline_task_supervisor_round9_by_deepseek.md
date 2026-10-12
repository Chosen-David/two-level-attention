# twolevel 基线任务监督 round9（by deepseek）

- 时间：2026-10-12 14:25
- 范围：round9 例行审查（RULER quest 64K 打分收官 + 128K 11/11 全覆盖派单）
- 结论：**64K quest 11/11 打分 AVG 26.25（对拍 fullkv Δ=-22.29，完全符合 Quest 论文趋势）；128K quest 11/11 全覆盖在跑；无新 bug。**

## 1. S-T023 三件套进度总览

| 组件 | Quest | FullKV（锚点） | 状态 |
|---|---|---|---|
| 8B LBv1 16 任务 | **AVG 41.35** | AVG 49.39 | ✓ 已上主表资格 |
| 8B LBv2 503 | quest 30.22 | fullkv 32.01（mavg 32.21/aavg 31.61） | ✓ Quest Δ=-1.79 |
| 32B LBv1 16 任务 | **AVG 38.08** | AVG 43.28 | ✓ Quest Δ=-5.20 |
| DeepSeek LBv1 16 任务 | —（MLA 不支持） | AVG 40.55（16K 口径） | ✓ fullkv 锚点 |
| RULER 32K 11 任务 | AVG 42.85 | fullkv 59.38 | ✓ Δ=-16.5 |
| RULER 64K 11 任务 | **AVG 26.25**（本轮） | fullkv 48.54 / mavg 49.42 / aavg 47.51 | ✓ Δ=-22.29 |
| RULER 128K 11 任务 | **11/11 全覆盖在跑** | fullkv AVG 46.8（11 格） | ~16:3X 全齐 → 打分 |

## 2. 本轮动作

1. **64K quest 11/11 打分**（.24 64K 补派 8/8 完成 + rsync 回本地）：
   - staging root `/tmp/e119_64k_quest/L65536`（symlink 到本地产物），`--yarn-factor 2.0`，n=100/格。
   - **AVG 26.25**：single_1 59/single_2 79/single_3 4/multikey_1 71/multikey_2 6/multikey_3 0/
     multiquery 14/multivalue 16.25/cwe 0.8/fwe 9.67/vt 29。
   - 对拍 e119 64K 锚点：fullkv 48.54、mavg 49.42、aavg 47.51 → **Quest Δ=-22.29**
     （32K 时 Δ=-16.5，长度翻倍掉点放大，检索大掉点 single_3 100→4/multikey_2 50→6 等）**符合 Quest 论文趋势 ✓**
   - TLI 64K（mavg 49.42/aavg 47.51）≈ dense 维持，无掉点。落盘 e119_ruler64k_formal_quest_vendor.json。
2. **128K quest 11/11 全覆盖**：.24 8 卡 64K 补派完成后 GPU0-3 空 → 派 128K 缺 4 任务
   （multivalue/cwe/fwe/vt，TS 20261012-1415xx，--yarn factor 4.0）→ 11/11 全在跑。

## 3. 在飞任务核实（无失败）

| 机 | 卡 | 128K 任务 | 进度 |
|---|---|---|---|
| .251 | 0/1 | single_1/2 | 85/100（~14:3X 完成） |
| 本地 | 0/1 | single_3/multikey_1 | 24/100（~16:2X） |
| .187 | 1/4 | multikey_2/3 | 28/100 |
| .187 | 7 | multiquery | 19/100 |
| .24 | 0-3 | multivalue/cwe/fwe/vt | 新启动 |

- .187 32B quest qmsum 尾格（GPU3 93%）→ 32B quest 产物全齐。
- 无失败任务，无新 bug。

## 4. 下一步

1. .251 single_1/2 完成（~14:3X）→ 2 卡回收（可补其他缺口或空置）。
2. 128K 11 格全齐（~16:3X）→ rsync 回本地 + 打分（--yarn-factor 4.0）对拍 e119 128K 锚点
   （fullkv 46.8/mavg/aavg），完成 RULER 三件套（32/64/128K）。
3. 主表拼接（RULER 三件套 quest vs TLI vs fullkv + LBv1 8B/32B + LBv2 + DeepSeek），标注 quest legacy cap null。

— by deepseek（64K quest 11/11 打分 Δ=-22.29 + 128K quest 11/11 全覆盖）
