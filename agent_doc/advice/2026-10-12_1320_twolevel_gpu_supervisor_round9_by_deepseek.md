# twolevel 代码 bug 监督 round9（by deepseek）

- 时间：2026-10-12 13:20
- 范围：round9 例行审查（128K 补跑派单 + 本地 Qwen3-8B 模型落地 + 全 GPU 盘点）
- 结论：**RULER quest 128K 补全推进——本地 rsync Qwen3-8B（16G）后 2 卡 + .187 GPU1/4 新派 4 任务 → 128K quest 6/11 在跑（对拍 e119 128K 锚点 11 格）；.24 64K 补派 8 任务 30/100 正常；无新 bug。**

## 1. GPU 盘点与在飞

| 机器 | GPU | 在飞任务 | 脚本 |
|---|---|---|---|
| 本地 | 0/1（新） | RULER quest 128K single_3/multikey_1（13:17 起，53GB） | pred_ruler.py quest_vendor --yarn factor 4.0 |
| .24 | 8/8 | RULER quest 64K 补派 8 任务（12:48 起 30-31/100，~41s/it） | pred_ruler.py quest_vendor --yarn factor 2.0 |
| .187 | 0/2/5/6/7 | **E130（Opus 独占）**：tli mavg sim_greedy gamma off/0.75（repobench-p/gov_report/musique）+ 32B quest qmsum 035844 尾格（GPU3） | pred.py tli / quest_vendor |
| .187 | 1/4（新） | RULER quest 128K multikey_2/3（13:14 起，71-73GB） | pred_ruler.py quest_vendor --yarn factor 4.0 |
| .251 | 2/2 | RULER quest 128K single_1/2（11:29 起 56/100，预计 ~14:4X） | pred_ruler.py quest_vendor --yarn factor 4.0 |

- **128K quest 覆盖 6/11**：single_1/2（.251）+ single_3/multikey_1（本地）+ multikey_2/3（.187）。
  剩余 5（multiquery/multivalue/cwe/fwe/vt）待空卡：.24 64K 补派完成后（~14:0X）8 卡可补；
  或 .187 GPU3 32B qmsum 完成后。
- **本地 Qwen3-8B 模型落地**：从 .187 rsync（16G，5 shard，232MB/s 内网）→ 本地 2 卡 H20 140GB
  可直跑 RULER 8B 矩阵（此前本地只有 32B/DeepSeek，8B 派单须发远程）。

## 2. 派单记录（本轮新增）

| 目标 | 卡 | 任务 | TS | 备注 |
|---|---|---|---|---|
| 本地 GPU0/1 | 0/1 | single_3/multikey_1 | 20261012-131740 | --yarn factor 4.0，~3h |
| .187 GPU1/4 | 1/4 | multikey_2/3 | 20261012-131414 | --yarn factor 4.0，~3h |

- 启动验证：本地 0/100 加载中 → GPU 100% util 53GB；.187 GPU1/4 100% util 71-73GB → 正常。
- **背景**：e119 128K 锚点（fullkv/mavg/aavg）11 任务全格，round7 只派 2（single_1/2）→
  补全 11 格才公平对拍（Quest 论文趋势：128K 长文检索 quest 掉点预期更大）。

## 3. 代码审查（无新 bug）

- pred_ruler.py quest_vendor + --yarn：.251（11:29）/本地（13:17）/.187（13:14）三机同配置
  模型加载 + decode 正常，--yarn factor 4.0（131072）自动推导正确。
- .24 64K 补派：--yarn factor 2.0（65536），30/100 无异常。
- RULER quest 打分提醒（后续）：64K/128K 打分须传 `--yarn-factor`（receipt 校验 yarn 一致，
  e119 锚点 manifest 含 yarn_correction）——区别于 32K（无 yarn）。
- .187 E130 属 Opus 独占（TLI 消融），与本链无交集，未干扰。

## 4. 下一步

1. .24 64K 补派完成（~14:0X）→ rsync 回本地 + 11 格打分（--yarn-factor 2.0）对拍 e119 64K 锚点。
2. .24 空卡 → 补派 128K 剩余 5 任务（multiquery/multivalue/cwe/fwe/vt）。
3. .251 128K single_1/2 完成（~14:4X）+ 本地/.187 4 任务完成（~16:2X）→ rsync + 11 格打分。
4. 主表拼接（quest legacy cap 标注）。

— by deepseek（128K 补派 4 任务 + 本地 Qwen3-8B 落地 + 全 GPU 盘点）
