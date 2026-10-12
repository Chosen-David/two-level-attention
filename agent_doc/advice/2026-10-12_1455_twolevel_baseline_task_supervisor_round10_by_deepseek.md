# twolevel 基线任务监督 round10（by deepseek）

- 时间：2026-10-12 14:55
- 范围：round10 例行审查（RULER quest 128K 进度盘点 + .251 single_1/2 完成回收）
- 结论：**.251 128K single_1/2 完成（DONE rc=0）→ rsync 回本地，128K quest 2/11 产物落地；其余 9 格推进正常；无失败、无 bug。**

## 1. RULER quest 128K 进度盘点（11/11 覆盖）

| 机 | 卡 | 任务 | 进度 | 预计完成 |
|---|---|---|---|---|
| .251 | 0/1 | single_1/2 | **100/100 ✓（14:29-14:30 DONE rc=0）** | 已回收 2 卡 |
| 本地 | 0/1 | single_3/multikey_1 | 39/100 | ~16:2X |
| .187 | 1/4 | multikey_2/3 | 28→~40% | ~16:0X |
| .187 | 7 | multiquery | 19→~30% | ~16:0X |
| .24 | 0-3 | multivalue/cwe/fwe/vt | 14-15/100 | ~17:2X |

- 128K 产物落地：本地 L131072/pred 已有 single_1/2 完整（gen 目录 + tli_gen 指针 + yarn_receipt），
  single_3/multikey_1 为 partial gen 目录（边跑边写，完成才切指针）。
- 最慢瓶颈 = .24 4 任务（~17:2X）→ 128K 11 格全齐约在 ~17:3X。

## 2. 本轮动作

- **.251 128K single_1/2 回收**：99/100 观察 → DONE rc=0 → rsync 回本地
  （L131072/pred，gen 目录 + yarn_receipt + tli_gen 全量，72KB）→ 2 卡空置。
- **.187 32B quest qmsum 尾格**（035844 批 card3）：58%（116/200，~43s/it，还需 ~1h）→
  完成后 32B quest 全齐（qmsum 为双跑格，053150 已有，不影响 32B quest 16/16 结果）。

## 3. 无失败、无新 bug

- .24 128K 4 任务启动正常（14-15/100，~108s/it）；.187 multikey_2/3 + multiquery 正常；本地 2 格正常。

## 4. 下一步

1. 128K 全齐（~17:3X）→ rsync 全部产物回本地 + 打分（--yarn-factor 4.0）对拍 e119 128K 锚点
   （fullkv 46.8/mavg/aavg）。
2. .187 32B qmsum 尾格完成 → 32B quest 16/16 收官确认。
3. RULER 三件套（32/64/128K quest vs TLI vs fullkv）齐 → 主表拼接。

— by deepseek（.251 128K single_1/2 完成回收 + 128K quest 2/11 落地）
