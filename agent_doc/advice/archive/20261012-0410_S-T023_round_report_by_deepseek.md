# S-T023 对比基线 round 汇报（by deepseek）

- 时间：2026-10-12 04:1X
- 触发：32B quest 发现真 bug（.187 info.py 缺 quest_vendor 分支 → 全任务白跑）+ 修复重派 + trec 预算 Pareto 全收敛
- 结论：**.187 旧代码缺 quest_vendor 分支导致 32B quest 全部任务在写文件前崩（pred 空、rc 假 0），已 rsync 修复并重派 14 任务；trec 预算 Pareto 全收敛（512=53.5 < 1024=59.0 < 32768=72.5 ≈ fullkv=71.5），Quest 核心结论在 Qwen3-8B 双任务确证。**

---

## 1. 真 bug：.187 info.py 缺 quest_vendor 分支（32B quest 全灭）

- **现象**：.187 32B quest（TS=20261012-004348）跑了一夜，pred 目录 0 个文件；日志大量
  `Traceback ... pred.py:503 get_method_name_with_info ... sparse_attn/info.py:514 raise ValueError`，
  但 `done rc=0` **假成功**。
- **根因**（源码核验）：
  - .187 的 `sparse_attn/info.py` 是**旧版**，method 分支只有 quest/twi/tia/tli/none，
    **无 quest_vendor 分支** → `--method quest_vendor` 走 else `raise ValueError`。
  - .187 的 pred.py 在任务 forward 完成后、**写 pred 文件之前**调用
    `get_method_name_with_info`（out_fn 命名）→ 崩 → 文件不写、任务白跑。
  - 本地/.24/.251 的 info.py 是**新版**（有 quest_vendor 分支）→ 8B quest 正常。
  - rc=0 假成功：ValueError 被 pred.py 吞（traceback 打印后正常退出码）→ 派单脚本无法
    检测失败，整整一夜 8 卡空转。
- **修复**：`scp 本地 info.py → .187`（md5 66b1b90ec250ae38d7473b1e25da4843，含
  quest_vendor 分支）；kill 旧派单进程树（4 python + dispatch 脚本）；重派 14 任务
  （narrativeqa/qasper 本地 TS=20261012-002743 已有）到 .187 卡 2/3/4/6/7（避开 8B
  任务占用的 0/1/5），每卡 2-3 任务。
- **验证**：重派 5 进程正常推进无 ValueError（passage_count 7/50 最快）；首个任务
  完成后确认 pred 文件落盘。

## 2. trec 预算 Pareto 全收敛（Qwen3-8B，Quest 核心结论双任务确证）

| token_budget | qasper | trec |
|---|---|---|
| 512 | 29.70 | 53.5 |
| 1024 | 36.43 | 59.0 |
| 32768 | 46.57 | 72.5 |
| fullkv | 47.47 | 71.5 |

- 双任务均 **512 < 1024 < 32768 ≈ fullkv** 严格单调 → Quest（budget↑ → acc↑）确证。
- 全预算 sanity 两任务全过（qasper Δ=-0.90、trec Δ=+1.0）→ **quest_vendor 适配层无 bug
  判定彻底完成**，8B quest 16 任务 AVG 41.35 上主表条件齐备。

## 3. 资源布局

| 机 | 卡 | 任务 |
|---|---|---|
| .24 | 0/3/4/6/7 | DeepSeek fullkv 并行收尾（**12/16**，剩 gov_report/lcc/repobench-p/samsum） |
| .24 | 2 | trec512 sanity（已完成 53.5，卡空闲） |
| .187 | 2/3/4/6/7 | **32B quest 重派 14 任务**（info.py 修复后，每任务 6-8h） |
| .187 | 0/1/5 | 8B 任务（S-T019/22）占用 |
| 本地 | 2 | 32B narrativeqa/qasper 已有（TS=20261012-002743） |

## 4. 待办

1. **DeepSeek fullkv 16/16 完成** → eval 打分（MLA dense 锚点）
2. **32B quest 14 任务**（预计 ~20h）→ 完成后与本地 narrativeqa/qasper 合并 → 打分 + 空卡贴合
3. 主表拼接：quest 8B 41.35 + fullkv 锚点 + DeepSeek
4. 教训入库：**跨机派单前必须核对 info.py/pred.py md5 一致**（本次 .187 旧代码漏同步）

— by deepseek（源码核验：.187 vs 本地 info.py method 分支对比；修复后重派验证）
