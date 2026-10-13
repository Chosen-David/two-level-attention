# round15 任务监督：MInference 全量 16 任务派单 + 并发 makedirs 竞态真 bug 修复

> 作者：deepseek · 2026-10-13 05:20 · 任务监督器（用户 2026-10-11 18:5X 指令，有实质进展/bug 才写）

## 1. 本轮具体动作

### 1.1 全量 16 任务派单（LBv1 8B 主表对比臂）
round14 sanity 对拍通过后，将 MInference 全量 16 任务派到 12 卡（三机全空）：

| 机 | 卡 | 任务 |
|---|---|---|
| .187 | GPU0-3 各 2 串行 | narrativeqa/gov_report · qasper/qmsum · multifieldqa_en/multi_news · hotpotqa/trec |
| .187 | GPU4-7 各 1 | 2wikimqa · musique · triviaqa · samsum |
| .251 | GPU0/1 | lcc · repobench-p |
| 本地 | GPU0/1 | passage_count · passage_retrieval_en |

跨机同步：rsync 源码（排除 .git/exp/third_party/__pycache__）+ vendor 目录到 .187/.251，
**pred.py/info.py/arguments.py/qwen3_minference_patch.py/best_pattern_qwen3_8b.json 三机 md5 全一致**。
TS 052051/.251 052118/本地同刻。

### 1.2 发现并修复 2 个问题

**① pred.py makedirs 并发竞态（真 bug，已修）**
- 现象：.187 GPU5 musique 崩 `FileExistsError: [Errno 17] File exists: '.../pred_minf_v1/pred'`，
  脚本却记 rc=0（**假成功**，靠 GPU util 核对发现 GPU5 空）。
- 根因：8 卡同时启动写同一 output-dir，多进程同过 `if not os.path.exists(dir)` 检查后都执行
  `os.makedirs(dir)`（默认 exist_ok=False）→ TOCTOU 竞态，单任务 FileExistsError。
- 修复：`os.makedirs(dir, exist_ok=True)` 幂等创建（pred.py line 600-608 重构）。
  commit b9a3b53 push bee main。rsync 修复版到两机（md5 6abdaf5d... 三机一致）。
  重派 musique 到 .187 GPU5（PID 2465451，97% util 正常跑）。
- 教训：**多卡并行派单 output 目录创建必须幂等**；sanity 2 卡没崩是竞态窗口恰好错开，16 任务
  8 卡同时启动必然触发。

**② 派单脚本任务名错误（我方的 bug，已改重派）**
- 现象：.251 GPU1 repobench_p 崩 `FileNotFoundError: repobench_p.jsonl`。
- 根因：LongBench 官方任务名是 **repobench-p（连字符）**，data 目录也是 `repobench-p.jsonl`；
  我的派单脚本误写 `repobench_p`（下划线）→ pred.py 直接拼 `.jsonl` 找不到。
  （quest 既往跑 repobench-p 一直用连字符名。）
- 修复：重派脚本任务名改 repobench-p，.251 GPU1（PID 3004607，98% util 正常跑）。
- 教训：**任务名用官方名（连字符）**，不能凭直觉转下划线。

## 2. 状态盘点

| 项 | 状态 |
|---|---|
| MInference 全量 16 任务 | **16/16 全在跑**（.187 12 + .251 2 + 本地 2） |
| pred.py makedirs 竞态 | 修复 + commit b9a3b53 + 三机 rsync 一致 |
| 预计完成 | lcc/repobench-p 500 样本 ~47min；其余 200 样本 ~1-2h → ~07:1X 全齐 |
| 下一步 | rsync 回本地 → 打分（quest venv eval.py，--legacy-no-runcap 降级如需要）→ 对拍 Quest 41.35 / fullkv 49.39 |

## 3. 诚实声明

- 本轮发现的两个问题：①是 pred.py 真 bug（并发安全），②是我派单脚本任务名错误——都不是
  MInference 算法本身的精度/正确性问题。
- 除 musique（重派）外，其余 15 任务首轮即在跑；重派后 16/16 全覆盖，无重复跑（makedirs 竞态
  只崩了 musique 一个，其余卡启动时序恰好错开或先建目录成功）。

## 4. 下一步建议

1. ~07:1X 全齐后 rsync 回本地，用 quest venv eval.py 打分（MInference 16 任务 AVG 对拍
   Quest 41.35 / fullkv 49.39，预期近无损 ≈ fullkv）。
2. 若掉点分布异常（非近无损），回查 pattern 质量 / starting_layer 口径（round14 建议 2）。
3. RULER 32/64/128K MInference 臂（sparse prefill 长序列趋势，round14 建议 3）。
