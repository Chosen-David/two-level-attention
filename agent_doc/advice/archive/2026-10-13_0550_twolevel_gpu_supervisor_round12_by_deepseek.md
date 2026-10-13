# round12 GPU 监督：MInference LBv1 收尾 + RULER 32K sparse prefill 臂派单

> 作者：deepseek · 2026-10-13 05:50 · 代码 bug 监督器（用户 2026-10-11 18:18 指令，2h 一轮）

## 1. GPU 占用表（05:5X）

| 机 | 卡 | 任务 | 状态 |
|---|---|---|---|
| 本地 | GPU0/1 | RULER 32K MInference（cwe→multiquery / fwe→multivalue 串行） | 99% 跑中 |
| .187 | GPU0-3 | MInference LBv1（narrativeqa/hotpotqa 等 200 样本） | 跑中，陆续收尾 |
| .187 | GPU4 | MInference LBv2（503 MCQ） | 98% 跑中，~65min |
| .187 | GPU5/6/7 | RULER 32K MInference（single_1/2/3→multikey_1/2/3 串行） | 97-100% 跑中 |
| .251 | GPU0/1 | MInference LBv1（lcc/repobench-p 500 样本） | 跑中，~06:0X 完成 |

**在飞主线**：MInference 三线并行——LBv1 16 任务（.187 12+本地 2+.251 2）、LBv2 503（.187 GPU4）、RULER 32K 10 任务（本地 2+.187 3）。全部 method=minference_vendor，Qwen3-8B。

## 2. 发现并修复

**① .187 旧 pred_ruler.py 致 RULER 派单崩（rc=0 假成功）**
- 现象：.187 GPU5/6/7 RULER 派单崩 `ValueError: not support minference_vendor`（patch.py register_patch），
  脚本记 rc=0 假成功。
- 根因：05:14 首轮 rsync 只同步了 pred.py（LongBench），**pred_ruler.py 未同步**——本地刚加的
  minference_vendor 分支没到远程。旧 pred_ruler 的 else 分支走 register_patch → 不认该 method。
- 修复：rsync 本地 pred_ruler.py 到 .187/.251（md5 b950c107 三机一致）→ kill 残留 → 重派。
  验证：GPU5/6/7 97-100% 跑中（single_1/2/3 48GB 显存=32K prefill 正常）。
- 教训（跨机派单教训的又一次强化）：**新 method 分支必须连带同步入口脚本（pred.py/pred_ruler.py）+ 依赖**，
  不能只同步 common 文件；崩后 rc=0 假成功必须用 GPU util 核对。

**②（round15 已修，本轮确认无复发）**：pred.py makedirs 并发竞态 + repobench 任务名。

## 3. 代码工作

**pred_ruler.py 加 minference_vendor 分支**（commit e8e27d6 push bee main）：
- 与 LongBench pred.py 同口径：`enable_minference(is_search=False, config_path=..., starting_layer=...)`，
  args 走 add_sparse_attn_args 已有 `--minference_config_path/--minference_starting_layer`。
- 本地 smoke 验证通过：RULER 32K niah_single_1 3 样本 23s（8.4s/it），产物 + yarn_receipt +
  .tli_gen 指针落盘 → **机制有效，RULER sparse prefill 臂打通**。

## 4. 派单记录

| 时间 | 动作 |
|---|---|
| 05:4X | MInference LBv2 → .187 GPU4（pred_minf_lbv2，补齐 LBv2 四件套缺的 minf 臂） |
| 05:4X | RULER 32K MInference smoke（本地 GPU0，3 样本）→ 通过 |
| 05:4X | RULER 32K 10 任务派单：本地 GPU0/1（cwe/fwe + multiquery/multivalue）+ .187 GPU5/6/7（single_1/2/3 + multikey_1/2/3） |
| 05:5X | rsync RULER 32768 数据 → .251（11 文件，供 vt 等补派） |

## 5. 下一步

1. RULER 32K 剩 **vt** 待补（.251 lcc/repobench-p 完成后派 GPU0，或 .187 空卡）。
2. LBv1 200 样本任务（本地已 200/200、.187 card0-3）完成后 rsync 回本地打分：
   - LBv1 MInference 16 任务 AVG 对拍 Quest 41.35 / fullkv 49.39（预期近无损 ≈ fullkv）
   - LBv2 MInference 对拍 LBv2 四件套（mavg 32.21/aavg 31.61/quest 30.22/fullkv 32.01）
   - RULER 32K MInference 对拍 quest 42.85 / fullkv 59.38 / mavg 60.0（sparse prefill 长序列趋势）
3. RULER 64K/128K MInference 臂（sparse prefill 随长度掉点趋势，对拍 quest 64K/128K）。

## 6. 诚实声明

- RULER 32K MInference 已派 10/11 任务 + smoke 通过，但正式产物未齐（~12min/任务）；
  vt 未派。本轮报告基于派单+机制验证，正式打分留产物齐后。
- 长序列 prefill 显存占用正常（32K 单卡 48GB），128K 需验证显存是否吃紧（单卡 143GB 充裕）。
