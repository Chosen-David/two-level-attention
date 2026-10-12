# twolevel GPU 监督 round5（by deepseek）

- 时间：2026-10-12 07:5X
- 范围：S-T023 竞品复现任务链（Quest vendor 8B/32B + 派单延伸）
- 结论：**pred.py 三机版本漂移调查收官——判定无精度 bug（quest_vendor 生成协议三机一致）；但发现 .187 缺 DeepSeek-V2-Lite-Chat 分支（IS_DSV2=0，潜在陷阱）→ 已 sync .24 版本对齐（.187/.24 md5=f7069942 一致）。32B quest 13/16 完成，3 在跑（均有双跑）。新派 LBv2 quest + 全预算 sanity 扩展 4 任务到 .24（正常推进）。**

## 1. pred.py 三机版本漂移调查（本轮主动作）

- **md5**：本地 411de5ad / .187 d098facf / .24 f7069942 三机不一致，逐段 diff 核实。
- **判定（无精度 bug）**：quest_vendor 生成协议（prompt 模板 / samsum 分支 / else 手动 argmax 循环 + B04 EOS 判停）**三机逐字节一致** → Qwen3 quest 数据可比性不受影响。
- **发现潜在陷阱**：.187 pred.py **缺 DeepSeek-V2-Lite-Chat 分支**（grep IS_DSV2 计数=0；本地/.24 计数=4）。DeepSeek 需 model.generate 路径（031e6b8 修复），.187 若派 DeepSeek 任务会走 else 手动循环在 4.56.2 DynamicCache 下崩。
- **修复**：备份 .187 旧版（/tmp/pred_187_backup_*）→ scp .24 版（f7069942，含完整 IS_DSV2 分支）覆盖 .187 → **.187/.24 md5 对齐 f7069942**，py_compile 通过。在跑进程内存已加载不受影响。
- **版本分层记录**：远程 .187/.24 = f7069942（DeepSeek 分支版）；本地 = 411de5ad（+095/096 run_cap_manifest + eager guard）。**本地新版不能直接 sync 远程**——依赖 run_cap_manifest.py（远程缺）与 sparse_attn.ops.eager_decoding guard（远程缺）→ ImportError 崩（已验证）。

## 2. 32B quest 进度（13/16 完成）

| 源 | 完成（行数） | 在跑 |
|---|---|---|
| 本地 | narrativeqa/qasper/passage_count | — |
| .24 053150 | 2wikimqa/trec/triviaqa（200/200/200）| gov_report + qmsum（05:31 起 ~2.5h）|
| .187 035844 | multifieldqa_en(150)/musique/samsum/lcc(500)/hotpotqa/2wikimqa/passage_count/**multi_news(新完成)**（8）| gov_report / repobench-p / passage_retrieval_en(双跑格) |
| .187 g01 | passage_retrieval_en（200）| repobench-p（与主批双跑）|

- 剩 3 个独特任务在跑：gov_report（.24+.187 双跑）、repobench-p（.187 主批+g01 双跑）、qmsum（.24）。预计 1-3h 全部完成。
- 3 双跑（gov_report/passage_count/repobench-p/passage_retrieval_en）由 086 best-file（行数门 200）合并时裁决，无害仅浪费算力。

## 3. 新派单（打满 GPU，出论文数据）

- **发现本地无 Qwen3-8B 模型**（只有 32B/DeepSeek；8B 在 .24/.187 `/home/wangyuanshuo02/Qwen3-8B`）→ 8B 派单发往 .24。
- **.24 GPU0/3/4/5 派 4 任务**（TS 20261012-072618，独立输出目录）：
  - GPU0: **lbv2 quest** 8B（quest 三件套基线，503 样本，14/503 推进 ~3h）
  - GPU3/4/5: **hotpotqa/narrativeqa/musique 全预算（32768）sanity** 8B（"全预算≈dense"证据扩展，hotpotqa 54/200 ~16min 将完成）
- 已确认 4 进程正常推进无崩溃（load shards 完成 + 样本循环推进）。

## 4. GPU 与下一步

- .24：GPU1/2 32B quest + GPU0/3/4/5 新 8B quest + GPU6/7 空闲；.187：5 个 32B quest 进程 + GPU1/5 空闲；本地 GPU0/1 空闲（无 8B 模型，32B quest 3/3 已完成）。
- 下一步：
  1. 32B quest 16/16 齐 → 合并 pred_qv_1024_16{,_g01,_24}/ + 本地 → 086 best-file → eval 打分 + 空卡贴合（对拍 Quest 论文检索掉/摘要维持趋势，sanity 硬门禁）。
  2. LBv2 quest + sanity 扩展完成后打分（eval.py lbv2 scorer）。
  3. 主表拼接：quest 8B 41.35 + fullkv 49.39 + DeepSeek 40.55 + 32B quest（完成后）。

— by deepseek（三机 pred.py md5/diff 实查；.187 缺 DeepSeek 分支定位 + sync 对齐；32B quest 进度逐目录行数实查；.24 派单 4 任务 + 启动验证）
