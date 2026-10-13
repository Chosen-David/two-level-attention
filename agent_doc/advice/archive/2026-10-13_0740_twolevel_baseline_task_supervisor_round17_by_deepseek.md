# S-T023 任务监督器 round17 报告（2026-10-13 07:40，by deepseek）

## 一句话
128K MInference 全量 11 任务三机在飞（~1/3 进度）；补派 **RULER 128K quest 臂缺口**（.187 GPU7 先行验证，yarn 4.0 生效）；32B pattern 搜索脚本就绪；ClusterKV 硬阻塞复查确认。

## 本轮完成动作

### 1. 128K MInference 在飞进度确认（约 08:05 快照）
| 机器/任务 | gen 行数/100 | 预计完成 |
|---|---|---|
| .187 single_1（先行 07:10） | 48 | ~09:05 |
| .187 single_2（先行 07:10） | 48 | ~09:05 |
| .187 cwe/fwe/mk1/mk2/mk3（07:18） | 31-33 | ~09:30 |
| 本地 multiquery/multivalue（07:17） | 31-32 | ~09:30 |
| .251 single_3/vt（07:18） | 30 左右 | ~09:30 |
- 实测 ~68s/it（比 24.6s/it 慢——128K 长样本推理），非预估 41min/任务，实际 ~80-100min/任务

### 2. 补派 RULER 128K quest 臂（发现缺口，实质推进）
- **缺口**：exp/trace/results 下 e119_ruler128k_formal_* 只有 fullkv/mavg/aavg，**quest_vendor 128K 缺失**（Quest 长序列掉点关键数据）
- **派单**：.187 GPU7 先行验证 niah_single_1（pid 2493121，--yarn --yarn_factor 4.0 --quest_token_budget 1024 --quest_chunk_size 16）
- **确认生效**：log 实证 `[yarn] factor=4.0 (effective context=163840)` + `task=niah_single_1 L=131072 method=quest_vendor`，GPU7 100% 53GB
- 先行通过后批量派 quest 128K 剩余 10 任务（对拍 quest 32K 42.85/64K 26.25 的掉点趋势）

### 3. 32B pattern 搜索脚本就绪（32B 臂前置）
- 确认本地 Qwen3-32B 模型齐全（/home/wangyuanshuo02/models/huggingface.co/Qwen/Qwen3-32B，17 shards）
- 已写 `exp/vendor_baselines/minference_vendor/dispatch_search_32b_local.sh`（qwen3_minference_search.py --model 32B --seq-len 8192 → best_pattern_qwen3_32b.json）
- 待本地 128K 完成（~09:30）后执行 → 32B 三口径（LBv1/LBv2/RULER）派单

### 4. ClusterKV 硬阻塞复查（无进展，确认死锁）
- 清华镜像 `pip index versions pylibraft-cu12` 能列 24.10.0，但 `--only-binary` 实测**无 wheel**（24.10.0 仅 sdist）
- 25.6+ 有 wheel 但需 python>=3.11 且**移除 cluster API**（round11 记忆）→ 无法满足 ClusterKV
- 确认硬阻塞属实：24.10.0 必须源码编译 → 依赖 github raft → 本机 github 大传输超时。**建议后续走 conda-forge/内网镜像或 proxy 再解**

## sanity 对拍硬门禁状态
- MInference 三口径已过（LBv1 47.71 Δ-1.68 / LBv2 31.21 Δ-0.80 近无损；RULER 32K 49.94 / 64K 50.29 vs fullkv 48.54，掉点/持平在论文稀疏 prefill 范围内）
- 待 128K MInference + quest 128K 完成后补齐长度趋势三档表

## 在飞 / 待办
- **128K MInference 11 任务**：~09:30-10:00 全齐 → score_ruler_formal --yarn --yarn-factor 4.0 打分
- **quest 128K**：GPU7 先行 → 批量 10 任务
- **32B pattern 搜索**：本地 128K 完成后执行 → 32B 全量三口径
- **ClusterKV**：阻塞（需镜像/代理解 raft 源码依赖）
- RULER 128K 各臂打分齐后整理 32/64/128K 三档对比表（minf/fullkv/quest/mavg/aavg 五臂）

## 资源状态
- .187 GPU0-6：128K MInference 7 任务满载；GPU7：quest 128K 先行
- 本地 GPU0/1：128K multiquery/multivalue 满载
- .251 GPU0/1：128K single_3/vt 满载
- **三机 12 卡全满载**，无空闲
