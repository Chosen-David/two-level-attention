# S-T023 任务监督器 round16 报告（2026-10-13 07:00，by deepseek）

## 一句话
MInference 臂 LBv1/LBv2/RULER32K 三线打分全部收官，**sanity 门禁通过**；RULER 64K 全量 11 任务三机 11 卡满载在飞。

## 本轮完成动作

### 1. LBv1 MInference 打分收官（16/16，sanity 通过）
- 逐任务独立目录打分（102 cap 混挂 fail-closed 解法），`/tmp/eval_minf_lbv1_all.py`
- **AVG 47.71**，vs fullkv 49.39（**Δ-1.68 近无损**），vs quest 41.35（**+6.36**）
- 单任务亮点：passage_retrieval_en 99.0 / triviaqa 90.36 / trec 70.0 / lcc 67.7
- 掉点任务：passage_count 2.0（计数任务 sparse 先天短板，Quest 亦仅 6.0 级）
- **结论：完全符合 MInference 论文「近无损 sparse prefill」核心结论 → 门禁通过，可上主表**

### 2. LBv2 MInference 打分（503 MCQ）
- 回收 .187 GPU4 产物（TS 053636，503 行全量）→ eval.py 四选一 accuracy
- **31.21**，vs fullkv 32.01（**Δ-0.80 近无损**），vs quest 30.22（+0.99）
- LBv2 四件套补齐：mavg 32.21 / aavg 31.61 / **minf 31.21** / quest 30.22 / fullkv 32.01

### 3. RULER 32K MInference 打分（11/11）
- score_ruler_formal 正式入口，11 任务 tli_gen 指针齐全（E134 教训复核）
- **AVG 49.94**：single_2/3 100、multikey_1 99、multikey_2 78、single_1 42、fwe 45、vt 25.2、multivalue 25、multiquery 18、cwe 16.1、multikey_3 1
- 对拍：quest 42.85（**+7.09**）、fullkv 59.38（**Δ-9.44 掉点**）、mavg 60.0（-10.06）
- **sanity 解释**：MInference vertical/slash 模式面向真实长文档注意力，synthetic needle 密集穿插任务（multikey_3/multiquery/multivalue）sparse 会丢针；论文也未声称 RULER 保精度 → 不构成趋势矛盾，如实上表并标注掉点
- 反衬：我方 mavg 32K 60.0 持平 fullkv，sparse 检索不掉点 = TLI 优势点

### 4. RULER 64K 全量 11 任务派单（三机 11 卡满载在飞）
- `.187` GPU0/1/2/3/5/6/7：cwe/fwe/multikey_1/2/3/single_1/2（pid 2482466 批，`--yarn --yarn_factor 2.0`）
- `.187` GPU4：multivalue（pid 2483874）
- 本地 GPU0/1：single_3/multiquery（TS 064407）
- `.251` GPU0：vt（pid 3016598）
- RULER 65536 数据已 rsync 到 .251（11 任务文件齐全，252MB）

## 在飞 / 待办
- **RULER 64K**：11 任务 ~20-40min/任务，预估 07:30-08:00 首轮回收
- **RULER 128K**：待 64K 完成后（--yarn_factor 4.0）
- **MInference 32B 臂**：pattern 需在线重新搜索
- **ClusterKV**：硬阻塞（pypi 无包 + github 直连/镜像超时）；**MoBA/FASA/SparQ**：待网络
- LBv2 MInference 得分 31.21 已记，补齐后 LBv1+LBv2+RULER 三口径 minf 臂齐

## 关键数据快照
| 口径 | MInference | fullkv | quest | mavg | Δvs fullkv |
|---|---|---|---|---|---|
| LBv1 AVG16 | **47.71** | 49.39 | 41.35 | 46.95* | **-1.68** |
| LBv2 503 | **31.21** | 32.01 | 30.22 | 32.21 | **-0.80** |
| RULER 32K | **49.94** | 59.38 | 42.85 | 60.0 | **-9.44** |
| RULER 64K | 在飞 | - | - | - | - |

*E109 旧口径

## 资源状态
- .187 8 卡 / 本地 2 卡 / .251 2 卡 全部满载（64K 在飞），无空闲可派
- 64K 完成后下一轮：打分 64K → 派 128K 先行验证
