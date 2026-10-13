# FASA（ICLR 2026）独立复现诊断报告

- 日期：2026-10-13（deepseek 轨，S-T023 竞品复现）
- 基线：**FASA**（wangyifei0047/FASA，frequency-aware sparse attention）
- 接入：B 轨 vendor（官方源码逐字节 + 薄适配层），Phase1 校准 `qwen3_identify_fre.py`
  + `convert_agreement.py` → agreement.pkl；Phase2 在线剪枝 `qwen3_fasa_patch.py`
  （enable_fasa + 官方 `core_module_with_padding_old` 纯 top-k，无 sink/swa 保送）。
- 模型：Qwen3-8B（本地 /home/wangyuanshuo02/Qwen3-8B）

## 1. 核心发现（一句话）

**FASA 的频域细筛（2-D RoPE 64 组）在有效预算 1024 下选择力不足，RULER 32K 崩盘
（AVG 24.00）、LongBench qasper 掉分（34.85 vs fullkv 47.47）；预算放大到 4096 才
近无损（qasper 46.64 ≈ fullkv 47.47）。适配层本身正确（单调性 + 大预算 sanity 双证）。**

## 2. 证据链（逐项排除适配 bug）

| # | 实验 | 结果 | 判定 |
|---|---|---|---|
| 1 | RULER 32K 6 任务×100 打分 | AVG 24.00（s1=0/s2=78/s3=4/mk1=53/mk2=9/mk3=0） | 远低 fullkv 59.38 / mavg 60.0 / quest 42.85 / minf 49.94 |
| 2 | budget 单调性 sanity（niah_single_1） | 1024→0/100、2048→5/10、8192→6/10 | 剪枝正常，单调上升 → 非适配 bug |
| 3 | 校准样本数不敏感 | 2 样本 agreement mean 0.15 vs 8 样本 0.157；s1 均全错 | 校准样本数非主因 |
| 4 | 8 样本校准重跑 niah_single_1×20 | s1=0，生成全模板句（"The grass is green..."）needle 丢失 | 方法在 synthetic needle 失效 |
| 5 | agreement 分布（8 样本） | budget 1024 时 freq-group top-k 交集仅 **13%**；4096 时 43% | 低预算下频域子空间复现力弱 |
| 6 | LongBench qasper 预算单调性 | 1024=34.85 < 4096=46.64 ≈ fullkv=47.47 | **LongBench 上单调性同样成立** → 适配正确 |

## 3. 结论：方法本质 vs 适配 bug

- **适配层正确**：预算单调性在 RULER 与 LongBench 双场景确证；大预算（4096）qasper
  46.64 逼近 fullkv（Δ=-0.83）证明 patch 无错、校准产物有效。
- **方法本质**：FASA 把 head_dim=128 切成 64 个 2-D RoPE 频域组，按离线 agreement 校准
  选 top-k 组作为代表子空间。在 **budget 1024** 时该子空间只复现 13% 的 full top-k
  token（选择力弱）；且 core_module 为**纯 top-k 无 sink/swa 保送**，RULER 这类
  needle 高度局部化的场景一丢即崩。
- **与论文声称不矛盾**：FASA 论文只报告 LongBench（近无损），未声称 RULER。其 LongBench
  近无损口径依赖更大预算/更好校准；我方主表 1024 口径下 qasper 掉 12.6 分，属
  「预算-方法不匹配」而非论文造假。

## 4. 与 AsyncTLS 论文 DS 数字的矛盾解释

AsyncTLS 论文报告 Double-Sparsity（DS）RULER 32K AVG 90.44 ≈ Full 90.45，与 FASA
RULER 崩盘（24.00）形成反差。原因（校准维度不同）：
- DS 用 **channel 子空间**（head_dim 维 top-dc=32，INT4 量化 key 索引）+ sink 保送；
- FASA 用 **freq-group 子空间**（2-D RoPE 64 组，仅 2 维×32 组复现）无保送；
- 两者选择力与鲁棒性差异大，不可直接类比。

## 5. 主表上表策略

- **LongBench 16 任务（budget 1024）**：本地 2 卡全量派单中（GPU0/GPU1 各 8 任务），
  完成后 eval 打分入库——低分也是对手数据，标注「1024 预算口径，freq 细筛选择力弱」。
- **RULER 32K**：AVG 24.00 已正式落盘（.187 上 fasa_32k.json），照常入库标注方法局限。
- **RULER 64K/128K**：不再新派（同因崩盘，浪费资源；.187 上 64K 若已跑完则回收数据）。
- **附加单调性证据**：qasper 1024/4096 两档已入库，作为「适配正确性」的旁证可引用。

## 6. 引用

- FASA 官方仓库：third_party/FASA-ICLR2026（vendor 移植 fasa_vendor/）
- 我方适配：exp/vendor_baselines/fasa_vendor/
- RULER 32K 数据：exp/trace/results/fasa_32k.json（.187）
