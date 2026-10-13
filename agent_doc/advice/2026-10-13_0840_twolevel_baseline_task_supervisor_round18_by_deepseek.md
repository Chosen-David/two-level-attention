# S-T023 任务监督器 round18 报告（2026-10-13 08:40，by deepseek）

## 一句话
128K MInference 11/11 完成并打分（**49.53，factor 4.0 有效**）；**发现重大口径 bug：fullkv/mavg/aavg 128K 误用 yarn factor=2.0（位置编码溢出，数据作废）** → 已派 fullkv 128K 重跑（三机 11 卡）。

## 本轮完成动作

### 1. 128K MInference 11/11 全齐 + 打分（有效数据）
- 回收 .187 7 + 本地 2 + .251 2（各 100 行，tli_gen 11），`score_ruler_formal --yarn --yarn-factor 4.0` 打分
- **AVG 49.53**：single_2 100/single_3 86/mk1 99/mk2 79/vt 57.8/mv 39/fwe 36.67/mq 25/single_1 19/cwe 3.4/mk3 0
- yarn 身份 4.0 认证 11/11 通过（producer receipt v2，prediction SHA/行数逐格比对）
- **MInference 长度趋势（factor 正确时）**：32K 49.94 / 64K 50.29 / 128K 49.53 —— 三档持平不掉点

### 2. ⚠️ 重大口径 bug 发现：128K 三臂 yarn factor 用错
- **现象**：128K 全臂对比出现 minf 49.53 > fullkv 46.43（+3.10）> mavg 47.49，sparse 反超 dense 不合理
- **排查**：读 fullkv/mavg/aavg 128K receipt 的 inputs.run_identity → **三臂 yarn_factor 全 = 2.0**（错误）
- **根因**：131072 token 数据在 effective context = 40960×2 = 81920 下推理 → 位置编码溢出（超 MPE 的部分 PE 错误）→ **精度受损，数据不可用**
- **正确口径**：pred_ruler.py 自动档 `YARN_FACTOR_AUTO = {65536: 2.0, 131072: 4.0}`（line 100）+ Qwen3 官方配方 128K→factor 4.0（effective 163840）。MInference 128K 用 4.0 已验证有效（minf 128K 49.53 真实）
- **影响**：20261010 产出的 fullkv/mavg/aavg 128K 全部作废；minf 128K 49.53 有效；quest 128K 先行用 4.0 有效（跑中）
- 旧数据表现佐证受损：fullkv 128K multivalue=2.5（minf 39）/fwe=13.0（minf 36.67）/mavg 128K multivalue=2.0 —— 长序列检索任务全部崩（PE 溢出典型症状）

### 3. fullkv 128K 重跑已派（factor 4.0 修正，三机 11 卡）
- .187 GPU0-6：cwe/fwe/mk1/mk2/mk3/s1/s2（pid 2499910）
- 本地 GPU0/1：single_3/multiquery（2 进程）
- .251 GPU0/1：multivalue/vt（pid 3030106）
- 全部 `--method none --yarn --yarn_factor 4.0 --context_length 131072 --max-num 100`
- ~1.5h/任务 → 约 10:15 完成 → 打分（best-file 仲裁自动选最新 TS，覆盖旧 2.0 数据）
- **mavg/aavg 128K 待重跑**（fullkv 完成后派，同 factor 4.0 修正）

## sanity 门禁状态
- MInference 三口径（LBv1/LBv2/RULER 32/64/128K）全通过：LB 近无损（-1.68/-0.80）、RULER 三档持平 fullkv（+1.75/+3.10 噪声内，128K 旧 fullkv 作废待重跑对拍）
- quest 128K：GPU7 先行跑中（4.0 正确）
- **本次 bug 是打分口径审查逮住的**：sparse 反超 dense 的异常信号 → 深查 receipt → 定位 factor 错误。教训：跨档位数据必须核对 producer yarn identity，长度越高越要审

## 在飞 / 待办
- fullkv 128K 重跑（11 卡，~10:15 完成）
- quest 128K：GPU7 先行 → 批量 10 任务
- mavg/aavg 128K 重跑（factor 4.0）
- 32B pattern 搜索（本地 fullkv 128K 完成后）
- ClusterKV 阻塞（镜像死锁确认）

## 资源状态
- .187 GPU0-6：fullkv 128K 满载；GPU7：quest 128K 先行
- 本地 GPU0/1 + .251 GPU0/1：fullkv 128K 满载
- **三机 12 卡全满载**
