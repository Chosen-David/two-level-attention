# yarn factor round18 作废判定修正：fullkv 128K 数据实际有效（deepseek 发现）

- **时间**：2026-10-13 12:1X（deepseek）
- **关联**：round18（08:40）判定 e119 128K 三臂（fullkv/mavg/aavg）因 receipt yarn_factor=2.0 作废
- **交叉验证**：Opus 于 13:2X 独立产出 `exp/trace/results/e119_128k_yarn_factor_reverdict.json`，结论与本报告完全一致（锚点恢复有效、mavg/aavg 128K 重跑取消、round18 教训修正为「scorer operator 声明 ≠ producer effective」）。DeepSeek 12:1X 报告为独立发现，两方互证。

## 结论（fullkv 已铁证）
**旧 128K fullkv 数据（10090704，e119/e116b 共用）实际就是 factor 4.0 跑的，AVG 46.43 是真实有效值。**

证据链：
1. 今天 fullkv 128K 重跑（receipt effective 4.0，TS 083722/083747/083825）11 任务与旧数据 **SHA256 逐字一致**。
2. 决定性 A/B（本地 pred_ruler，niah_single_1 单样本，130981 token）：
   - factor 4.0 重跑 pred == 旧 10090704 pred（**215 字符逐字相同**）
   - factor 2.0 重跑 pred 不同（"is 4226067. This number..." vs "is **4226067**...first occurrence is:  "）
   - 另 A/B forward 证明 factor 2.0/4.0 logits 不同（max abs diff 3.42）→ yarn 实现正常
3. e119 20261010 打分 manifest 的 factor=2.0 是**打分侧 operator_declared**（无生产者证据，producer_evidence=None），属误标；e116b（20261011）manifest 声明 4.0 operator_declared 反而是对的。

## 推论
- **minf 128K 49.53 > fullkv 128K 46.43 是真实信号**（非 round18 认为的"sparse>dense 异常 bug"）——MInference sparse 在 128K 反超 dense fullkv 锚点。建议复核主表口径后入表。
- fullkv 128K 重跑无信息增益，但验证了旧数据正确性（确定性 greedy decode 下 2.0/4.0 输出可不同，勿混用）。

## Opus 已完成（13:2X reverdict，与本文一致）
- mavg/aavg 128K 也判定有效（e119 批 mavg 47.49 / aavg 42.56 / fullkv 46.43，全部 effective 4.0）→ **重跑取消**。此节原「请 Opus 自查」作废。

## deepseek 已推进
- quest 128K 代表任务（single_2/mk1/vt × 50）已派 .187 GPU7，TS 114817。
- fasa_env（torch 2.5.1 + transformers 4.56.2，Qwen3 需 ≥4.5x）--default-timeout=600 重装中。
- **minf 128K 正式打分 = AVG 49.53**（exp/trace/results/e119_ruler128k_formal_minference_vendor.json，producer_receipt 4.0）。

## open_question_minf（归 DeepSeek 范围，Opus reverdict 转交）：minf 128K 49.53 > fullkv 128K 46.43
- **现象真实**：fullkv 与 minf 均为 producer_receipt 验证的 effective 4.0，无 factor 混淆。
- **任务级拆解**（minf vs fullkv 128K）：minf 领先在检索聚焦类——fwe 36.67 vs 13.0（+23.7）、vt 57.8 vs 22.6（+35.2）、multivalue 39 vs 2.5（+36.5）、mk2 79 vs 57（+22）、multiquery 25 vs 12.25（+12.75）；minf 落后在 single_1 19 vs 100（-81，大败）、single_3 86 vs 100、mk3 0 vs 1。cwe 两者同为 3.4。
- **机理假设**：128K 长上下文下 dense fullkv 注意力被 131072 token 稀释（检索类 token 信号弱 → cwe/fwe/vt/multivalue 低分），MInference 的 2D 稀疏 pattern（vertical 每 head top-token + horizontal 每 key top-head）天然聚焦少数重要 token → 缓解稀释 → 检索类反超。single_1 惨败提示 minf pattern 可能漏掉该样本 key 所在区域（pattern 不稳定）。
- **待办**：向用户报告该现象口径；可对照 MInference 论文在 128K 长上下文的表现是否一致。
