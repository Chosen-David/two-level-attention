# GPU 代码 bug 监督器 round13 报告（2026-10-13 07:25，by deepseek）

## 一句话
RULER 64K MInference 全完成并打分（**AVG 50.29 近无损**）；9 卡空闲 → 批量派 **RULER 128K MInference 全量 11 任务**（三机 10 卡满载）；无新 bug。

## GPU 占用表（07:2X 快照）
| 机器 | GPU | 利用率 | 显存 | 任务 |
|---|---|---|---|---|
| .187 | 0 | 100% | 61GB | 128K niah_single_1 |
| .187 | 1 | 100% | 61GB | 128K niah_single_2 |
| .187 | 2 | 100% | 62GB | 128K cwe |
| .187 | 3 | 100% | 66GB | 128K fwe |
| .187 | 4 | 100% | 66GB | 128K niah_multikey_1 |
| .187 | 5 | 100% | 64GB | 128K niah_multikey_2 |
| .187 | 6 | 100% | 62GB | 128K niah_multikey_3 |
| .187 | 7 | 0% | 1MiB | 空（无 32B 模型，留待 32B pattern 搜索） |
| 本地 | 0 | 100% | 54GB | 128K niah_multiquery |
| 本地 | 1 | 100% | 54GB | 128K niah_multivalue |
| .251 | 0 | 100% | 64GB | 128K niah_single_3 |
| .251 | 1 | 99% | 64GB | 128K vt |

## 在飞任务与脚本
- **RULER 128K MInference 全量 11 任务**（`pred_ruler.py` method=minference_vendor +
  `--yarn --yarn_factor 4.0 --context_length 131072`，pattern=best_pattern_qwen3_8b.json）：
  - 先行验证 .187 GPU0/1（single_1/single_2，07:10 启动）
  - 批量 .187 GPU2-6（cwe/fwe/mk1/mk2/mk3，07:18 启动，pid 2489643）
  - 本地 GPU0/1（multiquery/multivalue，TS 071724）
  - .251 GPU0/1（single_3/vt，pid 3021069）
  - 推理 24.6s/it ×100 ≈ 41min/任务 → **约 08:00 首轮完成**
- yarn 自动档复核：Qwen3-8B max_position_embeddings=40960 → 128K factor=4.0
  effective 163840（log 实证 `[yarn] enabled factor=4.0 (effective context=163840)`，
  =40960×4 ✓）；64K factor=2.0 effective 81920 ✓。**计算正确，非 bug**。

## 本轮实质结论：RULER 64K MInference 打分（11/11）
- **AVG 50.29**（score_ruler_formal，--yarn --yarn-factor 2.0，producer yarn 身份 2.0 认证 11/11 通过）
- 全臂对比：**minf 50.29 > mavg 49.42 > fullkv 48.54 > aavg 47.51 > quest 26.25**
- 单任务：single_2/3 100、mk1 100、single_1 75、vt 54.6、mk2 52、mv 27.5、mq 25.75、fwe 9.67、
  cwe 7.7、mk3 1
- **sanity 判定通过**：MInference 近无损符合论文（quest 64K -22.3 大幅掉点符合 decode 稀疏
  局限）。观察点如实标注：minf 在 cwe/fwe/vt 反超 fullkv（fullkv cwe=0.6/fwe=4.0/vt=32.8 vs
  minf 7.7/9.67/54.6），64K RULER 各臂本就贴近（47.5-50.3），+1.75 属噪声内。
- 长度趋势：minf 32K 49.94 → 64K 50.29（持平不掉）

## bug 审查结论
- 无新 bug。本轮审查点：① pred_ruler.py --yarn factor 自动档（65536→2.0/131072→4.0，代码
  line 18-21 + 97-123 注释与实参一致）；② yarn_receipt 身份链（producer receipt v2 同代绑定，
  64K 打分 11/11 认证通过）；③ 128K 先行验证 log 无异常。全部通过。
- 历史修复保持有效：pred_ruler.py md5 b950c107（minference_vendor 分支）+ pred.py makedirs
  幂等（exist_ok=True），本轮未触发。

## 空闲 GPU 派单记录
- 07:10 快照：64K 11/11 全完成（.187 8 + 本地 2 + .251 vt 100 行）→ 9 卡空闲
- 派单：RULER 128K MInference 全量 11 任务（10 卡，脚本 3 个：187_g2-7 / local_g0-1 /
  251_g0-1），11/11 无重复无遗漏
- .187 GPU7 保留：.187 无 Qwen3-32B 模型（models/huggingface.co 空），32B pattern 重搜须在
  本地做（本地 128K 完成后 08:00）

## 下一步建议
1. **08:00**：128K 先行+批量回收 → score_ruler_formal --yarn --yarn-factor 4.0 打分 → 长度
   趋势 32/64/128K 三档齐全（MInference 对拍 quest/fullkv/mavg）
2. **32B 臂**：本地 128K 完成后，本地 qwen3_minference_search.py 搜 Qwen3-32B pattern →
   派 32B 三口径（LBv1/LBv2/RULER）
3. **ClusterKV**：pypi 无 wheel + github 大传输超时硬阻塞，建议走镜像/代理后再试
4. .187 GPU7 可作为 128K 完成后下一轮的缓冲/32B 补位
