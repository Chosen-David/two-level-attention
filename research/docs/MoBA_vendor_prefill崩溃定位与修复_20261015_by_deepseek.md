# MoBA vendor RULER 全 0 分崩溃定位与修复（prefill 输出布局 bug）

日期：2026-10-15。作者：deepseek 轨（MoBA baseline 打分闭合中主动发现并修复）。

## 1. 症状

- MoBA RULER **32K / 64K 正式打分全部 0 分**（`moba_ruler32k_formal.json` / `moba_ruler64k_formal.json`，n=100 齐全）。
- pred 全为重复 token（`is the is the is the...`），**第一生成 token 就分歧**（dense=英文空格、moba=中文乱码）。
- 对照诊断：同一 RULER `niah_single_1` 3 样本，dense 全答对（4226067/4823498/2694522），moba `c4096_k12` 全崩。

## 2. 关键判定：c4096_k12 数学上 = dense

`c4096_k12` 在 32K（S=32612）：chunk=4096 → 8 chunks，filtered=7（尾 chunk 走 self-attn），
`moba_topk = min(12-1, 7) = 7` → **每个 query 选中全部已完成块 + self chunk = 完整前缀 ≈ dense causal**。
它仍全崩 → 必为 vendor 适配 bug（非 MoBA 方法局限）。

## 3. 定位链（全部实测）

| 步骤 | 实验 | 结果 | 结论 |
|---|---|---|---|
| A | 纯随机 QKV `moba_attn_varlen`(c4096_k12) vs dense flash varlen | max 0.000977 | 算法核心正确 |
| B | 逐层 hidden/attn 对比 dense vs moba | **layer0 就差 3.36，逐层放大到 layer35 528** | 崩在 attention |
| C | 捕获 patch 传出的 layer0 q/k/v，同批张量 moba vs flash | 0.0009 | q/k/v 喂给 moba 正确 |
| D | dense vs moba 的 layer0 q/k/v 逐位对比 | **0.0** | q/k/v 构造一致 |
| E | d/m 权重对比（q_proj/o_proj） | **0.0** | 权重一致 |
| F | 一锤定音：完整复现 dense 原生 forward 记录 o_proj 前后 | d_pre==enable_gqa、d_post==sdpa_repeatkv+o_proj（0.0）| **dense 模型输出正确** |
| G | moba hook 捕获的 layer0 输出 vs sdpa_repeatkv+o_proj | **3.36** | **moba wrapper 输出错** |

F/G 组合锁定：dense 模型输出正确（== 标准 repeat_kv attention），moba 模型 hook 输出错；
而 q/k/v、权重、moba_attn_varlen 核心全对 → **bug 只在 wrapper 的输出形状搬运（o_proj 输入布局）**。

## 4. 根因

`exp/vendor_baselines/moba_vendor/qwen3_moba_patch.py` prefill 分支：

```python
attn_output = attn.reshape(B, S, H, D).transpose(1, 2).contiguous()  # [B,H,S,D]
...
attn_output = attn_output.reshape(*input_shape, -1)                    # [B,S,H*D] ← 错
attn_output = self.o_proj(attn_output)
```

Qwen3 原生约定：attention_interface（sdpa）返回 `[B, S, H, D]` 连续（sdpa_attention_forward
内部 `transpose(1,2).contiguous()`），原生 `reshape(B,S,-1)` 得 `[B,S,H*D]`（s 外层 h 内层）。

wrapper 却先 transpose 成 `[B,H,S,D]` 连续（h 外层 s 内层），再 `reshape(B,S,-1)` 会把
内存按 h 外层重新分组 → **h/s 维度错位 → o_proj 输入错乱 → 每层 attention 输出崩**。

## 5. 修复（只改适配层，官方算法逐字节不变）

`qwen3_moba_patch.py`：
```python
attn_output = attn.reshape(B, S, H, D).contiguous()  # 去掉 transpose
```
`moba_attention_official.py` 保持 md5=da636efc 逐字节一致（铁律）。

## 6. 验证（修复后）

- **prefill logits**：dense/moba `c4096_k12` top5 完全一致
  `[' The',' ',' To',' Let',' Alright']`，argmax diff=False，max|d-m|=0.375（bf16 长序列正常累积差）。
- **e2e 生成**（RULER niah_single_1 sample0，max_new=64）：
  - `c4096_k12`：**answer_hit=True**，输出 `The number is 4226067...`（与 dense 一致）。
  - `c256_k4`：输出流利英文句子（`The grass is green...`）未答对——budget=1024 低预算
    检索力不足（FASA 1024 同样崩），**非数值崩**（修复前是 `is the is the` 完全退化）。
- **layer0 attn**：dense vs moba 3.36 → **0.0156**（online-softmax 合并正常数值差）。

## 7. 影响与后续

- **32K / 64K / 128K RULER + LongBench 已跑结果全部作废**，需用修复后代码重跑。
- 重跑清单：32K 主表（c256_k4）+附录（c4096_k12）11×2 格、64K 11×2 格、128K 四档、
  LongBench 主表+附录。
- 教训：vendor 适配层「形状搬运」是高风险点——凡是与原生 attention 输出布局有关的
  reshape/transpose 必须与 reference（sdpa）逐元素对拍，且 c4096_k12 全选块档是天然的
  dense 等价锚点（用它能快速区分「方法近似」与「适配 bug」）。
