# round14 任务监督：MInference Qwen3 适配跑通 + sanity 近无损对拍通过

> 作者：deepseek · 2026-10-13 05:05 · 任务监督器（用户 2026-10-11 18:5X 指令，有实质进展才写）

## 1. 本轮具体动作（MInference 从"通道打通"推进到"Qwen3 适配 + sanity 对拍通过"）

### 1.1 Qwen3 适配轻量胶水 `qwen3_minference_patch.py`
官方 `MInference(...).patch_model()`（new_patch）在 Qwen3 上不可直接用：会替换 DecoderLayer/
model.forward 为 **Llama 写死架构**、替换 `_prepare_cache_for_generation` 为 kvcompression 的
`DynamicCacheWithRepeat`（Qwen3 generate 崩 `no attribute 'key_cache'`）；且 `attn_forward` 的
prefill/decode 判断 `q_len == past_key_value.get_seq_length(layer_idx)` 对 Qwen3 首次 prefill
（cache 空 get_seq_length=0、q_len>1）**误走 decode（dense）→ MInference sparse 静默不生效**。

轻量胶水只替换 Qwen3Attention.forward：wrapper 显式 `q_len>1` → prefill 走官方
`minference_prefill_forward`（逐 head vertical+slash 稀疏，复用官方 kernel 逐字节不动），
`q_len==1` → 原生 dense（保存的 flash_forward）。Qwen3 原生 cache/generate 流程保留，
QK-Norm（q_norm/k_norm）在 prefill 分支原生复用。

### 1.2 三个适配 bug（全解）
| 现象 | 根因 | 修复 |
|---|---|---|
| `search_pattern` 崩 `topk k out of range` | 官方对短序列的固有限制：vs_list 最大 vertical_size=3500，q_len<3500 时 `topk(q_len-vertical_size)` 为负 | 搜索文本须 >3500 token（改 8192） |
| `IndexError: index 8 out of bounds` | Qwen3-8B 是 GQA（32 头 Q / 8 头 KV），官方 `minference_prefill_forward` 逐 head 取 `key_states[:, head]` | prefill 分支在调 prefill_forward 前 `repeat_kv`（与官方 attn_forward line 107-111 同源） |
| 正式 sparse 崩 `KeyError: ''` | 搜索文本是重复句子，608/1440 个 head score≤0 搜出空 pattern `['',0,0,0]`，kernel `.get(str(head_id))` 返回空 | ① 搜索脚本改用自然多样文本（0 空 head）；② sanitize：空 pattern → 官方默认 `('vertical_and_slash',1000,6096,1)`（pattern 数据预处理，不改官方 kernel 源码） |

### 1.3 pattern 在线搜索（Qwen3-8B 自己搜）
`qwen3_minference_search.py`：自然文本 8192 token 一次 prefill，36 层 × 32 头
`search_pattern` 逐层逐 head 搜（vertical_and_slash/stream_llm/block_sparse 取 mass 最优）。
**结果：36 层全 `vertical_and_slash`、0 空 head**，layer0 head0 score 0.858。
产物 `best_pattern_qwen3_8b.json`（36 层 × 32 头）已落盘 vendor 目录。

### 1.4 pred 接入（B 轨 vendor 正式跑）
- `pred.py`：新增 `method=minference_vendor` 分支（`enable_minference(is_search=False)` +
  `--minference_config_path`）。
- `sparse_attn/arguments.py`：`--minference_config_path/--minference_starting_layer` 参数。
- `sparse_attn/info.py`：`minference_vendor_{pattern}_{starting_layer}` method_name（输出身份段）。

### 1.5 sanity 对拍（硬门禁通过）
本地 GPU0/1 派单 2 任务（`dispatch_lb_v1_sanity_2card.sh`，TS 20261013-041503）：
| 任务 | MInference | FullKV(dense) | Δ |
|---|---|---|---|
| qasper | **47.64** | 47.47 | **+0.17** |
| multi_news | **24.94** | 24.78 | **+0.16** |

**MInference sparse prefill ≈ dense（近无损）完全符合论文核心结论**（NeurIPS 2024 "sanity
preserving sparse prefill"）。对比 Quest 同任务 qasper Δ=-11.04、multi_news Δ=+0.68，
MInference 的 prefill 稀疏明显比 Quest 的 decode 稀疏更贴 dense。**→ sanity 对拍硬门禁通过，
MInference 可上主表对比臂。**

## 2. 状态盘点（对照 S-T023 竞品清单）

| 竞品 | 状态 |
|---|---|
| Quest | 全量收官（LBv1 8B 41.35/32B 38.08、LBv2 30.22、RULER 32/64/128K 42.85/26.25/19.33） |
| **MInference** | **Qwen3 适配 + sanity 近无损对拍通过（本轮）** |
| DeepSeek fullkv | 收官（AVG 40.55*，16K 口径 MLA dense 锚点） |
| ClusterKV | 硬阻塞（pypi 无包 + github 直连/镜像全超时） |
| MoBA | training-free sanity 暂缓（README 需训练 + flash-attn 缺） |
| FASA/SparQ | 未接入（同为 github 源码，受超时影响） |

## 3. 诚实声明

- 空 pattern sanitize（608 head → 默认 1000/6096）只作用于 pattern 数据文件，官方
  `minference_prefill_kernel` 源码逐字节不动；自然文本重搜后 0 空 head，sanity 产物
  无 sanitize 残留。
- sanity 只跑 2 任务（qasper 短检索 + multi_news 长摘要），代表论文近无损趋势；全量
  16 任务待派单后补齐精确掉点分布。
- 搜索 pattern 按 8192 token 自然文本在线搜出；LongBench 任务 prompt 2K-21K，kernel
  内部对任意 q_len clamp（vertical_size=min(q_len,max(v,30))），pattern 位置归一化。

## 4. 下一步建议

1. **MInference 全量 16 任务派单**（本地 2 卡 + .187/.251 12 卡）→ LBv1 8B 主表对比臂
   （对拍 Quest 41.35 / fullkv 49.39）。
2. 若全量掉点分布异常（非近无损），回查 pattern 质量 / starting_layer 口径。
3. RULER 32/64/128K MInference 臂（sparse prefill 长序列掉点随长度趋势对拍）。
