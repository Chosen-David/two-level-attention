# two-level-attention (LoDA-SA)

两级索引稀疏注意力（Two-Level Indexer Sparse Attention，论文显示名 LoDA-SA）：
training-free 的长上下文 decode 稀疏方法——L1 块级粗筛（minmax 上界 + 近端均值双池）
+ L2 token 级 4bit 细筛 + sink/swa 正交强制区，配 CUDA graph 化 e2e 部署。

**当前叙事与全部主表数字的唯一锚点：[`agent_doc/CURRENT_NARRATIVE.md`](agent_doc/CURRENT_NARRATIVE.md)**
（生产口径 champion flags、跨模型族/跨规模加速比表、对比基线口径、
MLA 子空间链、已被推翻叙事索引——新读者从这里开始，不要读 archive）。

## 结构

- `sparse_attn/` — 主方法：indexer（TLI 主方法 + TIA 父类 + Quest 基线）、
  patches（Qwen3 / Llama-3.1 / DeepSeek-V2 三族注册）、ops（eager decode、
  CUDA graph 链、稀疏 prefill）、info（treatment 身份/manifest）
- `benchmark/` — RULER / LongBench 生产 harness + 各 vendor 基线 runner
- `exp/trace/results/` — 全部实验判决 JSON 与论文数据（只读归档）
- `exp/bench/` — 性能基准与回归门禁（`test_e127_gates.py` /
  `test_tli_batch_gates.py` 为改动后必跑的硬门）
- `exp/trace_archives/` — 历史探针/旧图/早期原型归档（不参与现行叙事）
- `agent_doc/` — 任务索引、现行建议与叙事锚点；`advice/archive/` 内
  报告已被后续判决推翻，仅作历史留档

## 快速开始

```bash
python -m benchmark.RULER.pred_ruler --method tli \
  --tli_far_method minmax --tli_near_method avg \
  --tli_near_pages 32 --tli_near_tokens off --tli_near_len 4096 \
  --tli_subspace tail --tia_level2_cmp_ratio 4 \
  --tli_sink_tokens 4 --tli_swa_size 128 \
  --tia_level1_topk 128 --tia_level2_topk 1024
```

注意：`--tli_subspace`（默认 full）与 `--tia_level2_cmp_ratio`（默认 2）
必须按上面显式传参，缺省会跑成非主表口径。
