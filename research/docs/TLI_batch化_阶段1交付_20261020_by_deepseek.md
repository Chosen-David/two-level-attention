# TLI 主方法 batch 化：阶段 1 交付报告（2026-10-20，deepseek 轨）

## 目标

TLI 主方法（HF eager 参考路径）decode 支持 batch>1 批量生成，验收口径 = **答案一致 +
有限步（≤64）逐位一致 + 门禁**（vendor batch 已证明 bs>1 与单序列存在 bf16 累积舍入差，
长序列物理不可逐位全程一致，用户已确认此口径）。

## 实现（正确性优先版）

### 1. `sparse_attn/patches/qwen3_attn_patch.py` — batch decode 分支

新增模块级 helper `_tl_decode_batched(self, indexer, query_states, key_states,
value_states, ctx)`：

- **right-pad KV 布局**（pred_ruler_batch left-pad 协议）：cache 行 b 有效 KV 占尾部
  `ctx.seq_lens[b]` 列、pad 前缀在前；逐行切 `k[b:b+1, -L_b:]` 即完整单序列 KV。
- **逐行单序列核心复用**：每行 `indexer.clear()` + `prepare_mask`（共享 indexer 全量重建）
  + `eager_decoding_attn` 单行——**无跨行增量状态，行间互不污染**。
- **全量 vs 单序列增量逐位一致（E133c 依据）**：prepare_index 全量重建与
  `_prepare_index_inc` 增量的 k_min/k_max/k_qat 逐位一致（块统计/量化确定性归纳）→
  B=1 batch 模式与旧单序列入口逐位一致（gate_bs1_equiv 依据）。
- 形状流：`query_states [B,H,1,D]`（cache.update 后 head-first）→ q `[B,1,H,D]`；
  k/v `[B,S_total,Hkv,D]` → 逐行 b：q_b `[1,1,H,D]`、k_b/v_b `[1,L_b,Hkv,D]`、
  cu=`[0,L_b]`、pos=`[L_b-1]`。返回 `[B,1,H,D]`。

decode 分支入口插入 batch 检测（B2 硬守卫之前）：

```python
_ctx = batch_decode_ctx()
if _ctx is not None:
    attn_output = _tl_decode_batched(self, self.indexer, ...)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    _paper_route(self.layer_idx, "sparse-decode-batch")
    return attn_output, None
```

- **零扰动铁律**：非 batch 模式（ctx None）走既有路径逐位不变；batch 分支独立于
  TLI_DECODE_DENSE 消融与 B2 守卫（batch 分支要求 past_key_values 非 None，decode
  分支进入条件已保证）。
- skip 层（dense）走原生 eager_attention_forward，消费 4D additive mask 排除 pad——
  由 pred_ruler_batch 的 `_decode_4d_mask` 提供，batch=4 冒烟已验证正确。

### 2. `benchmark/RULER/pred_ruler_batch.py` — batch 入口（阶段 1）

- `_BATCH_METHODS = {"none", "fasa_vendor", "tli"}`（Phase 1 白名单；quest_vendor 等
  batch_size>1 启动即拒绝，防静默产错）。
- dispatch 分支新增 `tli`（同步义务内联自 pred_ruler.py L184-185 + L444：
  `resolve_treatment_snapshot(args)` + `register_patch(model, args, snapshot)`）。
- **budget 采集对齐 pred_ruler.py L529-551**（本阶段修复重点）：chunk 前
  `metrics.clear()`、`with baseline_request(model) as baseline_receipt:` 解包 receipt、
  chunk 后 `budget = None if baseline_receipt else metrics.get_select_tokens()`。
  - tli 无 `_baseline_identity` → receipt 空 dict falsy → budget 取实际 select_tokens
    （如 1040.37，原硬编码 None 是 batch=1 vs old 逐字节一致的唯一差异）。
  - batch=1 时 chunk=单请求，budget 与旧入口逐字节一致；batch>1 时取该 chunk 全部
    行×步 select_tokens 均值（聚合口径，诚实标注非单请求口径）。

### 3. 精筛候选化（阶段 0）继承

逐行循环完全复用单序列核心 → 阶段 0 的 `score_fine` 候选化（TLI_FINE_GATHER）天然
继承，无二次重构。

## 验收（4 项门禁全部通过）

| 门禁 | 内容 | 结果 |
|---|---|---|
| **gate_bs1_equiv** | pred_ruler_batch `--batch-size 1` vs pred_ruler.py 旧入口同参数 JSONL 逐字节一致 | **PASS**（2 样本，每行 raw 字节完全相同；含 budget 字段） |
| **gate_anchor_batch** | 预算拉满==dense 锚点（batch 逐行 mask 全连 true=512/512）+ TLI_FINE_GATHER 0/1 mask 逐位一致 | **PASS**（G2 单元门禁 nd=0） |
| **gate_multiseq_consistency** | B=4 混合长度多步 decode 逐行 vs 单序列核心逐位一致；e2e batch=4 pred 内容与 batch=1 一致（budget 字段 batch>1 为聚合口径，属预期差异） | **PASS**（G3 单元门禁 16 项全过） |
| **E127 门禁回归** | test_e127_gates.py 阶段 0 后基线 | **PASS**（15 PASS / 0 FAIL） |

单元级门禁 `exp/bench/test_tli_batch_gates.py`：**26 PASS / 0 FAIL**（G1 8 项 + G3 8 项
+ G2 2 项 + G4 8 项）。

## 诚实局限

- **逐行循环（未张量化）**：batch 分支逐行调用单序列核心（indexer.clear + 全量重建 +
  eager_decoding_attn），正确性最稳但非性能路径。张量化（批量 einsum、sglang
  `select_decode_batched` 蓝图、eager kernel batch 维）留待 CUDA Graph 阶段一并做
  （阶段 2 目标：decode 循环图化吃 launch 开销）。
- **budget 字段 batch>1 聚合口径**：非单请求均值，与 pred_ruler.py 单序列口径不同
  （batch=1 时逐字节一致不受影响）。
- **全量重建**：batch 分支每行全量重建 index（无 per-row 增量），decode 步数多时
  有重复计算（E133c 已证与增量逐位一致，语义正确；性能优化留阶段 2）。

## 涉及文件

- `sparse_attn/patches/qwen3_attn_patch.py`（+59：_tl_decode_batched + batch 分支）
- `benchmark/RULER/pred_ruler_batch.py`（新增：batch 入口 + tli 挂载 + budget 采集）
- `exp/bench/test_tli_batch_gates.py`（新增：4 项单元门禁）
- 只读参考（不动）：sglang `python/sglang/srt/layers/attention/tli/{indexer,backend,kernels}.py`

## 阶段 2 衔接

CUDA Graph decode：固定 KV buffer（E134-pre StaticCache 已就绪）→ decode 循环图化
（B 固定、S_max 固定，预分配输入缓冲 + capture + replay copy_ 注入）→ graph vs eager
逐位对拍硬门（借鉴 E139 `exp/bench/e139_bs_graph_bitdiff.py`）。
