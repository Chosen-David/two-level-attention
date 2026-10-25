# E175：far-skip-kv-eviction codex 分支兼容性审阅

分支：`origin/codex/far-skip-kv-eviction-20261025` → `perf/gpt-kv-eviction`
审阅日期：2026-10-25
基准主树：perf/gpt-kv-eviction @ 320097b（merge commit，含 iter3 K1=64）

## 1. 冲突记录

| 文件 | 冲突类型 | 解决 |
|---|---|---|
| `sparse_attn/indexer/tli_indexer.py` | CONTENT（iter3 kernel 全部 vs codex 仅 3 处初始化） | 保留 HEAD（iter3 完整 kernel 链）+ 手工回溯 codex 3 处 `far_skip_*` 初始化 |
| `agent_doc/task/TASK.md` | 自动合并 | git 自动合并通过 |
| `.claude/`, `CLAUDE.md`, `CODEMAP.md`, `docs/far_skip.md` 等 | 新增文件 | 自动合并 |

最终 merge commit：`320097b`。

## 2. 审阅点清单

### 2.1 是否兼容 iter3 融合链（M/S/C Triton kernels）？

**结论：WARN（接口并行但物理不冲突）**

- **分析**：far_skip 走的是完全独立的 Python 运行时栈 —— `select_far_skip()` + `attend_far_skip()` + `FarSkipWorkspace`，从零构建 SDPA 输入。iter3 的 `_iter3_maint_kernel`/`_iter3_l1_scored_kernel`/`_iter3_l1_cand_kernel`/`_iter3_sel_base_kernel` 等仍在 `tli_indexer.py` 内服务 production `compute_mask` / `compute_mask_graph` 路径。
- **不冲突**：codex `try_far_skip_decode()` 在第一道 guard 后**直接 return**（line 98-99），不经过 `_compute_far_skip_delta` → `compute_mask` → iter3 kernel 链。两栈 patch 挂载位置相同（`qwen3_attn_patch.py` line 234），但由 `indexer.far_skip_backend` 标量门控互斥——默认 "legacy" 完全不进入 far_skip 路径。
- **WARN 原因**：文档 (docs/far_skip.md line 3) 明示 "has not been validated" 且 "CUDA Graph remains unsupported" ——与 iter3 主打的 graph 优势路线（E152v4 → iter3d 累计 −24~40% TPOT）是完全不同的生产栈。同一 attention 函数的两栈共存需要在 patch 入口严格互斥。

### 2.2 是否兼容 CUDA Graph capture（FarSkipLayer 与 TLIStaticCache）？

**结论：FAIL（明确不兼容，显式拒绝）**

- `try_far_skip_decode()` line 56: `or (q.is_cuda and torch.cuda.is_current_stream_capturing())` → raise ValueError ("no native sliding window")。
- `FarSkipLayer.update()` 每次 decode step 执行 `self.keys[..., slot:slot+1, :].copy_(key_states)` + metadata 写入，纯 eager 路径，与 TLIStaticCache 的 graph-static 张量约定（固定地址、无元数据更新）冲突。
- `FarSkipLayer` 并非 TLIStaticCache 适配——它走独立 DynamicCache 分支，HF 4.56 DynamicLayer 的子类。
- **文档印证**：docs/far_skip.md line 100-108 明示 "CUDA Graph remains unsupported"、"StaticCache... outside this implementation"。
- **生产含义**：当前 champion 生产口径（graph mode, `TLIStaticCache`）与 far_skip_kv_eviction 完全正交 —— 这是 eager B=1 实验/消融路径专用，不能用于任何 graph TPOT 测量。

### 2.3 FarSkipConfig 严格模式是否与现有 gate mask 不兼容？

**结论：WARN（语义不同但 fail-closed）**

- docs/far_skip.md line 60-63 明示 "not bitwise parity with the old gate" —— 老 sink-boundary bypass 允许非 sink far token 进入候选 (boundary_bypass=True)；far_skip 的 `select_far_skip` line 143-180 严格跳过 partial page 的未保护 far token，只保留 near token 进 L2。
- `configure_far_skip()` line 24 校验 `indexer.boundary_bypass` 必须为 True（生产几何已满足）。
- **实际影响**：生产默认 boundary_bypass=True；far_skip 在此之上做**更严格**截断——集合理应是老 gate 的子集。这意味着 far_skip "evict" 层 decode 期 attend 的 KV **⊆** legacy 老 gate 的 KV，不会读到老 gate 未读的 token。数学上是 safe-subsetting，但精度上是新语义（selection 结果与 production 4bit 路径不等）。
- **与 iter3 mask 链关系**：iter3 的 `_iter3_sel_base_kernel/scatter` 和 `_iter3_compact_kernel` 服务的是 4bit near/far 路径。far_skip **完全绕开**这条链——`_subspace_indices`/`_qat_indices` 只用于 coarse/fine scoring 选择维度（line 86-88），不产出 mask tensor。两者在代码路径上物理互斥（same patch function, different if-branch），不存在双 mask 混合风险。

### 2.4 cyclic buffer update() 是否支持增量 prefill？

**结论：FAIL（不支持，且明确拒绝）**

- `FarSkipLayer.update()` line 60-61: `if key_states.shape[0] != 1 or key_states.shape[-2] != 1: raise ValueError("evicted cache supports only single-token, batch-one decode")`。
- `far_skip_cache.py` line 56-75 的实现：稳态 decode 单 token 写入 cyclic slot，是 **1-token update 专用**。
- transformers 4.x 兼容：`FarSkipLayer(DynamicLayer)` 走 HF 4.56 标准 DynamicCache 接口（`update/get_seq_length/reset`）; 普通 DynamicLayer 则走标准 DynamicCache.append 路径——**接口兼容**。但不支持 chunked prefill（multi-token update）。
- **生产影响**：生产 prefill 必须是 `legacy` backend（让老 DynamicLayer 吸入全部 KV）；然后在 decode 首步通过 `evict_far()` 一次性 `from_layer()` 把已有 DynamicLayer 转成 FarSkipLayer（这个过程已经拷贝了全量历史 sink/tail）。之后才能 decode 1-token-at-a-time。这意味着 prefill→decode 切换是一次性的、不可逆的 layer 替换。

### 2.5 flag 接口是否干净接入 arguments.py？

**结论：PASS（干净）**

- `arguments.py` line 370-373: `_add_tli_args()` 里加入 2 个新 flag：
  - `--tli_far_skip_backend {legacy,compact}` default="legacy"
  - `--tli_far_skip_cache {retain,evict}` default="retain"
- `info.py` line 133-134 / 416: 默认 identity 已注册（legacy/retain = no-op）。
- `patch.py` line 55-62: 注册准入（仅 TLI Qwen3, head_dim=128, no sliding window）。
- 校验闭环：`configure_far_skip()` line 16-32 枚举了所有非兼容配置并显式 reject（async/moba/sigma/per_q_head/cluster/proj/rope_subspace...）。
- 默认值 legacy/retain 不改变任何现有臂行为——`getattr(args, "tli_far_skip_backend", "legacy")` 全仓 fallback 安全。

## 3. 额外发现

### 3.1 patch 挂载点在 `qwen3_attn_patch.py`，不在 repo `eager_decoding.py`
codex 主入口 `try_far_skip_decode()` 挂在 `patches/qwen3_attn_patch.py` line 234-242——这是 shadow 包 `~/tli_ako/solution_iter3/sparse_attn/patches/qwen3_attn_patch.py`（或等价）路径。repo 内的 `sparse_attn/ops/eager_decoding.py`（如有）**不含** far_skip 调用。测试通过直接调用 `select_far_skip/attend_far_skip` 函数，不依赖 shadow 包。

### 3.2 submodule `FarSkipWorkspace` 是请求级 append-only
`FarSkipWorkspace`（`far_skip_workspace.py` line 9-101）—— request-local、append-only、单流状态。每次 prefill 由 `try_far_skip_decode()` line 62-63 reset。跨请求不共享，使用规则清晰。

### 3.3 `evict_far()` 不可逆
`FarSkipLayer.from_layer()` line 25-55 是**一次性转换**：从 DynamicLayer → FarSkipLayer + 物理蒸发 far KV。之后同一 request 内 `skip_far=True` 层永远走紧凑路径。gate 切回 / cache mode 切换 / crop / beam / batch / offload / legacy export 均显式 raise。

## 4. 总结

| 审阅点 | 评级 | 一句话 |
|---|---|---|
| iter3 M/S/C kernel 共存 | WARN | 路径互斥未冲突，但共用同一 attention 函数入口需严格 if-branch 隔离 |
| CUDA Graph 兼容 | FAIL | 显式拒绝 stream capturing; 与 production TLIStaticCache 路线正交 |
| gate mask 严格子集 | WARN | 新语义严格 subset of 老 gate，非 bitwise 等价，无读越界风险 |
| 增量 prefill | FAIL | 仅 1-token decode 更新；不可 chunk prefill，prefill→decode 一次性转换 |
| flag 接口 | PASS | legacy/retain 默认安全，校验闭环，argparse/info/patch 三处一致 |

**整体评估**：codex far_skip_kv_eviction 是**充分 fail-closed、并行于主树**的 eager-only 实验路径，不改动 iter3 主链任何代码。作为消融分支合并到 main 是安全的（默认 flag = legacy/retain 零影响）。**不应**被视为 graph mode 性能优化的合入候选——两者解决的问题不同（延迟 vs 显存）。
