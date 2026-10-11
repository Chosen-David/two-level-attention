# two-level-attention 代码 Bug 审查 Round 1（by deepseek）

- 时间：2026-10-11 18:18
- 方式：逐条核对 kimi3 清单（F1-F12/R01/B03/B04/B10/C-1/075）修复状态 + 独立通读
  `sparse_attn/indexer/tli_indexer.py`、`sparse_attn/patches/`、`sparse_attn/metrics.py`、
  `sparse_attn/info.py`、`benchmark/LongBench/pred.py`，并做反例数值复现。
- 结论：**仍存在 1 个影响冠军配置正确性的缺陷（GPT Static cluster tail coverage，cluster(0,0)
  特有）、2 个边界配置缺陷（F6 near 超 γ 配额、near kmeans guard 缺失）、1 个低风险记录（F12）**。

---

## 1. 已确认修复（红-绿双证，逐条核验源码）

| 编号 | 问题 | 修复位置 | 核验 |
|---|---|---|---|
| F1 | skip 层 sink 永久丢失（S=4352/mavg 实测 sink 0/128） | `tli_indexer.py:1328-1330` skip_far 时 `topk_mask[..., :sink_tok]=True` | ✅ |
| F2 | qwen3 patch use_cache=False/多头/chunked prefill/4D float mask 静默零输出或崩溃 | `qwen3_attn_patch.py:63-124` decode 分支三重硬守卫显式 raise | ✅ |
| F3 | GLM-4 等未支持模型标稀疏跑 dense（结论污染） | `patch.py:68-79` method≠none 且 patched=0 → raise + GLM hint | ✅ |
| F4 | σ 死参数静默退化为默认两级 topk | `tli_indexer.py:1071-1080` fail-closed raise | ✅ |
| F5 | e64 分区 near_base 用 pad 后长度 | `tli_indexer.py:915` 真实 S 口径（075 补正投影路径） | ✅ |
| F7 | `_basis` 跨请求残留（proj_basis×static_pair） | `clear()` 重置（line 1359-1364）；`_basis_all` 不重置（离线常量，派生自愈） | ✅ |
| F8 | `_last_q` 跨请求残留 | `clear()` 重置（line 1355-1358） | ✅ |
| F10 | `metrics.py` k_delta 疑似应为极差 | `metrics.py:48` `k_max - k_min` | ✅ |
| F11 | ab 串用 flag 原值非生效值（subspace=full 时 indexer 强制关） | `info.py` subspace 生效值逻辑 | ✅ |
| R01 | BOS-aware 裁剪 | `pred.py:207-214` | ✅ |
| B03 | samsum full_ids 截断 | `pred.py` | ✅ |
| B04 | 首 token EOS 判停 | `pred.py:258-265` | ✅ |
| B10 | near 区 swa 被扣两次（near 实际宽 = α·mid−swa） | `tli_indexer.py:915` e64 分支 `near_base = S_real - swa_tok` | ✅ |
| C-1 | far 池空静默落单池（sink 不保证、超预算） | `tli_indexer.py:1209-1217` guard 只包 far 选择段 | ✅ |
| 075 | SWA 强制区落 padded 尾（S=6145 实测 65/128） | 多路径真实 S（line 1059/1096/1145/1203/1300） | ✅ |

## 2. 仍存在的缺陷

### ① GPT Static cluster tail coverage bug —— 未修复（cluster(0,0) 特有）
- **producer** `_maybe_build_kmeans`（line 390-394）：`(0,0)` 分支 `near_base = S`（**真实 S**）
  → `far_hi = (S - swa_tok)//bs * bs`（簇覆盖上界）。
- **consumer** `compute_mask`（line 916-919）：`(0,0)` 分支 `near_base = kt * bs`（**padded**）
  → `far 边界 = near_blks * bs`。
- B10 修复**只覆盖 e64_partition 分支**（line 911-915），`(0,0)` 分支未同步 → producer/consumer
  簇覆盖边界错位。
- **反例复现**：`S=4097, bs=64, sink_blocks=2(sink=128), swa=128`：
  - producer far_hi = (4097−128)//64×64 = 3968，簇覆盖 [128, 3968)，Tfar=3840；
  - consumer kt=ceil(4097/64)=65，kt*bs=4160，near_blks=(4160−128)//64=63，far 边界=4032；
  - `swa_lo_tok = S_real − swa_tok = 3969 < 4032` → near 段 [4032, 3969) 倒挂为空；
  - cavg/ccluster 路径：`w_far=min(3840, min(4032,3969)−128)=3840`，far 段=[128,3968)，
    near 段空 → **token 3968（real）无任何候选源，永不可选**。
- 默认路径（far 用细筛分 p）`mid_p=p[...,128:3969)` 含 3968 → 不受影响。
  **这正是"默认跑得对、cluster 臂才崩"的隐蔽性来源。**
- 影响面：cavg/ccluster（0,0）组合、非块对齐序列长。**若 E125 静态冠军候选含 cluster 臂，
  必须先修 consumer `near_base` 同步真实 S 口径再锁定。**

### ② F6（near 超 γ 配额）—— 未修复（E109 网格内不可达）
- `line 1236` `k2_far = min(far_budget, Tfar, K2_mid)` + `line 1268` `k2_near = max(0, K2_mid - k2_far)`：
  far 区不足（`Tfar < far_budget`）时 far 预算浪费、`k2_near = K2_mid - Tfar > nt_near`，
  near 突破 γ 配额。
- kimi3 标注"E109 网格内不可达"（far 区在该网格下总是足够）；非网格短序列（LB）配置可达。
- 建议：登记待办，不改（避免扰动在飞链）。

### ③ `_update_near_cluster` kmeans guard 缺失 —— 本次独立新发现
- far 侧有 4 倍样本守卫（`_build_far_kmeans:420` `if Tfar < far_clusters*4: return`），
- near 侧只有 `Tn < 256`（`_update_near_cluster:502`）。
- **Tn ∈ [256, far_clusters×4)（far_clusters=256 时 [256, 1024)）时，以 256 簇对不足样本跑
  kmeans（簇数 ≥ 样本数 → 退化/大量空簇）**。LB 短序列 near 区可能落入。
- 建议：登记待办，随 ① 修复窗口一并加 4 倍 guard。

### ④ F12（消费端无全 False 行防线）—— 记录
- `eager_decoding.py:45-48`；kimi3 标注"契约内不可达"，风险低。

## 3. SG 端口（sglang 侧，属 E118 待办）
- `python/sglang/srt/layers/attention/tli/` 存在；S1/S2 有修复测试（test_s1_causality.py、
  test_s2_single_pool_sentinel.py）。S3-S8 详细验证留待后续集成工作（E118）。

## 4. 建议优先级
1. **P0**：① 若 E125 冠军候选含 cluster 臂 → 修 consumer `near_base` 为真实 S 口径再锁定。
2. P2：③ 加 4 倍 guard（随 ① 修复窗口）。
3. P2：② 登记待办（不扰动在飞链）。
4. ①②③ 均不改动默认冠军配置 mavg（E123 新口径 47.91 与 fullkv 48.02 持平）。

## 5. 监督器计划（用户 18:18 指令）
- 每 2h 触发一轮：审查各 GPU 在跑任务（脚本/进程/锁文件）→ 本地实际验证可疑 bug →
  真 bug 立刻 fix 后汇报（写本目录 *_by_deepseek.md）→ 审查 GPU 空闲 → 派新任务补漏。
- Round 2 起将含 GPU 派单审查与修复执行记录。

— by deepseek（本地实际核验源码 + 反例复现）
