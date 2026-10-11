# E127 迭代日志（ako4all 循环）

格式：每 iter = bench 运行 + 本日志追加 + git commit。基线 = 当前 main 的 TLI HF 路径（env 关）。

## Summary（随迭代更新）

| iter | 方向 | S=64K 全路径 GPU ms（median） | vs 基线 | 门禁 |
|------|------|------|------|------|
| baseline | 现状（无 env 开关） | 8.29 GPU / 9.62 CPU wall | 1.00× | — |
| iter1 | metrics 惰性化（去 2×.item() 同步/层步） | 8.18 GPU / 8.68 CPU wall | 1.01×GPU / 1.11×wall | 2/2 PASS |

## Iterations

### [baseline] 初始化 bench 与分阶段计时
- 结果：exp/results_efficiency/e127_stage_bench_baseline.json（4 档 S × normal/skip 层）
- 分析：S=64K 每层每步 GPU 8.29ms（prepare_index 2.89 / compute_score 2.80 /
  mask 0.30 / metrics 0.07 / eager 2.23），×36 层 ≈ 300ms/token。单步 137 CUDA
  launch。三大 O(S) 全量重算 + 72 次/步 host 同步是结构性热点。

### [iter 1] metrics 惰性化
- 改动：sparse_attn/metrics.py —— add_select_result 增 env 门（TLI_E127_FAST=1）
  惰性路径：GPU 端存 (where(cond, sum, -1), bs, H) 三元组，get_select_tokens
  读取端一次性逐条 .item() 后做与旧路径完全相同的 Python float64 算术。
- 原理：旧路径每层每 decode step 做 q_ids[i].item() + b_mask.sum().item() 两次
  host 同步（36 层 × 每 token = 72 次），每次同步 = 排空整个 launch 队列，
  CPU 无法跨层跑在 GPU 前面。惰性化把同步推迟到生成结束后的唯一一次读取。
- 数字（S=64K normal 层）：metrics 阶段 CPU 3.72ms → 0.09ms；全步 CPU wall
  9.62 → 8.68ms（-10%）；GPU busy 基本不变（符合预期——本项优化的是停顿不是
  计算）。drain 停顿现转移至 eager_decoding 的 .item()（iter 3 消除）。
- 门禁：exp/bench/test_e127_gates.py metrics 2/2 PASS
  （get_select_tokens 逐位一致 old=21887.578125 == lazy；sync_debug_mode=error
  下 20 次调用零同步异常）。
- commit：见 git log
