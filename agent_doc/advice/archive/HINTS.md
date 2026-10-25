# E127 任务硬约束（来自派单 prompt，持久化）

## 隔离与 git 纪律
- 工作目录：/tmp/e127_kernel_wt（git worktree，分支 e127-kernel-opt）
- **严禁 git push 到任何 remote（bee/origin）**——在飞实验链依赖主仓稳定，分支只留本地等主 AI 验收
- 主仓工作树不动；tli_indexer.py 的 md5 身份锚点 = bf5e26c4d6123875b2786b9ed886da5f（改了会污染 E125/E116b 可比性）

## 不许动的东西
- `sparse_attn/indexer/tli_indexer.py` 的选择逻辑（含 d85af51 尾巴覆盖修复，不许回退）。
  若确有必要动该文件：**停下来在报告中说明，等主 AI 决策**。
  → 规避方案：所有 selector 路径优化走**新文件子类**（bit-identical 重实现），
  通过 `patches/patch.py` 的 env 开关接线。
- 不改口径/协议/数值行为：attention 数学语义（mask、预算边界、保护区）逐位不变。
  优化只许改变计算效率。

## 正确性门禁（违反即回退）
- torch.allclose 对拍：优化前后同一输入的 attention 输出（bf16 合理 atol/rtol = 1e-2）
- selector 路径改动：score_dict 与 mask 必须 bit-identical（torch.equal）
- 既有回归套件必须绿（仓根跑）：
  - test_near_swa_boundary.py / test_near_swa_redgreen.py（near-SWA 边界）
  - test_c3_near_scale_fix.py（C3）
  - test_e122_gamma_off.py、test_e125_cluster_tail_coverage.py、test_e125_near_kmeans_sample_guard.py
  - benchmark/RULER/test_e116f_publish_atomic.py、exp/trace/test_e113_greedy_triton.py 等
  - 优化路径开启时（env 开）也要跑相关子集

## 环境
- 本机 2×H20-3e（CUDA_VISIBLE_DEVICES=0/1），torch 2.8.0+cu128，triton 3.4.0
- 模型（e2e 对拍用）：/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlp-ckpt/sunyueqing/Qwen3-8B
  （注意：~/Qwen3-8B 是远程机路径，本机不存在）
- python3 = /home/fluentllmenv/bin/python3

## 代表性配置（baseline 与优化都按此测）
- mavg 冠军：far_method=minmax, near_method=avg, α=0.25, β=0.125, γ=0.625
- K1=128 页（bs=64）/ K2=1024 token，S=8K/16K/32K/64K，D=128，GQA 32/8
- D' 层掩码启用（exp/trace/results/tli_layer_skip_mask.json，skip_ids 含 0,1,7,8-16,35）
  —— 注意部分层走 skip_far 分支，计时需覆盖 skip 与非 skip 两类层

## 交付物
1. 分支系列 commit（每步含前后数字）
2. bench 脚本入库（可复现）
3. 报告 → 主仓 agent_doc/task/task_details/S-T024.md + /tmp/e127_kernel_wt/E127_REPORT.md
   含正确 baseline 各阶段耗时表（E126 五臂计时套件会复用）
