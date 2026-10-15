# E139 活跃验收脚本：不一致必须终止，全部验收完成后才测速

## 身份与范围

- 延续 **TL-E139-GRAPH-GATE-20261015 / P2 测试可靠性**，不为同一问题新建 ID。
- 修前 main：`9449c8e9d827f4c079f9ecff5ce2cb8243273d58`。
- 对照实验分支 `e138-ablation`：`e1edf9c316fc4190dd0c29bc5d47477136f8bdab`；本轮不合并实验分支。
- 本轮仅修 `exp/bench/e139_bs_graph_bitdiff.py` 与
  `exp/bench/e139_step2_cand_bitdiff.py` 的验收控制流，增加共享门禁和进程回归。
  不修改 selector、attention/kernel、chunk2048、预算、near/far、sink/SWA、topk 或实验数据。
- 状态：上述两份活跃脚本 **fixed/rechecked（CPU 进程控制）**；历史 PoC
  `e139_l2_graph_poc.py`、`e139_l2_subseg.py` 不改，不能因此把原问题的全部历史范围标成已修。

## 根因与修复位置

修前 Step1 `:90–130` 只打印 d0/d_cap/d，非零仍继续性能测量；Step2
`:92–196` 累加 fail 却没有失败退出，而且在 K2=8192 验收之前已经输出
K2=1024 的性能。`zip` 忽略行数差异，`assert` 对 None/eligibility 的检查在
`python -O` 下消失。故调用者不能凭退出码或 DONE 认定正确性通过。

修后：

1. `exp/bench/e139_validation_gates.py:5–28`：先核对两臂均为完整列表、
   行数等于 group_ends、每行是 bool Tensor 且形状为 `(Hkv, group_end)`、
   对应设备相同，再逐位比较。当前脚本输入 Hkv=8；helper 显式允许传入其他 head 数。
   None、截短/额外行、shape/dtype/设备不匹配、bit diff 均 `SystemExit` 非零退出，
   不依赖 assert。错误包含 gate/K2/layer 或失败行位置。
2. Step1 `:86–116` 的各检查点调用共享门禁；失败不能到达测速与 DONE。
3. Step2 `validate(K2)` 完成 1024 和 8192 两档全部检查后才进入四臂测速。
   验收局部 tensor 随函数返回释放，测速重新创建 seed=999、K2=1024 的 indexer/QKV；
   原 2 次 warmup、7 次计时取中位数、CUDA 同步口径不变。移除重复 token-diff
   计数，只保留逐位相等与 diff_rows；不以此宣称性能收益。

## 修前失败与修后验证

`exp/bench/test_e139_validation_gates.py` 用 subprocess 执行真实顶层脚本，
仅替换 torch/CUDA、输入生成和 selector 依赖为不分配 GPU 的小型 test double。
mask 元数据仍覆盖 1008 个 group_end，故可以验证真实分支/退出行为，但**不是 GPU
selector 计算、Graph replay 或质量验证**。假的计时值一律不作为性能数据。

修前同一 main 文件完整执行，注入第 3 次 selector 返回最后一行错误值，实测退出码 0：

```text
[G1] capture 层 graph vs eager diff_rows = 1/1008
BENCH_ENTERED
选择链 wall: ...
=== DONE ===
```

修后同注入，退出码 1，且没有 BENCH_ENTERED、wall 或 DONE：

```text
[G1] capture 层 graph vs eager diff_rows = 1/1008
validation failed: [G1] capture 层 graph vs eager: 1/1008 mask rows differ
```

K2=8192 最后一层 replay 注入错误同样在任何测速前退出：

```text
[G2 K2=8192] 层 2（seed 2002）replay vs eager diff_rows = 1/1008
validation failed: [G2 K2=8192] 层 2（seed 2002）replay vs eager: 1/1008 mask rows differ
```

执行命令及结果：

```bash
python exp/bench/test_e139_validation_gates.py
# 2 test methods OK；内含普通/-O 两种解释器、66 个故障场景 + 4 个正常场景。
# 故障覆盖每个比较阶段，另含 None、少行/空/多行、dtype/shape/device、非 Tensor、ineligible。
PYTHONPATH=/tmp/twolevel-cpu-deps:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s sparse_attn/ops/tests -v
# 15 tests OK
PYTHONPATH=/tmp/twolevel-cpu-deps:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -O -m unittest discover -s sparse_attn/ops/tests -q
# 15 tests OK
```

额外以真实 torch CPU bool tensor `(2,3)/(2,7)` 调用共享 helper：等值通过；
bit/dtype/截短/shape/None 五种故障全部拒绝。测试夹具定义 SHA256：
`53e55875cbf05603b0276fdbfc3226d64b5a3d8b844e12a0fb2e8846fd3c646e`
（`exp/bench/test_e139_validation_gates.py`；这是注入夹具源码身份，不是 GPU QKV hash）。

环境：Linux x86_64，Python 3.12.14，torch 2.8.0+cpu；没有 nvidia-smi 或
`/dev/nvidia*`，torch.cuda.is_available=False。CUDA 设备/架构、驱动、CUDA 编译器、
Triton 和生产 kernel 配置未实测。测试不依赖 CUDA，也未占用远端训练资源。

## 历史影响、边界与下一检查点

- 纠正 `2026-10-15_e139_graph_oracle_by_opus.md` 中“两份活跃脚本已全部 fail-closed”
  的描述：该描述在本次修前 SHA 上不成立，本次才落实。保留原文及历史结果，不覆盖历史。
- 缺陷表示错误可被放行，并不证明已有 0/1008 日志为假，也不据此否定已跑质量数据。
  缺少完整逐项日志、只依赖成功退出/DONE 的验收不能追认为通过，需在同配置下重测。
- 本轮没有 GPU 正确性、速度、峰值显存或端到端 TTFT 数字。修前后 Step2 验收与测速
  顺序、存活 tensor 集合有变化；将来比较速度必须统一脚本与 warmup，不能混拼旧数。
- 相同输入 eager 两次只说明重复性，不证明相对旧实现等价；本次未扩大其证据含义。
  Graph=1 的内部 eager fallback 是否真正触发 Graph，仍需设备侧派发/缓存计数证据。
- 后续需在获准且空闲的 GPU 上重跑这两份脚本及 E140 cache/schema gates，保留实际
  QKV hash、软件/设备版本、缓存/派发信息；再验证独立分支上的 compact-block 候选。
  未经 GPU 验证的 kernel 候选仍不合入 main，本次不报告 prefill 提速。
