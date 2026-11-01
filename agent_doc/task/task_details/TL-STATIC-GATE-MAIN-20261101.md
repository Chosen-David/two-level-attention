# TL-STATIC-GATE-MAIN-20261101

## 目标
把静态 near/far gate 接入 `main`，decode 期读取预计算 skip 决策；GLM 静态标定结果随仓发布；验证 Qwen3-8B 动态标定路径仍能进入冻结/论文方法。

## 变更
- `sparse_attn/engine/frozen_runner.py`：`prefill_frozen_request` 支持 `static_gate`；有静态 gate 时只采集 core tokens，不再做动态 region profiling；预填审计对象不再携带 cache 实例，避免 JSON 序列化失败。
- `sparse_attn/policies/static_gate.py`：静态 gate schema、校验、原子读写。
- `benchmark/RULER/calibrate_static_gate.py`：单样本静态标定入口。
- `benchmark/RULER/pred_ruler.py`：`--nfa_static_gate` / `--nfa_static_gate_path`，fail-closed 加载。
- `exp/trace/results/nfa_static_gate_glm4_moe_lite.json`：GLM 47 层静态 gate artifact。

## 验证
- 本地 `py_compile`：静态 gate、RULER 入口、frozen runner、cache/kernel/info 全部通过。
- 静态 artifact 可加载：`model_type=glm4_moe_lite`，`num_layers=47`。
- 远端 Qwen3-8B 动态标定烟测通过：`10.238.122.187:8416`，`niah_single_1` L=32768，`--nfa_prefill_frozen_regions --nfa_frozen_region_mode dual`；receipt `gate_counts={'keep': 36, 'near': 0, 'far': 0, 'both': 0}`，36 层 `nfa_receipts` 均为 `graph_ready_sparse`。证明 prefill 动态标定 → freeze → decode 论文路径可运行。

## 边界
- 当前 GLM artifact 来自 `static_vs_dynamic_20261101` 的 124 个 static-dual-full256 样本按 mean mass、`epsilon=0.05` 聚合，聚合后 `skip_far=[]`、`skip_near=[]`。这是可加载的静态策略，但不会跳过任何层；如需真实 near/far 跳过，需用更严格阈值/频率口径重新标定。
