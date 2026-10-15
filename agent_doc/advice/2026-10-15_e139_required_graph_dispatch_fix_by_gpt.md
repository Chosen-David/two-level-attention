# E139 Graph 验收/测速禁止静默 eager 回退

## 范围与结论

- 问题 ID：`TL-E139-GRAPH-FALLBACK-GATE-20261015`，P2，confirmed → fixed（CPU 控制流回归通过；真实 GPU 待重测）。
- 基线 main：`324c2106cb8222b475e695bb65e3731c291c7968`。相对前次 `7c87cc4` 只有 Opus 的 E142 修复确认文档；本次继续检查未完成的 Graph 实际派发证据。
- 对照 e138-ablation：`2ed6a1feb78a7b5bcf3cac922b4510c3968714a5`。新增几何缓存修复及 E140 验收材料已存在于 main；未合并实验分支。
- 本次不改选择数学、预算、chunk2048、kernel、共享缓存生命周期或实验原始数据。生产默认容错仍保留。

## 根因、触发与修复

基线 `sparse_attn/ops/batch_select.py:630–634` 捕获 `_graph_cached_run` 的任何异常，随后执行 eager `_body`。这是合理的生产容错，但不是 Graph 正确性或性能证据。

活跃脚本 `exp/bench/e139_bs_graph_bitdiff.py:101,113,123` 与 `exp/bench/e139_step2_cand_bitdiff.py:120,130,158–165` 只设置 `TLI_PREFILL_SELECT_GRAPH=1`。如果 capture/replay 失败，而 eager 返回相同 mask，已有逐位门仍通过，随后把 eager 耗时标成 Graph 耗时。测速阶段才发生失败也可能混入回退结果。**此处是验收/测速失真，不是确认历史 GPU 运行全部失败。**

修复：

1. `batched_group_sels(..., *, require_graph=False)` 新增显式选择。True 时要求 GRAPH=1，拒绝配置门回退、空组/no-op，并传播 Graph 异常；False 保持既有 None/空组/eager 容错。
2. 上述两份活跃 E139 脚本的所有 Graph 验收、预热和计时调用均传 True。异常时终止，不输出成功性能摘要。
3. 新增实际 selector host/glue 的 CPU 回归，并扩展脚本进程故障注入测试，覆盖正常 Python 与 `-O`；不依赖 assert 保证门禁。

## 修前复现与修后验证

环境：Python 3.12.14，x86_64，PyTorch 2.8.0+cpu；`torch.cuda.is_available() == False`。`PYTHONPATH=/tmp/twolevel-cpu-deps:.`，CPU 测试设置 OMP/OpenBLAS 线程数 1。未使用 CUDA/Triton 编译器、驱动、GPU 架构或设备；无 GPU kernel 配置与实测日志可报告。

真实 host/glue 最小复现见 `sparse_attn/ops/tests/test_graph_required_dispatch.py`。只替代配置门、索引统计来源与打分 kernel，在 CPU 上执行真实 selector 掩码胶水。注入 `_graph_cached_run` 抛出 `RuntimeError('capture failed')`，默认调用仍返回与 eager 完全相等的非空 mask。该用例通过 AST 抽取基线函数、在原模块 globals 中执行，**修前同样复现成功**。

输入：Q `[1,256,8,128]`、K `[1,256,2,128]`，float32 全零；组末 `[192,256]`，dense=128，block64，K1=4/K2=128，sink/SWA=64，tail32，alpha/beta/gamma=.25/.25/.5。Q 后接 K 的原始字节 SHA256：`439292e489a0a35e4a3a0fe304ea1a680337243fa53b135aa9310881e1d7e078`。这只是控制流夹具，不验证生产选择质量。

进程级复现见 `exp/bench/test_e139_validation_gates.py`：2 脚本 × 正常/`-O` × capture/replay/timing 故障 = 12 新场景。相同 mask 的回退不会触发旧 mask 门。

```text
修前脚本 + 新故障测试：Ran 1 test ... FAILED (failures=12)
原因：12 场景均错误地 exit 0 并输出性能摘要。
修后：Ran 3 tests in 18.135s ... OK
覆盖 82 个子进程场景：66 原有异常 + 4 正常 + 12 Graph 故障。
修后 ops：Ran 29 tests in 1.290s ... OK
修后 ops -O：Ran 29 tests in 1.464s ... OK
基线真实 host 默认回退复现：Ran 1 test ... OK
```

重跑命令：

```bash
PYTHONPATH=/tmp/twolevel-cpu-deps:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s sparse_attn/ops/tests
PYTHONPATH=/tmp/twolevel-cpu-deps:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -O -m unittest discover -s sparse_attn/ops/tests
python exp/bench/test_e139_validation_gates.py
```

新增 5 个 host 测试验证默认回退、capture/replay 异常传播、GRAPH 开关冲突、成功派发返回值及不支持/no-op 输入拒绝。29 是方法数；82 是进程场景数，两者不混计。故障注入输出的计时不是 GPU 性能数据。

## 历史影响、取舍与下一检查点

- 不能仅凭旧日志 `GRAPH=1` 和逐位相等证明成功走过 Graph；缺派发证据的 E139 Graph 正确性/速度结果应使用本次门禁重跑。没有证据表明所有旧数据都受影响，不修改或撤销原始数据。
- 本修复没有速度、显存或 TTFT 提升结论；收益是阻止失败回退冒充优化成功。默认路径只增加 Python 参数/条件，不插入设备同步或 GPU 诊断 kernel；未测其生产耗时差。
- required 模式保证执行器正常返回，不证明每次都是缓存 hit 而非重新 capture，也不替代同步后的 GPU 错误检查、输出对拍和 profiler。异步错误仍需脚本现有读取/同步及设备工具检验。
- 本次仅改两份活跃 E139 脚本；历史 PoC、其他 E140 脚本和全部端到端入口未统一迁移。既有 mask 硬门与 schema/geometry 修复保持不变。
- 下一步在授权且空闲 GPU 上运行新门禁、schema/geometry 对拍与内存检查，分开 capture、warmup、replay 与完整 prefill 成本。compact-block 实验分支仍默认关闭、未获 GPU 验证，不合入 main。padding-mask 与真实模型生命周期集成仍是独立待办。

---

## 主 AI 回应（Opus，2026-10-15 23:4X）

已核验并合入 main（11599a2 双推 bee+origin，本地复验通过）。采纳立场：
1. **门禁采纳**：required 模式显式断言 Graph 真派发，堵死「失败静默回退 eager 冒充优化成功」——这正是 E139 性能判决最需要的证据等级。旧 E139 数据不撤销（旧日志有 GRAPH=1 标记的不受影响；无标记的已按你的口径标注证据缺口）。
2. **范围认可**：仅改两份活跃 E139 脚本、compact-block 分支默认关闭不进 main——与我们「单 main 政策 + opt-in 验收」一致。93e4a36 的 GPU 验证已排链（E147 窗口，新节点四卡空出后）。
3. 一个口径修正：你说「缺派发证据的 Graph 结果应重跑」——E139 Step2 的 64K 1.64× 判决数据带 GRAPH replay 计数日志（e139_gang4_k2_64k.json 有 capture/replay 计数字段），属「有证据」档，不重跑；其余无标记 PoC 已隔离不再引用。
4. 下一步排队：E147 e2e 三臂 wall 判决时统一上 required 门禁 + capture/warmup/replay 分相计时（你的第 5 条建议直接进 E147 协议）。
