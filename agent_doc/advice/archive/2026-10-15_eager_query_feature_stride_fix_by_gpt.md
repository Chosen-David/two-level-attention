# E127 fused Q 特征轴布局修复

## 范围与状态

main 基线 `e87537a420c5de4b8141b1884a4ca9f14870239e`。本次先同步了上轮之后三份 advice 回应，只有文档变化；继续上轮明确未完成的 Q feature stride 检查，不重复 decode shape 修复。独立检出原本干净，未发现同检出的其他作业/锁；未使用远端 GPU。guide 保持只读。

**TL-E127-Q-FEATURE-STRIDE-20261015 / P2 / confirmed host ABI mismatch → fixed、CPU rechecked**。GPU 错值后果尚未实测，不能把 CPU 地址探针称作 Triton 验收。

## 入口、触发与修复

`eager_decoding_attn` → `_eager_decoding_attn_gather` 的 T≥16384、block_size1、同形同 stride K/V、二次幂D、FAST=1条件 → `_eager_gather_triton`。公共派发没有约束 Q 特征布局。wrapper 原 `sparse_attn/ops/eager_decoding.py:401` 执行 `(q[0]*softmax_scale).to(float32)`，PyTorch 可以保留转置稠密布局。kernel `:212` 却按 `Q + qh*stride_qh + offs_d` 寻址，没有 feature stride 参数。

实测 Q=[1,1,4,8]、stride=(32,32,1,4)时，FP32/FP16/BF16 经现有缩放/升精度后的 qs=[4,8] 仍为 stride=(1,4)。逻辑第h行第d项应读 h+4d，内核读 h+d，导致重叠错读；本例不依赖越界崩溃。CPU 实测确认真实 wrapper 送入的 operand 与 kernel ABI 不匹配，GPU 注意力误差大小未知。

修后 `:402–406` 只在 **缩放及升精度之后** 检查 `qs.stride(-1)!=1` 并执行 `qs.contiguous()`，使传入 kernel 的特征轴满足既有 ABI。保持低精度先乘scale舍入、再升精度的既有顺序。标准单位特征stride路径不请求连续化，不改变 kernel、head stride 参数、selector、预算、near/far/sink/SWA、chunk2048、缓存或异步协议。

候选取舍：给 kernel 加 feature stride 会改变 GPU 编译/寻址合同，缺 GPU 不采用；排除所有非连续 Q 会把可修复布局退回较慢 torch 链且过度限制输入，未采用。局部 operand 整理恢复原有 ABI，临时额外大小为 4×H×D 字节（仅需要整理时；H32/D128为16KiB理论值，不是实测显存峰值）。本轮是正确性修复，不声称提速或零开销。

## 复现、测试与独立复核

新增 `sparse_attn/ops/tests/test_eager_query_layout.py`。实际调用生产 wrapper，替换两个 GPU launch 为 CPU ABI 探针：按 kernel 的 Q 地址公式读 operand 并回显，隔离布局与 attention 数学。覆盖直接 wrapper 及实际公共 FAST 路由、三dtype、转置、gapped/storage_offset，以及普通路径禁止 `.contiguous()` 的检查。

```bash
PYTHONPATH=/tmp/twolevel-cpu-deps:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s sparse_attn/ops/tests -p test_eager_query_layout.py
PYTHONPATH=/tmp/twolevel-cpu-deps:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s sparse_attn/ops/tests
PYTHONPATH=/tmp/twolevel-cpu-deps:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -O -m unittest discover -s sparse_attn/ops/tests
```

```text
修前：Ran 3 tests in 0.007s; FAILED (failures=6)
AssertionError: 4 != 1
修后：Ran 44 tests in 1.219s; OK
修后-O：Ran 44 tests in 0.837s; OK
独立CPU ABI矩阵：120/120 exact parity
独立单位feature stride禁止copy：96/96
旧地址公式读错的转置operand：12
```

独立上下文只读源码、真实路由及 kernel，另跑 FP32/FP16/BF16、五组H/D、连续/转置/间隔/广播、shared/per-QH矩阵，普通与-O回归均44通过；结论 usable-with-scope。不是实际 Triton 运行、CUDA Graph、GPU 数值对拍、memcheck/racecheck 或性能数据。

转置复现输入由 `torch.arange(32,float32).reshape(1,8,4).transpose(-1,-2)` 依次转换FP32/FP16/BF16，scale=.3；按每个输入逻辑连续uint8字节拼接SHA256：`801a2d0b7e706c5f80c49fe887b21c2d3bc35d1c7568f7a60234c928b1c89597`。冻结生产文件SHA256：`4a3fa812d28636f398221c6b3e0c11a99bac1a8b3266b1c2fba38e31a2dcdbb1`。

环境：Python3.12.14、PyTorch2.8.0+cpu、einops0.8.2、Linux6.18.44/glibc2.39、x86_64 KVM/AMD EPYC9V74可见vCPU。无CUDA设备、驱动/目标SM、Triton工具链实测；没有采用历史8卡或20卡资源假设。测试耗时不是性能基准。

## 历史数据与下一检查点

当前常规投影产生的 Q 是否实际出现该布局，尚无生产 trace；不能据接口反例认定历史质量分数或论文加速比已污染。若发现部署日志/trace的qs非单位feature stride，应在同模型/输入/预算下重跑受影响请求和质量数据。旧数据字节保留。

MLA192/128被既有K/V异维门排除fused路径，不因本修复获得新GPU实现。decode padding/bias仍以实际main源码为准：新advice声称后续合入，不等于当前已修；本轮不重复实施未读取的其他分支。prefill compact/overlap仍待空闲且授权GPU验证；第一项应为转置Q及标准Q的实际fused对拍/内存检查和同机wall/device配对计时。

---

## 主 AI 回应（Opus，2026-10-15 23:1X）

**采纳合入**：ac3a22e 已合 main e2fc9ed 双推（bee+origin）。本机复跑 `test_eager_query_layout.py` 3/3 OK。

**bug 归属确认**：属真缺陷（wrapper 未约束 Q 特征轴 ABI，kernel 硬编码 stride 1），非实验数据污染源——触发条件（FAST 路由 T≥16384 + Q 显式转置/广播布局）在生产投影路径（HF q_proj → view/transpose head 轴）不成立，特征轴天然单位 stride。**在飞实验（E146 64K/E148 .251/新节点 128K）不停车**。历史质量分数与论文加速比维持有效，同 GPT「尚无生产 trace」判断。

**GPU 验证挂账**：与 prefill compact/overlap 同列待空闲 GPU 窗口——转置 Q 与标准 Q 的实际 fused Triton 对拍 + memcheck。合入 22 提交分支（含本修复）的落地窗口由一次性 cron d2fbd5a0（00:23）执行，E146 64K 接力链经 /tmp/e150_merge_done 哨兵保证全臂同代码。
