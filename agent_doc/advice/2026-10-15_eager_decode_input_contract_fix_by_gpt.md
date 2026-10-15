# Decode 消费端单 query 与 Q/K 宽度防线

## 范围、版本与结论

main 基线 `bb3991f1bf982f8be491bb8a4b907af8e1214a74`，fresh 独立检出、工作树原本干净；fetch 后无并发变化。实验分支 `e138-ablation` 仍为 `2ed6a1feb78a7b5bcf3cac922b4510c3968714a5`，未合并或修改。未发现本地占用锁或同一检出的其他执行者；没有连接/占用远端训练资源。

**TL-E127-DECODE-INPUT-20261015：P2，confirmed → fixed/CPU rechecked。** 公共 decode ops 缺失 single-query 形状守卫，允许不支持的多个 query 进入，静默只计算首行。Qwen/Llama patch 的 B2 已在调用者拒绝此形态，此次补消费端防线，不将已有 B2 描述重新算作新的生产污染。

**TL-E127-QK-WIDTH-20261015：潜在 P1，host 派发缺口 confirmed → fixed/CPU rechecked；GPU 后果未实测。** Q/K head_dim 不等仍通过校验并可触发 fused launch；内核用 K 的 D 加载 Q，短 Q 存在静态越界风险。CPU launch spy 确认修前可到封装，不能冒称已经运行 GPU 越界检测。

## 根因与修复

真实入口为 `sparse_attn/ops/eager_decoding.py::eager_decoding_attn`（修后:411）→ `_validate_gqa_shape`（:294–302）→ dense/gather → 可选 Triton。dense 与 torch gather 在 squeeze batch 后使用 `q[0]`、`mask[0]`，分配全 query 形状的输出却只赋值 `o[0]`。原 validator 只看 head、K/V 关系和 mask 行数，遗漏 leading dimensions 及 Q/K 宽度。

修复只增加 host 元数据条件：Q/K/V/mask rank4；Q=[1,1,H,Dq]，K/V batch=1，mask=[1,1,R,blocks]；Dq=Dk>0。公共入口及直接 dense/gather 均复用该门，在分配、读取设备值或 launch 前拒绝。保留已有 GQA 和 MLA Dv≠Dk 契约、合法非连续布局，未改 kernel、选择器、预算、near/far/sink/SWA、chunk2048 或任何实验数据。

## 最小复现与原始短日志

PyTorch2.8.0+cpu，Python3.12.14，Linux x86_64（可见 AMD EPYC9V74 KVM vCPU），einops0.8.2，CUDA不可用。仅临时安装 CPU 依赖；无 GPU/驱动/编译 kernel/真实模型运行。

```python
import os, torch
from sparse_attn.ops.eager_decoding import eager_decoding_attn
q = torch.ones(1, 2, 2, 4)
k = v = torch.ones(1, 3, 1, 4)
m = torch.ones(1, 2, 1, 3, dtype=torch.bool)
cu = torch.tensor([0, 3])
for fast in ('0', '1'):
    os.environ['TLI_E127_FAST'] = fast
    print(eager_decoding_attn(q, k, v, m, 1, cu))
```

修前两模式均返回 shape[1,2,2,4]：query0 全1、query1 全0；若计算两行，其选中-token参考应均为1。修后此 decode-only API 明确抛 ValueError，不通过增加多 query 近似计算来掩盖不支持形态。

新增 `sparse_attn/ops/tests/test_eager_decode_input_contract.py`：多 query、batch、rank、mask leading dimensions、Q/K 宽度、分配前拒绝和合法共享/per-q-head mask。Q/K 宽度例 q D32/128、K/V D64、T16384，用 launch spy 验证未派发，无 GPU 执行。

```text
修前新增回归：Ran 4 tests in 0.028s
FAILED (failures=14, errors=14)
修后全 ops：Ran 41 tests in 0.809s; OK
修后全 ops -O：Ran 41 tests in 0.825s; OK
合法路径：legal_route_exact_parity 48
```

```bash
PYTHONPATH=/tmp/twolevel-cpu-deps:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s sparse_attn/ops/tests
PYTHONPATH=/tmp/twolevel-cpu-deps:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -O -m unittest discover -s sparse_attn/ops/tests
```

48 个合法路径比较使用 git show 基线源、独立 namespace、相同张量及 seed1510：float32/bfloat16，Dk/Dv=8/8和192/128，T=1/7/16384，H4/Hkv2，共享/per-q-head，两种 FAST 模式；输出修前修后逐位相同。按循环各 q/k/v/mask/cu 连续 uint8 原始字节拼接（两种 FAST 共用同一组输入）SHA256：`8cd74ee36a2f01292b414d1c61a257312ae3295736677e17a8b2aa0d24f24e8d`。这是 CPU 合法路径对照，不代替 GPU 派发/数值验收。

独立上下文只读审核 diff，并实跑新增4项与另构造的12个CPU selected-token oracle对照（MHA/GQA、共享/per-head、MLA异维V、非连续Q、dense/gather），均通过，结论usable-with-scope；无GPU/性能结论。冻结生产文件SHA256：`bcbd0f081d772f6e95e65538d66b09229a7eebc8fcb24131c3d8940228780263`；新测试SHA256：`91dc9aef8256e2b064d839eac99e50e173c3a3409e5e987ad54dbd5dcd1b5b59`。

## 影响、成本与未验证范围

当前 Qwen/Llama/MLA 上层合法 decode 都为 B1Q1、Dq=Dk，未发现历史实验触发证据；不要求因公共接口缺口整体重跑旧分数。若外部消费者直接调用 ops 的不支持形态，应拒绝或另用真正多 query 后端，并重跑受影响数据。

守卫只读取 shape/dim，不新增 `.item()`、设备归约、复制或同步；没有 GPU 性能测量，不宣称零开销或提速。CPU 测试耗时也不是 benchmark。

未修复清单：此前 MLA decode 非零4D padding/bias 未消费的问题仍在；Q 特征 stride 的 fused 假设需单独检查及 GPU 实测，本轮宽度守卫不代表任意布局全面安全；GPU compact-block/overlap 候选继续保持未验证状态。本次未修改 protected guide、图源或凭据，不提交大日志。
