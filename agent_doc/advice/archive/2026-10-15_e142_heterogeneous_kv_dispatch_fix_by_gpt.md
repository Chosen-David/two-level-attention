# E142 异维 K/V：显式隔离不兼容的 Triton 消费路径

## 结论与版本

- **TL-E142-HETERO-KV-DISPATCH-20261015，P1，fixed/rechecked（host 派发 + CPU 数学回归）**。
- 基线 `7aa0c1518e1fdda8f35a884326dbbaefb72cc060`；本轮先检查 `13be23c..7aa0c15` 的 E142 MLA 接入。实验分支仍为 `e1edf9c316fc4190dd0c29bc5d47477136f8bdab`，未合并。
- 修复仅涉及 `sparse_attn/ops/eager_decoding.py` 的 host 布局判定，不改 Triton kernel、选择器、算法预算、chunk2048 或近似方式。新增测试位于 `sparse_attn/ops/tests/test_eager_kv_hetero_dispatch.py`。
- 没有 GPU 内存错误实测、真实模型/数据对拍或性能结论；CPU 派发 spy 不等于运行 GPU。

## 入口、触发与原行为

E142 为 `eager_decoding_attn(..., allow_kv_hetero=True)` 放宽 V head_dim/stride 校验，以支持 MLA QK=192、V=128。基线 `eager_decoding.py:319-327` 只要求 V 末维连续；但 `:554-576` 的 fused 路由没有重新校验 K/V head_dim 和 stride 相等，仅靠 D=192 不是二次幂偶然避开此路径。

这不是说当前 DeepSeek 192/128 配置已触发越界：**该配置仍会回退 torch**。新增公开参数允许的其他输入，例如 Dk=64、Dv=32/128，或 Dk=Dv=64 但 token/head stride 不同，也能通过 validator。T≥16384、block_size=1、Triton 可用、W×8≤T 时，它们会进入不兼容 fused wrapper。

基线 `:185-223` 内核同时用 `stride_kt/stride_kh/D` 寻址 K 和 V；`:377-382` wrapper 的输出宽度也固定为 Dk。故可能错误寻址、返回错误宽度；较小 V 可越界。例如 T=16384/Hkv=2/Dk=64/Dv=32，选末 token，kernel 推导的最大 V 元素偏移为 2,097,151，而连续 V 的最大合法偏移仅 1,048,575。此地址结论来自源码算术，**没有故意在 GPU 上执行越界**。

根因是把“torch 可接受的异构布局”混同于“共享 K/V 地址参数的 kernel ABI”。捕获 Python 异常后回退不足以保证安全：错误 GPU 地址可能异步报错，或静默读到错误值。

## 最小修复与取舍

1. 增加 `_triton_kv_layout_compatible`，显式比较 K/V shape 及 token/head/element stride。
2. fused 派发前必须通过该门；不兼容布局继续现有 torch gather/dense 路径，不全局关闭 Triton。
3. `_eager_gather_triton` 再做同一检查，直呼 wrapper 也在分配/cast/launch 之前拒绝。
4. 保留相同非连续 K/V 布局和 `allow_kv_hetero=False` 的原行为，不强制 `.contiguous()`，不增复制。全部判断为固定数量 host 元数据比较，无 `.item()` 或新增 GPU 同步；没有测量这几个 host 比较的延迟，不能宣称零性能开销。

未选择改写 kernel 的 V 寻址及输出维度：那需要新增 GPU 正确性、memcheck 和性能验收。当前最小 host 修复避免扩大 GPU 风险，保留正常同布局快路径；192/128 MLA 仍走原来的 torch 路径，不是 MLA 提速补丁。

## 修前红证、修后回归

CPU：Python 3.12.14，torch 2.8.0+cpu，einops 0.8.2，x86_64，OMP/BLAS 各 1 thread；`torch.cuda.is_available()=False`。无可用 GPU 架构/驱动/CUDA 编译器证据，不占远端训练资源。

测试设置 `_HAS_TRITON=True` 并用 launch spy 替换 fused wrapper，仅检测**实际 host 调用链**是否选错后端；fallback 使用真实 CPU torch 运算，对照独立逐 head、显式选 token 的 softmax attention。共享/per-query-head mask 都覆盖，含 V 较小/较大、token 间隙、head-major 布局、192/128 MLA、短序列和 dense 显式开关；相同连续及非连续布局均检查 fused 路由仍存在。

默认红证：seed=142，float32，Q=[1,1,4,64]、K=[1,16384,2,64]、V=[1,16384,2,32]，每行选 3 token（含首尾），W_upper=32，block_size=1，cu=[0,16384] int64。输入按 q/k/v/mask/block_size/cu 顺序串接字节 SHA256：`95f7c969ada145bdd9c33db2e397651665b3d49ee65a2fd07b2ab85a1131f5b4`。

```text
BASELINE: tests=5 failures=9 errors=0
8 个异维/异 stride × mask 模式子例：incompatible K/V reached fused launch
直呼 wrapper 子例：AssertionError: allocated（没有提前拒绝）

修后 python -m unittest discover -s sparse_attn/ops/tests -q
Ran 20 tests ... OK
修后 python -O -m unittest discover -s sparse_attn/ops/tests -q
Ran 20 tests ... OK
既有 test_e132a_fix_111_116.py 的 test_116_v_shape_gate：6 pass / 0 fail
git diff --check：通过
```

完整 suite 的 20 项包含原有 15 项和新增 5 个测试方法；子例数不冒充方法数。fallback float32 数学误差容限预设 `atol=rtol=1e-5`。正常同布局 fused 只验证派发，未执行 kernel，因此不能从 spy 返回值推导数值或速度正确。

复现命令（需要现有 CPU torch/einops 环境）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest sparse_attn.ops.tests.test_eager_kv_hetero_dispatch -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -O -m unittest discover -s sparse_attn/ops/tests -q
```

## 历史影响、未覆盖与下一步

- 旧默认 `allow_kv_hetero=False` 的同布局路径仍受原 validator 保护；不据此否定历史 Qwen 性能结果。当前 MLA 192/128 也没有该误派发条件，未发现本问题已污染的历史数值。
- 需重新验证任何使用 `allow_kv_hetero=True` 且 Dk 是二次幂或 K/V stride 不同的实验；此前结果不能凭“没有抛异常”认可。
- E141 Q/K 重建报告仍未被源码修复；E142 的真实 attention patch 使用模型自身 RoPE helper，且 decoder 先 norm，因此**不能把 E141 probe 的缺项直接认定为该 patch 同样缺项**。但其注释引用的 rope-only 质量结论仍需要 E141 重测支持。本轮未重复提交旧发现。
- 新 MLA patch 的完整模型 prefill/decode、padding/mask、首次单 token 和 cache 复用语义未完成验证。主线 kernel 没有 GPU 验收；Graph 实际 replay 证明及实验 block-list 性能候选继续保留检查点。
- 下次先核对远端改动；有目标 GPU 时先做安全合法输入的数值差分和派发跟踪，再测完整成本，不重放修前越界 kernel。维持 guide 只读，不覆盖原始数据或其他人的实验。
