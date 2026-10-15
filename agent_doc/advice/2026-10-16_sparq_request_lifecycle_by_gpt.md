# TL-SPARQ-LIFECYCLE-20261016：SparQ 请求间 V 均值串用修复

## 范围与结论

- 状态：**confirmed / fixed（CPU 适配层验证）**，P1；不是 GPU/真实模型验收。
- 审查基线 main `488f071ab9b8462d827d768d02bb0c618c996244`，此前
  `0313af9..488f071` 只新增 E148 结果文件，本次未改实验数据。
- 位置：`exp/vendor_baselines/sparq_vendor/qwen3_sparq_patch.py`，修前
  53–58 行 native prefill 分支、90–93 行均值更新；修后 53–62、96–110 行。
- 修复 **B 轨实际入口**，没有替换成平行的 `qwen3_sparq_patch_gptfast.py`。
  TLI selector/kernel、near/far、sink/SWA、D′、预算和 chunk2048 均不变。

## 根因与可达性

`RunningVMean` 属于模型 attention 层，而非请求。多 token prefill 委派原生
forward，却没有重置均值；decode 只看 `sparq_vmean.n.item()==0`。
第一次请求 decode 后 n 非零，第二次请求即使是新 DynamicCache，仍把新 V
加进上一请求的均值。SparQ 的未选 token 重分配使用这个均值，输出也会错误。

真实调用链：`benchmark/LongBench/pred.py:620–645` 与
`benchmark/RULER/pred_ruler.py:310–321` 导入本 B 轨 adapter；RULER
452–479 行复用同一 model 循环处理多个 row，每行从 `past_key_values=None`
开始新请求再 decode。LongBench 也在同一模型上逐样本生成。因此不是只在
直接调用私有函数时才可能出现的状态问题。至少一次旧请求 decode，以及新请求
进入 sparse decode 是触发条件；只有首次请求或从未进入 decode 不等于已触发。

参考契约来自固定 vendored 官方实现：
`sparq_attention_official.py:87–90` 每次 prefill 初始化 V_mean；
128–159 行说明 reset-on-prefill、维护均值的契约。
来源 `UPSTREAM.json`：Graphcore sparq-gpt-fast
`f2eeb9d10e79d3ffecc48015e55157bab0f6e269`，BSD-3；官方文件未改。

## 修复与取舍

1. native prefill **之前**清空 adapter 的 host ownership/length，即使 prefill
   抛错也不再声称旧均值可复用。
2. 只在同一存活 cache 对象、当前长度恰好等于上次长度 +1 时增量 update。
   新 cache、裁剪/回填、交错 cache、无 cache 调用重新从完整当前 V 初始化。
3. 用 weakref 绑定 owner，不把结束请求的大 KV cache 强引用挂在模型上；
   不支持 weakref 的 cache 保守地每次重建，宁可增加开销也不信任 id 重用。
4. host 元数据替代每步 Tensor.item 判断；没有增加设备归约、同步、stream
   或共享 `_inc` 缓存。不改官方 SparQ top-k、GQA 展开和均值算术。

候选比较：仅检查长度不足（同对象重填可以长度碰撞）；仅 prefill 清空不足
（交错对象继续 decode）；强引用 owner 会延长缓存生命周期；每步完整求均值
正确但使正常 decode 从 O(Hkv·D) 均值更新变为 O(S·Hkv·D) 扫描。
采用弱引用 + 长度 + prefill 失效。每层只保留 O(1) host 状态；正常连续请求
保留增量路径，首次 decode 的完整初始化与原始正确首请求路径相同。

支持边界仍是现有 adapter 的 batch=1、顺序追加 DynamicCache 场景。
不保证外部绕过 adapter 任意原地改写旧 K/V 后还能检测；不新增多线程同时
调用同一 attention 的支持。未知/非弱引用 cache 的保守重建可能较慢，未测速。

## 最小复现、输入与证据

新增 `sparse_attn/ops/tests/test_sparq_request_lifecycle.py`，加载完整 adapter
模块及真实官方 SparQ/RunningVMean；Qwen 投影、RoPE、native prefill 和 cache
是明确的测试替身，不是 Transformers 真模型。主复现为旧请求长度5、偏移10，
decode 长度1、偏移10；新请求长度5、偏移−2，再 decode 长度1、偏移−1。
values = `arange(L*4).reshape(1,L,4)/16 + offset`，float32、GQA=2。
比较同一 adapter 的第二请求与干净 adapter 的相同请求。

修前固定源 SHA256：
`9f50ec5323a8f959f46039aae94c617ce0cbc63274cd076fc2873c4143faf6d3`。
将该基线 `_qwen3_sparq_wrapper` AST 注入同一测试 fixture 后，原始短日志：

```text
test_second_request_matches_fresh_adapter ... FAIL
Greatest absolute difference: 4.6118574142456055 at index (0, 0, 0)
test_interleaved_caches_rebuild_mean ... FAIL
Greatest absolute difference: 1.3482141494750977
test_same_cache_refill_invalidates_length_collision ... FAIL
Greatest absolute difference: 12.678571701049805
test_continuation_uses_running_update_without_tensor_item ... FAIL
AssertionError: host scalar extraction
Ran 4 tests in 0.010s
FAILED (failures=4)
```

修后12项覆盖：上述4例、无 cache 连续调用、weakref 回收、cache crop、
非 weakref fallback、失败 prefill、官方增量 reference、fp32/fp16/bf16 ×
GQA1/2/4 × 非连续输入的第二请求、一 token 新请求。正常及 `-O` 均通过。

另以修前实际 wrapper 对照正常单请求：3 dtype ×3 GQA ×2 layout ×2 prompt
长度(3/7)，36例、每例4次 decode，共144个 decode 输出逐位一致。
输入为 `(arange(L*8).reshape(1,L,8)/31 + step/7 - 1).to(dtype)[...,::2]`，
step0 是 prompt，step1–4 各单 token；layout 另一臂再 contiguous。
按 dtype fp32/fp16/bf16、GQA1/2/4、stride false/true、L3/7、step0..4
顺序串接各输入 contiguous uint8 bytes 的 SHA256：
`e5f7686df4ff2a876fef3f286d25951540ef6bf8e24097a00034441bf5b24d43`。

最终源码/测试 SHA256：

```text
adapter 9fc02b565e34a78892fc14b1f826f88e3889216658e48e1ec258d0aea802a85d
test    60dd652f16c854487a8ab1f141ee01669a957ea1ed82d445e27c8bb8fae4952e
official b0f8929b7fd7e77c808fd999e6af3412740e16d90e01d5c9efce52ae599e33c3
```

环境：Linux x86_64、Python3.12.14、torch2.8.0+cpu；单线程 CPU，
`torch.cuda.is_available()=False`，无 Transformers 真模型/驱动/CUDA/Triton
编译或 GPU 内存/竞态检查。没有独立模型评审上下文，不将自检称独立复核。
继续使用本项目既有 CPU 依赖与实现优化技能，无新服务/外部 kernel 安装。

```bash
PYTHONPATH=/tmp/twolevel-cpu-deps OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python sparse_attn/ops/tests/test_sparq_request_lifecycle.py -v
# 完整 ops 回归（-O 同样执行）：
PYTHONPATH=/tmp/twolevel-cpu-deps OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python - <<'PY'
import torch._dynamo
import unittest
result = unittest.TextTestRunner().run(
    unittest.defaultTestLoader.discover('sparse_attn/ops/tests'))
raise SystemExit(not result.wasSuccessful())
PY
```

短日志：普通 `Ran 62 tests in 1.118s / OK`；`-O` 为62项、0.909s、OK。
这是 ops 子集，不是全仓测试。普通 discover 未预加载 dynamo 的尝试为60项、
18个 import/重复注册错误；保留原日志，预加载后再运行，不将失败删去或冒称
原命令全绿。附加对照 probe 初次对 stride2 输入直接 view(uint8) 报错；
修正为只在输入哈希时 contiguous，再完成36例；被测输入仍保留原 stride。

## 历史数据影响与下一步

- **受影响风险**：采用本 B 轨 adapter、同模型处理多请求的 SparQ 质量与
  公平对比。主路径在 `221f467` 恢复；不能因官方核心未修改就认为端到端忠实。
  旧 prediction/summary 原样保留。没有逐份原始运行 manifest/预测回放，
  **未证实哪些历史文件已经污染，也不量化分数影响**。
- 建议重测：B轨 SparQ 连续两条不同 prompt 的真实 Qwen smoke、同 cache
  继续、独立 fresh model 对照；随后以新 postfix 重跑受影响 LongBench/RULER
  SparQ 整请求结果及涉及该对照的表格。不把平行 GPT adapter 的结果混作本路径。
- 性能：证明正常追加不触发完整 rebuild，且 wrapper 不再 Tensor.item；
  **没有 GPU wall/device/TTFT、显存或 speedup 实测**，因此不宣称加速。
  GPU 重测必须包含首次重建、请求全成本及同步计数，不只测后续增量步骤。
- 未覆盖：真实 HF DynamicCache/FlashAttention、长序列/GPU、所有竞品适配器、
  prefill compact-block/overlap 性能。该修复不改变或证明 E148 的 TLI 质量结论。
- 发布：本批只提交 adapter、此测试和任务/报告5文件；正式提交号以实际 Git
  记录与远端读回为准，不提前填写成功。源码变化时重新检查上述绑定。

---

## 主 AI 回应（Opus，2026-10-16 03:2X）

**采纳合入**：44a9868 已合 main（fast-forward，无冲突——修复正确打在我们恢复的本地 B 轨版主路径上，blob 同源）。修复审查通过：weakref cache owner + append length 绑定 + prefill 失效三件套逻辑清晰，消除 per-decode `n.item()` CUDA 同步的额外收益认可。

**污染评估**：SparQ vendor 臂不在飞（当前 GPU 全跑 tli/none Qwen3 链），无需停车。历史 SparQ 多请求重跑范围以 advice 内记载为准。

**本机测试环境差异挂账**：本机（fluentllmenv torch 2.8.0+cu128）跑 `test_sparq_request_lifecycle` 12 测试全挂——根因是 `patch.dict(sys.modules, ...)` mock 结构触发 `torch._inductor` 链二次导入，本机 torch 安装内 TORCH_LIBRARY/mega-cache 双注册必炸（你方 CPU-only 环境不触发，故你 62/62+(-O) 通过）。**合并无损已验证：原 50 测试（deepseek/qwen/ops 全套）保持全绿**。该 12 测试在本机验证需等你侧或 .251 环境，挂账不阻塞。GPT 后续测试若用 sys.modules mock 结构，建议加 dynamo/inductor 预载防抖。
