# TL-STATIC-MASK-20261016：恢复授权后的当前 main 复验与修复

日期：2026-10-18。来源：用户明确恢复 TwoLevel 修改及直推 main，并要求保持定时任务运行。
本轮起点 `4828e4df4a236692379778d80426fd9d0774cec6`；发布前同步
`784c1a4038417a084e087c6df05e46f9d26c8e41`（仅他人的论文文档增量）。
实际实验分支 `e138-ablation` 为 `2ed6a1feb78a7b5bcf3cac922b4510c3968714a5`，只读对照，未合并。
原工作区未跟踪文件、旧暂停锁和历史结果保留；本轮使用独立 detached 检出。
两个映射的 `/home/wangyuanshuo02/sglang/` 人类文件在本执行环境不存在；已读本仓 guide，未修改任何 guide 文件。

## 问题、触发与根因

生产文件 `sparse_attn/patches/qwen3_attn_patch.py` 与暂停时 `48e616e` 的该文件完全相同，
SHA256 `fa94cf350630d31c870b19ca8e3d313bec5a68d4f0081fad704dcf492b3d024f`。
本轮不是把旧通过记录当作当前验收：已重建 CPU 环境并重新复现。

- 原 L117–137：`equal(mask, mask.tril())` 仅排除上三角真值。全零 mask 或缺少历史边的下三角子集也通过，随后被替换成 `None`，SDPA 恢复本应屏蔽的边。
- 原 L221–233：decode 未要求 bool，非零 float additive bias 也可能被当成恒等 mask；层间仅复用一个布尔判据，另一 tensor 或原地修改后的 mask 没有重新验证。
- 原 L59：`get_max_cache_shape() is not None` 将 Transformers 4.56.2 的 DynamicCache 无上限哨兵 `-1` 当作 StaticCache，产生无必要的长度标量读取及静态分支状态。

## 修复及契约

- 当前 L31–60 新 helper：prefill 必须精确等于全真下三角；decode 必须为全真有效前缀；容量尾部必须全假。仅接受对应形状的 bool mask，禁止吞掉 float bias。
- 复用绑定同一 live tensor、版本、长度、阶段、形状、stride、dtype 和设备；使用 weakref 避免 cache 延长大 mask 生命周期。没有版本计数的 inference tensor 每次重验。
- 当前 L95–119：正容量才走 StaticCache；`None`/`-1` 走动态长度接口。
- 两个实际调用点均接入 helper，继续限制无 sliding window。非恒等 prefill mask 保留给原后端；不支持的 decode mask 继续明确拒绝。
- 不改 kernel、选择预算、chunk2048 或数学近似。未扩展 sparse decode 的 padding/bias 支持；也未声称拒绝后的 cache 更新可回滚。

## 当前验证

环境：Linux x86_64，容器可见 Intel Xeon Platinum 8573C，Python 3.12.14，
Torch 2.8.0+cpu，Transformers 4.56.2，einops 0.8.1。无 CUDA/Triton、驱动或 GPU 运行证据。
固定种子 20261018；小型随机初始化 Qwen3Attention（hidden16、Q4/KV2 heads、D4、L3），
没有下载模型权重。输入 hash 和源码 hash 见结果 JSON。

测试通过 AST 加载**实际生产函数体**以排除无关 CUDA indexer 导入；实际调用 HF StaticCache/DynamicCache、Qwen3 层、RoPE 与 CPU SDPA。不是全模型或原模块完整导入验收。

```text
修前：Ran 4 tests ... FAILED (failures=3)
修后：test_qwen_static_mask.py      Ran 12 tests ... OK （普通 / -O）
      sparse_attn/ops/tests         Ran 66 tests ... OK （普通 / -O）
      tests/test_e152*.py           Ran 16 tests ... OK （普通 / -O）
      test_longbench_request_capture.py Ran 11 tests ... OK （普通 / -O）
```

共 105 项指定 CPU 测试在普通与 `-O` 下分别通过，不是全仓测试。
专项包括完整 512 种 3×3 bool mask、全屏蔽行输出为零、跨层替换、原地与 alias 修改、
inference tensor、非连续 stride、尾容量、非法形状、浮点 bias、弱引用释放及静态/动态合法输出逐位一致。
原始失败和通过日志、复现脚本、输入/源码身份存于
`agent_doc/results/static-mask-resume-20261018/`。

## 性能审查：已测与否定结果

同 CPU、同输入、同权重、同 dtype 的完整单层 forward profiler 显示：
`aten::_local_scalar_dense` 从 1 次变为 0 次；输出逐位一致。
`contiguous` 均为 1 次 / 192 bytes，`clone` 均为 3 次 / 576 bytes。
最初“动态分支会多做两次 K/V contiguous”的推断被该输入反驳：原路径内部还有 contiguity 条件，进入分支不等于执行复制。
保留首版探针的失败；修订探针仅报告实际计数，未把负结果隐藏为复制收益。
这些是 CPU operator 计数，不是 latency benchmark、峰值内存或 GPU 同步节省实测。

正确性修复可能有必要开销：不同层 mask、修改后 mask、无版本的 inference tensor 将重新检查；
prefill 检查仍是 O(S²)，会生成三角参考，GPU 上存在归约/同步成本。
不得把本修复称为整体 TTFT 加速。

## 后续与历史实验边界

1. E134-pre 的同 mask/相同状态是否满足等价旁路需依据真实回执重验。当前未证明旧正式样本触发错误，不改历史数据或抹除旧数字。
2. 需在已授权且空闲的 GPU 上对真实模型验证标准/屏蔽 mask、prefill→decode、跨层与 inference_mode；测完整 TTFT/请求时间、峰值显存和同步，单流为基线。未完成 CUDA Graph/多 stream 验收。
3. 新增待核线索 TL-E151C-ROUTE-20261018：当前 batch_select `_gate_ok` 未排除 `_e151c_quota`，而批量预算仍使用固定 gamma；reference_prefill 没有 `_e151c_on` 激活逻辑。需要独立复现调用组合，再决定明确拒绝/逐组回退，不能据此撤销已有 E151c 历史分数。
4. prefill 浪费仍静态可见：reference 每行块重复 `k.to(float32)` 且检查空行会触发 host bool；candidate 组选择汇总后按全前缀 argsort，并集实际宽度还有 `.item()`。先核真实派发，再用候选分组紧凑索引路线做同语义全成本比较；本轮没有 GPU 性能结果，不晋升未验 kernel 候选。
