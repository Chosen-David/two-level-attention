# 单 query decode padding 契约修复

问题 ID：**TL-DECODE-PADDING-Q-20261019**。基线 main：`bc968288c7e69ec3d93b6d2acf546a3dcd26f962`。

作者类别：GPT 代码审查执行者；日期：2026-10-19。处理：adopt，仅采纳有修前失败、CPU 实际 wrapper/consumer 验证的包装层修复；defer 生产 GPU/真实 selector 的质量与性能验收，不把独立审查结论当作新授权。

## 触发、根因与改动

Qwen3/Llama 在 batch=1、q_len=1、缓存长度 S>1 时，显式传入支持的 `[1,S]` bool padding mask（包括全 True），会用同一个 KV mask 索引长度为 1 的 Q，必然形状不匹配。生产入口 cache.update → wrapper packing → indexer.prepare_mask → eager_decoding_attn 中，错误发生在 selector 前。

- `sparse_attn/patches/llama3_attn_patch.py:83–100`；`sparse_attn/patches/qwen3_attn_patch.py:246–270`（本补丁行号）：只压紧 K/V，保留单 query。
- 守卫要求 mask 精确 `[1,S]` 且最后位置当前 query 有效；不支持的输入在 selector 前明确拒绝。全屏蔽/屏蔽当前 query 不再依赖偶然 IndexError。
- 保持已有压紧位置、L2/预算、selector、kernel、prefill chunk 和 None mask 行为；没有改成新近似算法。既有守卫仍位于 cache.update 后，不承诺拒绝输入后的 cache 事务回滚。

## 验证与原始短日志

修前在该基线真实运行专项前两个方法，使用随机初始化的 HF Attention/DynamicCache、实际生产 CPU consumer（AST 加载真实 forward，selector 为 all-selected double）：

```text
IndexError: The shape of the mask [1, 4] at index 1 does not match the shape of the indexed tensor [1, 1, 4, 4] at index 1
Ran 2 tests in 0.325s
FAILED (errors=34)
```

修后命令（分别再以 `python -O` 执行）：

```sh
OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_decode_padding_query.py -v
OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_qwen_static_mask.py -v
OMP_NUM_THREADS=1 python -m unittest discover -s sparse_attn/ops/tests -p 'test_eager_*.py' -v
```

```text
普通: Ran 6 tests in 0.934s / OK; Ran 12 tests in 0.199s / OK; Ran 15 tests in 2.047s / OK
-O:   Ran 6 tests in 0.932s / OK; Ran 12 tests in 0.210s / OK; Ran 15 tests in 1.752s / OK
decode-padding input SHA256: 713468324be3aacd8c6188a34132a8c3a7421a12f20bbbb2df6a1634efae3e85
```

合计 33 项方法，新增专项包含 192 子场景：S=2/3/4/7/65、空/错误形状 mask、所有三位前缀、屏蔽当前 query、None、非连续 mask/hidden、KV头1/2/4、float32/float64。输出对 HF SDPA `atol=1e-6, rtol=1e-5`。输入哈希覆盖参数、随机权重、输入 tensor 字节、shape/dtype/stride，seed=20261019。短序列 fast=1 仍回退 dense，**不是 gather 覆盖**。

独立上下文另外验证多步 decode、fp16/bf16/fp32 与 >16K 紧凑 KV 的实际 torch gather；见 [独立报告](../results/decode-padding-q-20261019/independent-review.md) 及同目录复跑脚本。该报告单独列出 selector double 与真实派发边界。

主控复跑便携脚本同样退出 0：342 配置、1026 步，长场景 dense 回退计数 0，原始 JSON SHA256 `2e25ff9643f3420fb120a43bb93450bcff8e6e691822eec58e2f6a967934b5f7` 与独立结果一致。

环境：Linux 6.18.55 x86_64/glibc2.39，Python3.12.14（Clang22.1.3），PyTorch2.8.0+cpu，Transformers4.56.2，单线程 OMP；无 CUDA runtime/可用 GPU，未编译新 kernel，驱动/架构 GPU 配置不适用。使用小型随机权重，不下载模型或占用未知训练任务。

## 性能、历史影响和未完成项

这是从确定崩溃恢复接口契约的正确性修复；上述 unittest 时间不是修前/修后性能基准。质量仅指测试输出数值一致，不代表实模质量。速度、峰值显存、kernel 数、TTFT 尚无可比实测，不作收益主张。

显式 2D mask 路径新增末位 bool 标量检查，CUDA 上会引入 host 同步，需在生产设备重测；原有 boolean packing/复制未优化。None mask 无此新增检查。未验证真实 selector 的增量缓存、后缀替换、GPU fused attention 或 CUDA Graph；不能把 CPU gather 当 GPU 证据。

历史成功的 None/canonical StaticCache 路径不因本问题自动失效；凡声称显式 2D padding 的 Qwen3/Llama 稀疏 decode 实验须重新确认实际派发及修前失败记录，修后在对应 GPU/真实 selector/实模重跑。未修改历史实验数据。后续继续处理候选内 L2/无效 gather 等既有优化，不因本次有界修复关闭任务。
