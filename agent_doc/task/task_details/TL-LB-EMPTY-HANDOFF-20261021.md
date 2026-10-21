# TL-LB-EMPTY-HANDOFF-20261021

## Plan

来源：持续代码审查要求覆盖空/短输入与 prefill→decode 状态交接。基线
`58c0d437c86c3cf568eadf076e65d4d08b8def23`；范围只含 LongBench 手动 greedy
入口的 0-token 边界，不改 selector、kernel、预算、chunk2048 或非空输入语义。

验收：material 为空且 question 非空时，以 question 首 token 建立 cache 后继续逐
token question；两段都空时在模型调用前明确拒绝；空 question 的 BOS 归一化不索引
`[0,0]`；原非空 material 路径调用顺序保持不变。普通与 `-O` 相邻身份回归一致。

## Progress

相邻的短 prompt 分割修复会把不超过 100 字符的无锚点 prompt 全部放入 question。
Qwen3 类无 BOS tokenizer 可让 material 编码为 0 token，但旧 `_manual_generate` 仍先
调用 `model(input_ids=material_ids)`；修前生产函数在拒绝空输入的模型替身上确定得到
`RuntimeError: model received an empty token sequence`。完全空 prompt 还会在 question
BOS 检查中无条件读取 `[0,0]`。

已新增 `_drop_leading_bos` 的空宽度门禁；`_manual_generate` 在 material 为空时用
question 首 token 以 `past_key_values=None` 建立 cache，再处理余下 token。两段都空
则在任何模型调用前抛出 `ValueError`。非空路径保持 material→question 的原顺序。
新增测试用拒绝 0-token 的模型替身覆盖空/短/非空交接、BOS 与真实入口接线。

## Evidence / Boundary

CPython 3.12.14，Linux 6.18.55 x86_64；未安装 Torch，`nvidia-smi` 不存在。
普通与 `-O` 各 28 项通过，含本问题 6 项、prompt split 5 项以及 LBv1 源/模板/token
身份、同数据集一致性和评分数据集绑定；两种模式 `compileall` 与 `git diff --check`
通过。`tests.test_longbench_request_capture` 因缺 Torch 未收集，旧 LBv2 脚本的评分侧
因缺 `jieba` 未完成；这些依赖缺口不计通过。

另一次扩大到 31 个“源码未直接 import torch”的顶层测试模块的探索性运行并不构成
有效全库回归：171 项中 12 失败、11 错误、1 跳过。错误包含间接 Torch 依赖、旧测试
的顶层模块导入约定，以及最新论文资产 manifest 与旧 E140 测试不一致；原始负结果
已保留在本轮终端记录，未把其中已运行的项目拼成通过数，也未改论文资产或旧测试。

当前没有真实 tokenizer/model/GPU，故没有声称真实 Qwen3 输出、质量、TTFT、显存、
kernel 数或性能变化。生产复验需覆盖空 material + 多 token question 的 KV cache
逐位对照、完全空输入的 CLI 错误传播，以及可选 `paper-capture=timing` 仍要求非空
material 的既有范围。详见
[中文简报](../../advice/2026-10-21_longbench_empty_handoff_by_gpt.md)。
