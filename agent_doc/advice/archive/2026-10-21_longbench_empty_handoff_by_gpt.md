# LongBench 空 material 状态交接修复

问题 ID：`TL-LB-EMPTY-HANDOFF-20261021`。代码基线：
`58c0d437c86c3cf568eadf076e65d4d08b8def23`。

## 触发、根因与修前失败

LongBench 手动 greedy 路径先用 material 建 KV cache，再逐 token 消费 question。短
prompt 分割已正确把不超过 100 字符的无锚点文本完整交给 question；对 Qwen3 这类
不自动加 BOS 的 tokenizer，material 的实际宽度可为 0。旧实现仍无条件执行首次
material forward，所以会把形状 `[1,0]` 的输入送进模型，而不是从 question 建立
cache。完全空 prompt 的 `q_input` 还会在 BOS 判断处直接读取 `[0,0]`。

使用从基线源码 AST 加载的真实 `_manual_generate`，输入 `material=[]`、
`question=[11,12]`，并用仅拒绝零宽输入的最小模型替身复现：

```text
{"eos_token_id":null,"material_ids":[],"max_gen":1,"question_ids":[11,12]}
input SHA-256: f2d0daebfd4b3e21f13f401e5c36168c41874a154313a3bad1c002153bfac7b0
```

```text
RuntimeError: model received an empty token sequence
```

新增回归在生产修改前先失败：

```text
AssertionError: pred.py helper mismatch:
expected={'_manual_generate', '_drop_leading_bos'}, found={'_manual_generate'}
FAILED (errors=1)
```

## 修复与语义边界

- `benchmark/LongBench/pred.py:205-213` 的 `_drop_leading_bos` 先检查 token 宽度，再读取首 token；
  非空、非 BOS 输入原样返回，真实 BOS 仍按旧规则删除。
- `benchmark/LongBench/pred.py:216-259` 的 `_manual_generate` 非空 material 路径未改。仅当 material 宽度为 0 且 question
  非空时，用 question 首 token、`past_key_values=None` 调用 question 阶段建立 cache，
  然后继续原逐 token question 路径。
- `benchmark/LongBench/pred.py:361-366` 的入口改用同一 BOS helper。两段均为空时，
  在任何模型 forward 前以明确 `ValueError` 拒绝，避免依赖后端产生
  不一致的空张量错误。

这不改变 near/far 选择、sink/SWA、预算、D′、GQA、chunk2048 或 kernel；也没有
通过减少工作量制造性能主张。

## 实际验证

环境：CPython 3.12.14；Linux 6.18.55 x86_64；Torch 未安装；无 `nvidia-smi`。

```text
python -m unittest tests.test_longbench_empty_handoff \
  tests.test_longbench_prompt_split tests.test_lbv1_input_identity_v2 \
  tests.test_lbv1_input_uniformity tests.test_lbv1_dataset_binding -v
Ran 28 tests in 0.066s — OK

python -O -m unittest tests.test_longbench_empty_handoff \
  tests.test_longbench_prompt_split tests.test_lbv1_input_identity_v2 \
  tests.test_lbv1_input_uniformity tests.test_lbv1_dataset_binding -v
Ran 28 tests in 0.047s — OK

python [-O] -m compileall -q benchmark/LongBench/pred.py \
  tests/test_longbench_empty_handoff.py
git diff --check

candidate pred.py SHA-256:
28becef9ab7ac96b2a1c9a2fc66f16f39e488b9d7463650f8d8c0c4bc7091ab9
candidate test SHA-256:
318e0a39d76aaf84df24fc288fe2fada5b97e6313e97e0b45254667ee12230cd
```

额外相邻测试没有被包装成通过：`tests.test_longbench_request_capture` 因当前环境缺
Torch 在导入期失败；旧 `benchmark/LongBench/test_lbv2_parser.py` 的评分侧因缺
`jieba` 失败（其无依赖的 17 个 parser 向量先通过）。

探索性扩大到 31 个“源码未直接 import torch”的顶层模块时，间接依赖和既有资产
漂移仍使 171 项出现 12 失败、11 错误、1 跳过；其中包括间接 Torch 依赖、旧测试
顶层模块路径约定，以及最新论文资产 manifest 与旧 E140 测试不一致。该结果不是
本补丁回归口径，已作为负结果保留，没有选择性汇报其中通过项或顺手改动论文资产。

## 未验证范围

当前没有真实 Qwen3 tokenizer/model、Torch 或 GPU，未验证生产 KV cache 逐位输出、
真实模型质量、TTFT、设备时间、峰值显存、kernel 数或 host 同步变化；因此不报告
性能收益。生产重测应对比：非空 material 原路径、空 material + 单/多 token
question、完全空输入错误，以及 sidecar 中实际 token hash 与模型收到的 token 序列。
可选 `paper-capture=timing` 的现有采集器仍明确要求非空 material；本次未在无 Torch
环境扩张其遥测合同，也不把默认生成修复等同于该可选模式已支持空 material。
