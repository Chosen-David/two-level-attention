# TL-LB-SHORT-PROMPT-SPLIT-20261021

## Plan

来源：用户要求审查真实入口、保留修前失败并覆盖空/短输入。基线
`62522a685cb298c04093510d499c8da6182b7be5`；范围只含 LongBench 手动生成前的
纯字符串路由，不改 selector、kernel、外层 chunk2048、预算或选集语义。

验收：无锚点/锚点缺失的 0–99 字符输入完整进入 question 阶段；长输入仍仅取
末 100 字符，存在且更晚的任务锚点仍优先；普通与 `-O` 相邻输入身份回归一致。

## Progress

修前 `q_pos=max(len(prompt)-100,-1)` 后无条件按该位置切片；非空短输入得到
`prefix=prompt[:-1]`、`question=prompt[-1:]`。实际最小复现中长度 1/12 分别只把
1 个字符送入 question 阶段；新增回归在生产 helper 缺失的修前树上以
`pred.py must define one _split_prompt_question` 失败。

已把规则收口到 `benchmark/LongBench/pred.py::_split_prompt_question`，以
`max(0, len(prompt)-100, anchor_pos)` 消除负索引。入口消费同一 helper；测试覆盖
空、1/12/99/100/117 字符、锚点存在/缺失及拼接守恒。普通与 `-O` 各22项通过，
含 LBv1 v2 源/模板/token身份、同数据集输入单一性与评分数据集绑定相邻回归；
`compileall`、`git diff --check` 通过。

## Boundary / Next

当前节点 CPython 3.12.14、Linux 6.18.55 x86_64；无 `torch`、无
`nvidia-smi`，没有真实 LongBench 源、tokenizer/model。故没有声称实际 token
逐位结果、GPU输出、质量、TTFT、显存、kernel数或性能变化。既有正式 LongBench
模板通常远长于100字符，本修复主要关闭短/退化输入与缺锚点边界；历史正式实验
是否受影响未知，若重测应优先检查自定义短样本和模板/聊天格式异常样本。

完整证据见
[中文简报](../../advice/2026-10-21_longbench_short_prompt_split_by_gpt.md)。
