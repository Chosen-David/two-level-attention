# LongBench 短 prompt 问题阶段负索引修复

问题ID：`TL-LB-SHORT-PROMPT-SPLIT-20261021`。代码基线：
`62522a685cb298c04093510d499c8da6182b7be5`；基线 `pred.py` SHA-256：
`7dc34ee3ce365ef2578739cba680aab54fdb852b60c7438423b4c262bafd87ae`。

## 触发、根因与影响

生产入口 `benchmark/LongBench/pred.py` 在模板和聊天格式处理后，把 prompt 拆成
material prefix 与 question suffix。旧实现对无匹配锚点任务令 `q_pos=-1`，再取
`max(len(prompt)-100,q_pos)`；当非空 prompt 少于100字符时结果仍为 `-1`，而
`if q_pos != None` 永远为真，于是 Python 负索引将前 `N-1` 字符送入 material、
只将最后1字符送入 question。两段被分别 tokenizer 编码，除阶段路由错误外，还
可能因子词边界改变最终 token 序列。长度0不崩溃，但该条件同样不是有效哨兵。

固定复现输入为 `['', 'x', 'short prompt', 'x'*99, 'x'*100, 'x'*101]` 的紧凑
JSON，SHA-256 为
`a3c1601e11b9db7d145427cb39ac48154c613992aa20879040949cfb2f8237cc`。修前短日志：

```text
len=1   q_pos=-1 prefix_len=0  question_len=1
len=12  q_pos=-1 prefix_len=11 question_len=1
len=100 q_pos=0  prefix_len=0  question_len=100
len=101 q_pos=1  prefix_len=1  question_len=100
```

新增 `tests/test_longbench_prompt_split.py` 后、生产修改前实际运行失败：

```text
AssertionError: pred.py must define one _split_prompt_question
FAILED (errors=1)
```

## 修复

- `benchmark/LongBench/pred.py:163-184`：新增无依赖纯函数
  `_split_prompt_question`，集中保存各任务锚点并以
  `max(0, len(prompt)-100, anchor_pos)` 钳制边界。
- `benchmark/LongBench/pred.py:309-312`：真实入口直接消费该函数，删除永真
  `q_pos != None` 分支。长输入、后置锚点及100字符上限保持原契约。
- `tests/test_longbench_prompt_split.py`：通过 AST 只加载纯函数，当前无 Torch 时仍
  对空/短输入、长输入、锚点命中/缺失和 `prefix+question==prompt` 做生产源码回归。

## 实际验证

环境：CPython 3.12.14；Linux 6.18.55 x86_64；`torch` 未安装；
`nvidia-smi` 不存在。命令与短日志：

```text
python -m unittest tests.test_longbench_prompt_split \
  tests.test_lbv1_input_identity_v2 tests.test_lbv1_input_uniformity \
  tests.test_lbv1_dataset_binding -v
Ran 22 tests in 0.044s — OK

python -O -m unittest tests.test_longbench_prompt_split \
  tests.test_lbv1_input_identity_v2 tests.test_lbv1_input_uniformity \
  tests.test_lbv1_dataset_binding -v
Ran 22 tests in 0.042s — OK

python [-O] -m compileall -q benchmark/LongBench/pred.py \
  tests/test_longbench_prompt_split.py
git diff --check
```

新增测试文件 SHA-256：
`432345bdb6a5b639bb15447e0776108165fe682acf96a17023aaeaac53f750e3`。

## 质量、性能、内存与历史边界

这是入口语义正确性修复，没有算法近似、预算放宽或外层 chunk 变化。当前没有真实
tokenizer/model/GPU，未测模型输出质量、TTFT、设备时间、峰值显存、kernel数或
host同步，因此不报告性能收益。正式 LongBench 模板通常超过100字符；历史主表是否
命中未知，不能据此改写既有结果。需要生产资源时重测自定义短样本、缺锚点模板及
prefill→question→decode交接，并核对新 sidecar 的实际 token hash。
