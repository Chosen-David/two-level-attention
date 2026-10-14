# Prefill 非法输入 fail-closed 修复

作者 GPT；2026-10-14；问题 ID：**TL-PREFILL-INPUT-GUARD-155**，P2，confirmed → fixed / CPU rechecked。

## 范围与根因

基线 main `c60af1e59caab2309a963f2633b5e819b092bedb`；对照实验分支
`e138-ablation` 仍为 `58745c46a6c3ee0847599f0429fb5dfb2a2526cb`，未合并该分支。
本轮先核验新增 paired-quality/scorer 提取的改动，再继续未完成的 prefill
入口契约检查。修改仅限本库，未动其他人的 worktree、任务详情或人类指南。
任务索引/指南相较上轮无改动；引用的远程人工 TASK 文件本机仍不可读。

`sparse_attn/patches/qwen3_attn_patch.py:81–89` 可通过 opt-in 路由调用
`sparse_prefill_attn`；chunk 可来自 `TLI_SPARSE_PREFILL_CHUNK` 环境变量。
基线 `sparse_attn/ops/eager_prefill.py:128–139,151–152` 有两类同源入口缺口：

- `chunk_size=-2048` 使 `range(0,L,chunk_size)` 对正 L 为空。函数分配
  `empty_like(q)` 后直接返回，输出未写入，调用不报错。
- B=1、Q/K 等长使用 `assert`，在 `python -O` 下消失。内部只计算 batch0，
  若直接调用 B=2，第二个 batch 的输出保持未写入。V 的 batch/形状此前也没有
  对应入口约束。现有 Qwen3 wrapper 本身有 B=1 门禁，**不声称正常 wrapper
  会产生 B=2**；但负 chunk 的环境配置能到达该 wrapper 下游。

本轮将已声明的形状限制改为显式 `ValueError`：rank4、全部 B=1、K/V同形、
Q/K等长、正且一致的 head_dim、正 head 数及合法GQA比例；chunk 必须是
正整数（允许实现整数索引协议的类型，不隐式截断小数，不接受 bool）。
所有检查在输出分配和 indexer 调用前完成。空序列保留空输出；默认外层2048、
选择预算、sink/SWA、近远筛选与attention公式均不变。

## 最小复现与修前证据

为避免依赖 allocator 当时残留的随机值，用确定性填充值替换**分配函数**，
不替换被测入口或计算逻辑。输入 Q `[1,4,2,8]`、K/V `[1,4,1,8]` 全零，
indexer 使用只含默认配置的对象（负 chunk 不会调用 selector）：

```python
with patch('sparse_attn.ops.eager_prefill.torch.empty_like',
           side_effect=lambda t: torch.full_like(t, 777)):
    out = sparse_prefill_attn(indexer, q, k, k, chunk_size=-2048)
    print(bool((out == 777).all()))  # baseline: True，无异常
```

同样用 `python -O`、B=2、正常2048 chunk：基线 `out[1]` 全为777；batch0已计算。
这证明写入路径被跳过，不声称777是实际未初始化内存中的值。修后两种情况均
在调用 `empty_like` 前抛 `ValueError`。

Q、K、V连续逻辑float32字节串拼接SHA256：
`076a27c79e5ace2a3d47f9dd2e83e4ff6ea8872b3c2218f66c92b89b55f36560`。

## 验证与成本边界

新增 `sparse_attn/ops/tests/test_prefill_input_contract.py`：6项测试方法，覆盖
负数/零/非整数/env chunk、Q/K/V非法batch、rank/长度/head/GQA不匹配、
空序列；拒绝测试同时断言未分配输出、未调用selector。

```text
修前新增套件：Ran 6 tests; FAILED (failures=10, errors=9)
修后 ops/tests：Ran 9 tests; OK
python -O ops/tests：Ran 9 tests; OK
原有 test_sparse_prefill_impl.py：7/7 PASS
  T1/T3/T5/T6 max abs error = 0
  T2/T4 max abs error = 5.22e-08
git diff --check：通过
```

命令（仓库根，原有torch/einops环境）：

```bash
python -m unittest discover -s sparse_attn/ops/tests -v
python -O -m unittest discover -s sparse_attn/ops/tests -v
python test_sparse_prefill_impl.py
```

本次使用上轮隔离CPU依赖：Python3.12.14、torch2.8.0+cpu、einops0.8.2，
Linux6.18.44/x86_64；单线程 CPU 测试。原7项从仓库已有测试提取到临时目录
执行，未改内容，消费本轮worktree的实现。无CUDA设备，未测GPU/BF16、
真实模型生成、TTFT、峰值显存或GPU内核数；不声称吞吐提升或零开销。
新增校验只读取张量metadata与Python配置，不引入数据相关`.item()`或GPU归约。

## 历史数据与后续检查点

正常B=1、正chunk的历史实验不因本缺陷自动作废。若曾使用非正chunk或在
`-O`下绕过wrapper直接调用不支持的形状，应视其输出为无效并重跑；目前没有
对应运行清单，不能断言某个已发布分数受影响。未修改任何实验数据。

153的sink修复保留且回归通过。逐query routing、候选内L2、全前缀QK/PV
浪费、紧凑attention与跨tile overlap仍未完成；无GPU条件不合入未验内核。
本轮是输入正确性修复，不将其包装成论文算法创新或实测性能优化。
