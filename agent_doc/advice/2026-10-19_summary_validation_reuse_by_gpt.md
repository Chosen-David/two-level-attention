# TL-SUMMARY-VALIDATION-20261019：消除结果汇总的重复全协议扫描

## 范围与定位

- 基线：`68c9eb350194943a657ba09ce3548e6ea9aa1132`；独立检出、非阻塞合作锁，无他人未提交改动。
- main 自上次 `f56437f` 后只有手稿文档变动；本轮重新 fetch 的实验分支 `e138-ablation` 为 `2ed6a1feb78a7b5bcf3cac922b4510c3968714a5`，未合并、未借用其实验结果。
- 实际入口：`exp/paper/workflow.py:97–100` 的 timing 命令调用 `summarize`；原 `exp/paper/contract.py:162–171` 在入口检查协议后，每条记录又经 `validate_timing:110–116` 扫描所有样本/支持单元并重建样本表。
- N 个样本、R 条结果时，重复协议工作为 O(R×N)，不是 attention kernel 瓶颈。本轮只优化 CPU 结果汇总，不改变 selector、预算、近似算法、chunk 或实验数据。

## 改动与边界

新 `contract.py:110–124,170–177` 将逐条校验提为私有函数：公开 `validate_timing` 每次仍完整校验协议；`summarize` 每次先校验一次协议、构建一次样本表，然后对每条记录执行原有全部门禁。不存在跨调用缓存，不复用记录校验结果。身份、路由、support、失败/缺失/重复分母、配对及输出不变。

适用真实入口读取 JSON 后持有普通 list/dict 的同步调用；不支持调用途中由其他线程或自定义迭代器修改协议，亦不新增这种支持。

## 修前与回归

修前执行新增专项 5 项：4 通过，1 失败；两条合法结果产生三次协议校验：

```text
test_protocol_is_validated_once_per_summary_not_once_per_row ... FAIL
AssertionError: 3 != 1
Ran 5 tests ... FAILED (failures=1)
```

修后增加真实 CLI 集成，普通与 `-O` 分别执行：

```text
python [-O] -m unittest discover -s exp/paper/tests -v
Ran 18 tests ... OK
python [-O] -m unittest tests.test_e152_request_timing tests.test_e152_result_store -q
Ran 16 tests ... OK
```

覆盖后续行非法身份/hash/路由/预算支持、重复/缺失/OOM/空输入、下一次调用修改协议、直接行验证入口、CLI 合法输出与非法输入不发布。现有 T2、controlled-support 等门禁回归保留。每模式共 34 项，不是 GPU 验证。

另直接从上述基线 commit 加载旧模块，逐字段替换后续行的14类非法/边界值并逐项删除协议字段，与当前模块做352场景差分，返回内容或异常类型/信息全部一致：`BASELINE_DIFFERENTIAL 352 cases; identical results or exception type/message`。这是同执行者补充核验，不冒充独立审查。

## 同机配对性能证据

命令：`python exp/bench/bench_paper_summary.py --baseline 68c9eb350194943a657ba09ce3548e6ea9aa1132 --samples 1 32 128 512 --pairs 5`。
CPU：AMD EPYC 9V74，KVM x86_64，9 个可见逻辑 CPU；Linux 6.18.55；Python 3.12.14 / Clang 22.1.3。单进程串行调用，无显式线程池或 SIMD 改造。
无 PyTorch、可见 nvidia-smi、CUDA/Triton/compiler/driver 测量；也没有访问历史训练设备。

每个规模两个实现各 warmup 一次，然后五轮交替 AB/BA，`perf_counter_ns` 测完整 `summarize`，包括协议、逐行验证及配对汇总；不含输入生成、导入、hash、JSON I/O。不是完整 CLI wall，也不是请求/TTFT。输入为两臂、两次 repetition 的合成合法回执，每单元64个support ID，协议语义和输入完全相同；输出逐次等于基线，运行后输入 hash 未变。

| 样本 / 结果行 | 基线中位 ms | 修后中位 ms | 比值 |
| --- | ---: | ---: | ---: |
| 1 / 4 | 0.247209 | 0.101402 | 2.44× |
| 32 / 128 | 19.802603 | 2.241939 | 8.83× |
| 128 / 512 | 259.957610 | 9.592591 | 27.10× |
| 512 / 2048 | 4392.279229 | 47.412189 | 92.64× |

完整短原始样本、输入/输出/实现 SHA256 在 `agent_doc/results/summary-validation-20261019/paired.json`；512样本输入 `71a6a91f3a224b046fbcda38c5defcd8fb1b44cc69a5bcb3776365659ab56911`。
计时前的同源码 A/A 检查（32/128样本各三对）中位比值分别0.970/0.922，无假阳性提速主张。最终收益远高于这次观察到的扰动，但不据此声明统计置信区间或跨机器保证。峰值内存未测；样本表多保留 O(N) 引用，不宣称内存改善。

## 历史影响及后续

历史 timing/质量数据不重写，汇总内容保持相同；本改动不会恢复任何历史 GPU/模型/科学有效性。本轮没有新增模型质量结果、kernel 数、host-device 同步或峰值显存数据。
本地人类 GUIDE hash 未变，指南保持只读；其引用的外部 sglang 两个指南路径在本执行环境不存在，不访问其他仓库。本项在独立库结果汇总授权范围内实施。
仍待：可用 PyTorch/GPU 上的批量选择、候选内L2、紧凑索引、fused attention及overlap验证；不把这次CPU汇总收益冒充这些待办完成。
