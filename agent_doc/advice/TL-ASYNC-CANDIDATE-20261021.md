# TL-ASYNC-CANDIDATE-20261021：候选评分与延迟掩码不兼容

基线 main `e6faba35342dbc21414e0212b27b68d209726b96`，2026-10-21。
本次重新 fetch 的实验分支 e138-ablation 为 `2ed6a1feb78a7b5bcf3cac922b4510c3968714a5`；
只用于对照，没有合并，也不把其测量归入 main。

## 问题、修复与历史影响

`sparse_attn/indexer/tli_indexer.py:1460–1498` 的 `_gather_fine_cand` 根据本步 L1
构造候选；`compute_mask` 先将候选散回全宽、其它位置填负无穷（2354–2393），
再在 async 模式交换 `prev_mask`（2676–2686）。若旧候选中的 token 不在当前
候选、bypass、sink 或 SWA 中，它的真实 fine 分数已丢失，不能由旧掩码恢复。
这会改变 softmax 分母、GQA 聚合和后续选集，不只是允许的浮点舍入误差。

触发需要 `tia_enable_async_topk=True`、候选化门控通过且相邻步有效候选变化。
默认关闭 async 的路径不因此判为错误。独立标准库反例中旧独有 token 的非零
概率变为0，最大概率 token 也变化；这是机制证明，不是实模质量或GPU结果。

最小修复在共享 helper 入口拒绝 async 候选化，返回既有 None 三元组。
两个调用者分别保留父类已计算的全宽分数（1670–1692）或进入既有全宽评分
分支（1899–1907）。不改预算、归一化、ties、保护区、外层chunk或kernel。
首步也统一回退，避免引入依赖内部缓存状态的额外执行分支。

历史上 E156 候选路径与 async 同开且候选变化的实验应重新核对/重跑；不能
由该缺陷推定所有历史实验失效。本修复恢复已有全宽 async 路由，并不认证
该路由所有既存 D′/缓存组合。手工注入候选 score_dict 不在此次公共生产路径验收内。

## 修前失败与验证范围

新增 `tests/test_tli_async_candidate_gate.py` 直接编译执行生产 helper 方法，
用选择器探针在张量操作前停止，核对门控和 prev_mask 不被修改。
修前3方法含2失败（async首步/后续步）；修后门控及相邻回归：

```text
python [-O] -m unittest tests.test_tli_async_candidate_gate \
  tests.test_tli_runtime_shape_contract tests.test_tli_debug_contract \
  tests.test_explicit_near_page_budget -v
Ran 16 tests ... OK     # 普通和 -O 各一次
```

覆盖 async 首步/后续步、同步候选入口、显式/推断长度及投影、MoBA、per-q、
sigma、cluster 等既有回退。独立上下文重新核验调用链、门控及机制反例；
原始日志和独审保留在私有 `.agent-runs/async-candidate-2151/`。
Linux6.18.55 x86_64、glibc2.39、Python3.12.14；无torch及可确认GPU，CUDA、
驱动、Triton/编译器配置不适用。实际张量、模型质量、速度和显存未测，不报告提速。
生产SHA256 `0346987f698934d9537f9ca9caeb750feb4684c290645969df718944d9670f7f`；
测试/输入fixture SHA256 `aa5e61c9e247527de6da0cf00b24ba6dcd0c8a3e8a29a743e57ea5ddeb9ab36c`。

## 同次性能审查：未修、未测速

- 当前 full-subspace 父类捷径先计算全宽 fine，再调用候选 helper 重算；主分支也在
  候选评分前分配全宽 score_fine（1855–1857），候选随后仍散回全宽做softmax。
- helper 的 L1 选择和 compute_mask 内的粗筛仍分别执行；候选构造含 nonzero、
  动态打包和 `.max().item()`（1517、1558），需要完整GPU计时，不能只报点积节省。
- prefill eager candidate 仍可能跨组并集后屏蔽、全前缀argsort打包
  （`sparse_attn/ops/eager_prefill.py:144–205`）；GQA预算问题沿用
  `TL-PREFILL-SCORE-BUDGET-20261019` 未决任务，不重复宣称已修复。

仓库 `exp/trace/results/e152v2_e2e_latency_verdict.json` 是2026-10-16历史汇总：
32K/64K/128K 的 dense TPOT为26.0/39.7/67.3ms，TLI为87.9/97.6/119.9ms。
这是历史decode配置的负结果，不是当前候选实现的新测量；原始重复样本及派发
需另核验，不能凭中位数差证明“只剩launch瓶颈”。E127的5.47倍是选择器旧新
阶段比较，不能当相对dense请求加速。本次保留上述负结果，不修改实验数据。
