# TL-E151C-INPUT-20261018：拒绝被误读为配额的 JSON

基线 `cb0c8f430b9e2595caef48590b01690a99d416b3`（2026-10-18）。相对上轮
4496349，新提交仅改论文；本轮继续处理上轮明确遗留的配置输入契约。

## 触发、根因和修复

`sparse_attn/indexer/tli_indexer.py:123–126` 原先对 JSON 根直接迭代并执行
`int(round(float(x)))`：`"32"` 被当作 `[3,2]`，`{"32":999}` 被当作 `[32]`；
空表、布尔值、数字字符串及负数也被接受。启用 E151c prefill 后错误值成为
far 预算输入；表内容与用户指定的对象值可能完全无关。

现于构造期（123–144 行）要求非空 JSON 数组，元素为有限且非负的数值，
明确排除 bool；float 转换溢出也统一为 ValueError。合法 oracle 均值保留
原 float 转换及最近偶数取整（32.5→32，33.5→34），仍允许零和超过 K2 的
配额，后者由原消费端裁剪。文件访问/JSON 语法错误沿用原异常。

只修改配置加载，没有增加逐 query 操作、host/device 同步或 kernel；没有改变
chunk2048、候选粒度、topk ties、保护区、GQA 或实际预算消费公式。
表长与层数、layer_idx 延后注入、γ-off/非 e64 是否支持配额仍待另行明确；
本修复不保证任意长度表或任意配置都可执行，也不宣称消除了全部配额问题。

## 实际验证

结果与可重跑基线加载器见 [结果目录](../results/e151c-input-20261018/)。
最终版本输入专项 3 个方法（17 个非法输入子例及合法取整控制），基线：
`Ran 3 tests; FAILED (failures=8, errors=8)`。其中 errors 是生产代码抛出
TypeError/OverflowError，而测试要求配置 ValueError；不是测试设施错误。
NaN 原本已抛 ValueError，其余 16 个非法输入未满足要求。不能将这些计为
16 个独立生产 bug。

修后包含既有真实 selector 路由的 E151c 13 项，普通及 -O 均通过。
相邻 E139 13 项、paper runtime 5 项普通/-O 均通过，每种模式合计31项。
原始日志随附；测试运行时间不作性能比较。

```sh
python agent_doc/results/e151c-input-20261018/baseline_runner.py # 预期退出1
python -m unittest discover -s tests -p 'test_e151c*.py' -v
python -O -m unittest discover -s tests -p 'test_e151c*.py' -v
python -m unittest discover -s tests -p 'test_e139*.py' -v
python -m unittest discover -s tests -p test_paper_runtime.py -v
```

后两条也以 -O 重跑；所有运行设置 OMP_NUM_THREADS=1、MKL_NUM_THREADS=1。
环境 Linux 6.18.55 x86_64、9 个可见 CPU、Python 3.12.14、Torch 2.8.0+cpu、
einops 0.8.1、Transformers 4.56.2，CUDA=False；无 GPU 架构/驱动/编译器数据。
本输入验证不生成张量；输入集合和测试文件 SHA256 见 SHA256SUMS。
质量、吞吐、TTFT、峰值显存及 kernel 数未测，不作提速主张。

## 历史影响与后续

历史运行若传入字符串/对象根、bool/字符串/负值，应回查原 quota 文件和实际
far 选集，不能将错误预算结果继续当指定 quota 实验；正常非空数值数组不因
本加载修复自动失效。未修改历史测量。此前配额路由修复的 GPU/真实模型重验、
候选内 L2 与 fused indexed attention 的成本优化仍未完成，保持后续任务。
