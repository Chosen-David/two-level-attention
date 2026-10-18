# TL-BATCH-SINK-ZERO-20261019：零 sink 不得进入使用 token 0 哨兵的批量选择

## 定位、触发与根因

基线 `ffaaf9fe51c86608a45b4a809890a1e065dd55a4`；本轮重新 fetch，实际实验分支 `e138-ablation` 为 `2ed6a1feb78a7b5bcf3cac922b4510c3968714a5`，其门禁也只检查对齐。实验分支仅作源码对照，未合并，未把其历史性能当 main 实测。

真实入口 `sparse_attn/ops/eager_prefill.py:472-498` 在 grouped prefill 且 `TLI_PREFILL_BATCH_SELECT=1` 时调用 `batched_group_sels`。`batch_select.py:239-282` 的 `_gate_ok` 原来允许 `sink_tokens=0`（0也是块对齐值）。但 `:176-213` 的 `_scatter_candidate_scores` 把所有无效 L1 补位槽改写至 token 0，并写入 `-inf`。只有 token 0 在保护区时，这种重复写入才不覆盖合法候选。

当 sink=0、合法 L1 候选包含 token 0、同时存在补位槽时，真实分数与 `-inf` 会发生冲突。CPU 原始选择胶水反例：K1=4、K2=65、S=256、SWA=64、α=β=.25、γ=0、Hq=2/Hkv=1、D=128；分数在 token 0 唯一最高，正确选集应为 `{0} ∪ [192,256)`，旧胶水实际选 `{1} ∪ [192,256)`，数量仍65，单纯预算计数无法发现错误。这里替换了 CUDA 打分并显式绕过 CUDA 门禁，只证明 CPU 选择逻辑问题，不是 GPU 复现。

[PyTorch 2.8 scatter_ 官方契约](https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.scatter_.html)（核查于2026-10-18 UTC）明确指出非唯一索引不能保证选中哪个源值，因此不能把 CPU 覆盖顺序当 CUDA 的确定顺序。

## 修复与取舍

`batch_select.py:270-273` 增加 `sink_tokens >= bs`，保留原有块对齐条件。合法零 sink 配置在任何 tensor 分配、索引缓存更新和 graph capture 前返回 None，由已有逐组路径处理；`require_graph=True` 明确抛 unsupported selector 错误，不让验收脚本把回退冒充 Graph 成功。

候选取舍：入口守卫（采用，O(1) host 比较，无新增张量/同步）；更换散写哨兵或新增额外列（暂缓，涉及 GPU gather/scatter、softmax归约宽度与 Graph，需要真实设备验证）。不改 kernel、选择预算、外层chunk2048、组语义、topk ties 或浮点归约顺序。sink4等非对齐配置本来已回退；sink64/128等安全对齐配置仍放行。不能把回退称为提速。

## 实际验证与环境

Python3.12.14，PyTorch2.8.0+cpu，Linux6.18.55 x86_64，容器暴露 Intel Xeon Platinum8573C、9逻辑CPU；CUDA设备0，驱动/SM/CUDA编译器/Triton执行配置均未验证。测试 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`，不涉及GPU计时/JIT。所选实现优化技能用于保留修前失败、按入口路径检验回退并限制硬件结论；未安装新工具或使用额外机器。

- 新增 `sparse_attn/ops/tests/test_batch_sink_guard.py`：修前5项中3失败（零sink错误放行、普通调用到达tensor工作、强制Graph到达tensor工作），修后5项全通过。
- `python -m unittest discover -s sparse_attn/ops/tests -v`：75项通过；`python -O`同样75项。
- `python -m unittest discover -s tests -p 'test_e139_shared_gqa_indices.py' -v`：7项通过；`-O`同样7项。
- `python -m unittest discover -s tests -p 'test_e151c_prefill_routing.py' -v`：10项通过；`-O`同样10项。
- `python test_e143_subblock_sink.py`：8/8场景通过，0跳过；`-O`同样通过。

合计每模式92个unittest与8个脚本场景。新测试覆盖保护sink正例、非对齐反例、零sink两种派发及真实CPU TLI grouped选择/attention与关闭batch逐位对照（3组选集与完整输出）。已有相邻测试覆盖GQA、ties、stride、配额和decode交接、空短输入等。没有运行全仓库测试、GPU kernel/Graph、真实模型质量或端到端性能；没有独立子Agent验收声明。

复现脚本和短原始日志见 `agent_doc/results/tl-batch-sink-zero-20261019/`。脚本显式绕过守卫后仍展示旧胶水冲突是预期行为；修复对象是可达性，不是修改此helper的内部契约。

输入hash：复现q+k原始float32字节 SHA256 `399b197b237326da66c0739ad8853308ea7ba769bce6f821949b9f56c7d4fcfc`；真实CPU回退测试q+k+v（seed190355）为 `cffb9fb3d0d7a4ecd81d2e0c2faa69121355e36e8c698b712c40c5b72df371ab`。参考值由明确的唯一最大值与SWA集合指定，不以被测分数实现充当正确性基线。

## 历史影响、成本及未完成项

质量：修复非法快路由，不能由合成例推断真实模型分数变化。速度/峰值显存/kernel数/host同步：未测；零sink将从批量回退逐组，可能更慢。安全sink路径只增加host标量判断，无速度结论。

需要核对并重跑：历史E139/E140b等 grouped prefill 消融中同时满足sink=0、batch=1且旧门禁其余条件成立的格，先确认真实派发，保留旧JSON并用新结果身份重跑GPU选集、attention输出、质量与总时延。不是所有sink=0实验都受影响：batch关闭、reference路径、未通过其他batch门禁的配置不经过本冲突；sink4主线原本回退，不能据此宣布其数据失效。保护sink对齐锚点不受该零sink缺口影响。

未完成：生产GPU回退及prefill→decode实机交接、受影响历史格清点/重跑、零sink安全批量kernel、候选内L2/紧凑索引/overlap性能优化。用户持续任务不关闭。
