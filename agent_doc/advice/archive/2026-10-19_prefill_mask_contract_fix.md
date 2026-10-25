# Prefill token 选择 mask 消费契约修复

问题 ID：TL-PREFILL-MASK-CONTRACT-20261019。基线 `3b0cfea1dac5598b44bfa28818b020c5a7e61227`；独立 clone 后 fresh fetch 确认。实验分支 e138-ablation 为 `2ed6a1feb78a7b5bcf3cac922b4510c3968714a5`，仅核验，没有合并。

## 触发、根因与有界修复

真实入口 `sparse_attn/patches/qwen3_attn_patch.py:168–186` 在 `TLI_SPARSE_PREFILL=1`、B=1、无 attention_mask 时进入 sparse_prefill_attn。reference 和 candidate 逐组消费 prepare_mask 的返回值时直接取 mask[0,0]，忽略返回的 block_size。前者可把仅一个 True 槽广播成全前缀；两者都可能静默丢掉额外 batch/query 维度，把块级选择误当 token 级选择。

最小真实 CPU 反例：Q/K=0，V[t]=t，L=7，H=2/Hkv=1，chunk=4，dense_below=0，selector 返回 `[1,1,1,1]` True、block_size=1。修前 reference 返回每行全前缀均值而不拒绝不完整 token mask；candidate 的同类短 mask 可能晚期报广播错误。新增门禁将这两类行为统一为 attention 前的明确 ValueError。

`sparse_attn/ops/prefill_contract.py:7–25` 只读 host 元数据：Tensor/strided/bool/同设备、形状 `[1,1,Hkv或H,T_pad>=prefix]`，block_size 为非 bool 的整数 1。接入 `reference_prefill.py:206–207` 和 `eager_prefill.py:463–464`。合法输入仍返回原 slice view，允许 padding、offset、非连续/零步长布局，不增加 tensor 归约、.item、复制或 kernel。普通 TLI 的 `get_block_size()` 明确返回 1（tli_indexer.py:2509）。

兼容边界：Quest 返回块宽（quest_indexer.py:81），不属于这里声明的 token-mask 接口。此次显式拒绝其 opt-in sparse-prefill 误接，不声称新增 Quest prefill 支持，也不修改默认 native/dense prefill。自动展开块 mask 会新增算法适配、预算和保护带语义，本次不采用。旧 ProtectionIndexer 测试夹具生成的是 token mask 却标 page_size，现只修正返回 metadata 为 1，原数值断言不变。

不变项：outer chunk 默认2048、组末 query 共享候选、预算、sink/SWA、GQA、归一化/topk ties、_inc、selector/kernel、batch_select 内部输出以及 overlap 均未修改。helper 在 selector 调用后执行，不承诺回滚 selector 内部状态；oracle 标志已有 finally 清理保持有效。

## 实际测试与环境

先加入测试再修实现。修前原始短日志：

```text
Ran 3 tests in 1.275s
FAILED (failures=1, errors=51)
AssertionError: ValueError not raised
```

其中一个失败是上述实际 singleton 输出未拒绝；51 个错误是17种非法返回×3路由未被守卫，继续进入 mock attention 后产生 TypeError，并非51个独立数值错误。合法24子例当时已通过。修前完整日志保留于执行检查点，不将64KB traceback入库；SHA256 `c9a93820e4d8dbe8e467ae88238ec1fa8d122a88702f025b5310556e2c02bd03`。随后增加 Tensor block_size 的拒绝子例（无修前记录），最终为54非法子例+24合法子例+1实际 singleton 反例。

命令（均 OMP_NUM_THREADS=1，各120s超时界限）：

```sh
python -m unittest sparse_attn.ops.tests.test_prefill_mask_contract -v
python -m unittest discover -s sparse_attn/ops/tests -v
python -m unittest tests.test_e151c_prefill_routing tests.test_e139_graph_contract tests.test_e139_shared_gqa_indices -v
```

专项3方法通过；ops普通和 `python -O` 各84方法通过。相邻测试覆盖真实 CPU TLI oracle 路由和 E139 shared-GQA/graph 契约；结果及独立复核见 `agent_doc/results/prefill-mask-contract-20261019/`。测试耗时不是性能对照。新测试独立绝对位置均值参考覆盖空/短/非整块、padding裁剪、stride、Hkv/H选择、sink/SWA和oracle清理；随机logit复核另见独立记录。

环境：Linux6.18.55 x86_64、Xeon Platinum8573C（容器可见9CPU）、Python3.12.14/Clang22.1.3、torch2.8.0+cpu。CUDA不可用，未占用远端GPU；驱动/设备架构/kernel编译配置不适用。依据实现优化技能的红绿与证据门禁，仅合入CPU可验证的协议守卫。

## 未验证与历史实验影响

这是恢复错误输入处理契约的正确性修复，无真实模型任务质量、速度、峰值显存、kernel数或TTFT收益声明。没有生产GPU、fused attention或双stream验证，未用CPU结果替代它们。内部 batched_group_sels 输出不在本次门禁范围，固定前缀替换后缀/增量缓存生命周期另行验证。

历史样本仅在返回短 mask、错误维度/类型或 block_size!=1 时需要排查并按原预算重跑；合法TLI结果不因此自动失效。没有修改历史实验数据。候选内L2、全前缀打包、重复搬运与流水重叠优化仍需已授权GPU上的端到端测量，本次不合入未验kernel。
