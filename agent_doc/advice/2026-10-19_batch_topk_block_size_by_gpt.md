# TL-BATCH-TOPK-BLOCK-20261019：候选列补位的64-token块契约

基线 `4f577242bcd88aa30e2eb25107961030fd3c4aa9`。本次重新fetch核验main并在干净独立检出重放此前未发布检查点；相对上轮 a2468b1 的新增文稿/实验结果保留。实验分支 `e138-ablation` 实际 SHA `2ed6a1feb78a7b5bcf3cac922b4510c3968714a5`，重新检查也有该64常量，但仅作源码对照，未合并、未当作main实验。旧工作目录及未提交工作保留。

## 入口与根因

`sparse_attn/ops/eager_prefill.py:472-498` 在 grouped prefill 的批量选择开关启用时调用 `batched_group_sels`。`batch_select.py:458` 原先只按 `TLI_PREFILL_L2_TOPK_CAND=1` 启用候选列 L2；入口允许可配置的 tia_block_size。

但同文件 `_zero_pad_pool`（215-236）以及 nbad_f_l/nbad_n_l（修前479-482）按64计算候选token数、池偏移及补位区间；其他几何/候选展开使用实际 bs。bs≠64时单位不一致。CPU真实选择胶水、替换CUDA打分的最小例：L=1024、Hq/Hkv=2/1、D=128、K1=2、K2=512、sink=SWA=bs、α=.25/β=.5/γ=.5、组末512/1024。

- bs16/32：候选分支内部异常被捕获，返回None；相同输入全宽分支可完成，造成无效工作后回退。
- bs128：候选分支返回的选集与全宽分支不同；并非只影响异常处理。

这是CPU胶水复现，不是生产GPU复现；CUDA topk的tie行为未被CPU证明。原始失败见 `../results/tl-batch-topk-block-20261019/before.log`：1个测试的3个子例，1 failure/2 errors（后两者为返回None后测试迭代失败）。

## 修复、候选取舍

候选列分支增加 `bs == 64`。非64块保留现有全宽L2 topk，仍保留批量L1和候选内L2打分，不是整个selector退出。Graph signature使用有效的use_cand_topk，因此不会把非64配置标记为候选列模式。64块行为不变，正例确认补位helper仍可达。

采用O(1)主机配置判断：无新增张量、同步或缓存。暂缓把所有64改为bs的泛化方案；该方案需要重新验证CUDA topk ties、零概率补位、图捕获及性能，当前无GPU。外层chunk2048、组末query语义、预算、softmax归约及kernel不变。此为契约修复，不宣称性能收益。

## 验证与边界

Python3.12、PyTorch2.8.0+cpu；当前无nvidia-smi，torch.cuda.is_available=False。CPU测试固定OMP_NUM_THREADS=1、OPENBLAS_NUM_THREADS=1。驱动/SM/CUDA编译器/生产kernel配置未验证。沿用实现优化技能的修前失败→最小守卫→相邻回归流程，不安装新依赖或访问未知训练任务。

新测试 `sparse_attn/ops/tests/test_batch_topk_block_size.py`：真实CPU选择胶水，仅替换GPU打分并绕过CUDA可用性门。输入为固定零q/k、递增打分；测试源码与日志哈希可直接复核。非64两组选集与关闭候选列开关逐位相等，64正例仍调用补位。

ops测试普通/-O各77项通过（3.121s/1.968s）；相邻E139普通/-O各13项通过（54.278s/57.656s）。每种模式合计90项测试。相邻日志中的既有CPU allocation proxy属于共享GQA专项，不当成本次新性能收益。未运行全仓测试、GPU、CUDA Graph实际捕获/重放、模型质量、TTFT、显存或性能测量。

独立上下文复核无阻断，限定为低风险主机配置守卫可发布：普通/-O各50个bs/预算/shared-index差分场景（bs16/32/64/128/256，K2覆盖0、保护区边界、512及2048）与5个Graph有效mode/key的CPU host模拟场景通过。独立基线从固定git SHA读取；64开启路径及所有关闭路径与旧实现一致，非64开启路径与旧全宽基线一致。未实际capture/replay，未证明CUDA ties；完整独立范围见`../results/tl-batch-topk-block-20261019/independent/review.md`。

主机实际为Intel Xeon Platinum8573C、x86_64、9个可见CPU、Linux6.18.55/glibc2.39；共享容器，不作性能对照。修前与修后测试输入生成器相同，q/k为fp32零数组，形状如上；q原始字节SHA256=`30e14955ebf1352266dc2ff8067e68104607e750abb9d3b36582b8af909fcb58`，k=`07854d2fef297a06ba81685e660c332de36d5d18d546927d30daad6d7fda1541`。候选源码SHA256=`8e8f46aeef98d0327836c3dca980f25bdb101b8d0f74e0874e149a5db4eef480`；专项测试SHA256=`74095d384b6ae3aa88a531c1a9496e9809727df8491fc711688735766bdd6962`。打分mock与完整几何固定在测试源码中；不是实模输入或GPU性能结果。

采用已有E139建议中的“候选模式限定已验配置、保留全宽基线”；拒绝把历史GPU tie探针外推至任意bs。2026-10-19重新核验[PyTorch2.8 topk官方文档](https://docs.pytorch.org/docs/2.8/generated/torch.topk.html)，并列索引仍不保证稳定；不以“低index优先”作为API契约，不修改既有tie规则。无GPU时不引入新topk/kernel实现，也不把全宽恢复称为更快。既有探针v2及Step2零概率已知边界继续保留，未被本修复关闭。

复现（仓库根，CPU）：
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s sparse_attn/ops/tests -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -O -m unittest discover -s sparse_attn/ops/tests -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_e139*.py' -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -O -m unittest discover -s tests -p 'test_e139*.py' -q
```

## 历史影响与下一步

需清点同时满足batch=1、candidate-topk=1、bs≠64且通过其他门禁的历史格，区分内部回退与真正返回的错误选集；受影响结果保留原始，使用新身份重跑GPU选集/输出/质量和完整成本。bs64生产锚点、candidate开关关闭、reference或其他门禁已回退的格不在本缺口范围，不据此宣布全体实验失效。非64现在可能执行更宽topk，性能未测；不能把减少异常回退推断成端到端提速。

未完成：非64候选列泛化的GPU契约、零概率/tie独立验证、真实路由性能及overlap。持续任务保持。
