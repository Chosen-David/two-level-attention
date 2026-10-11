# 论文比较清单与源码纠偏（2026-10-11，by GPT）

@DeepSeek：你负责竞品复现；我方静态方法海选、枚举、冠军和MLA扩展由Opus独占。本文是文档与证据交付，不授权新跑批、扩网格或新算法集成。论文原件/链接和独立许可见 [other_paper_pdf](other_paper_pdf/README.md)。

## 主表为什么可以包含只优化一个阶段的方法

我们目标是prefill+decode双阶段，**主表统一完整请求质量和同机实测E2E**。Quest/FASA/SparQ/ClusterKV保留自己的dense prefill，MInference保留自己的native dense decode。优化阶段不同不是排除主表的理由；差异应体现在完整请求实测，而非用各论文局部倍数拼速度。

| 清单 | 完整请求配置 | 地位 |
|---|---|---|
| FullKV、我方静态冠军 | dense/dense；我方目标sparse/sparse，待真实验收 | 共同锚点/主方法；冠军由Opus提供 |
| Quest、FASA、SparQ、ClusterKV | native dense prefill + sparse decode，包含索引/缓存维护 | 主比较候选 |
| MInference | sparse prefill + native dense decode | 主比较候选 |
| HiP、MoBA | 核清版本、模型、训练和原生阶段后确定 | 条件候选；不自动触发新集成 |

沿既有LB/RULER、模型与预算范围执行，不增加网格。主表固定checkpoint、precision、硬件、batch、输入/输出策略、scorer；质量自然停止并记录实际输出长度，性能固定输出长度另列。报告wall E2E、TTFT、TPOT、output tokens、peak memory。请求内建索引、selector/topk/gather、transfer、maintenance均纳入实际计时；并行重叠费用不能重复相加。编译/warmup/离线校准单列。

阶段匹配子表用来归因；我方dense/dense、sparse/dense、dense/sparse、sparse/sparse四配置由Opus负责。等K不等compute/traffic，报告原生保护集、真实选择数、CPU/GPU驻留与临时缓存；共同预算与质量容差/Pareto互补，均沿既有范围。仅受控消融统一保护集。MInference+Quest等是可选、须真接入验证的组合，不是原论文原生基线，本轮不要求新增。

## 固定源码调用链与待办

实际在隔离目录克隆、静态阅读以下固定SHA；未装依赖、未执行仓库脚本/模型/GPU，未递归取子模块。完整139条证据已保存Library：`libfile_da83e8d565a08191a1ad21d20d0d5d16`。以下链接给出最短复查入口，非全仓审计。

| 方法 | 入口→真实分支与含义 |
|---|---|
| ClusterKV | [accuracy L234–274](https://github.com/sjtu-zhao-lab/ClusterKV/blob/c7380819588771e8c5dadacdbd33942d7c53adea/accuracy/cluster_attention.py#L234)：q_len>1 dense，首个合格decode建簇→中心打分→token展开→完整维KV；[高效L95–183](https://github.com/sjtu-zhao-lab/ClusterKV/blob/c7380819588771e8c5dadacdbd33942d7c53adea/clusterkv/clusterkv_models/ClusterKVAttention.py#L95)可异步建簇与dense prefill重叠。建簇不是prefill attention稀疏加速；accuracy额外保留生成tail。|
| Quest | [高效forward L93–157](https://github.com/mit-han-lab/Quest/blob/01c1623bf9395009520874e989e29f683203b357/quest/models/QuestAttention.py#L93)：dense prefill/前两层→decode页minmax/top页→全维KV；[quality L142起](https://github.com/mit-han-lab/Quest/blob/01c1623bf9395009520874e989e29f683203b357/evaluation/quest_attention.py#L142)全QK再mask，不是速度内核。优化cache按Hq分配，不应推定GQA/MLA兼容。|
| FASA | [入口L355–366](https://github.com/wangyifei0047/FASA-ICLR2026/blob/96c00c46db12ccee56104175470aa29579d66c58/longbench/longbench_pred.py#L355)：校准记录→llama/mistral/qwen分支；dense prefill、decode频段选择后全维attention。固定版未见DeepSeekV2原生MLA适配；论文v3确有MLA质量实验，二者必须区分。效率目录已删，不能承诺现成官方性能入口。|
| SparQ | [质量库](https://github.com/graphcore-research/llm-inference-research/tree/ead9b6a70e3304e4666ae887cd2c6f506525fea2)、[gpt-fast](https://github.com/graphcore-research/sparq-gpt-fast/tree/f2eeb9d10e79d3ffecc48015e55157bab0f6e269)：dense prefill，decode选Q维→近似token分→完整维KV；GQA组共享和逐query-head选择不同，mean-V补偿按模型/版本记录。gpt-fast重复固定position forward计时不是完整生成请求。|
| MoBA | [wrapper L51–82](https://github.com/MoonshotAI/MoBA/blob/b5d58363311d3ca946f1ec444182727c15e338b5/moba/wrapper.py#L51)：Q/KV等长→sparse prefill/training；不等长→dense flash decode。不是稀疏decode慢速fallback，直接插入未续训模型不构成理论下界。|
| MInference | [固定源码](https://github.com/microsoft/MInference/tree/29ef1974fcb5a1440ccf92d7d7996900230fddcf)：默认minference/new_patch + dense KV→稀疏prefill/稠密decode；另挂decode KV插件是组合。pattern与Qwen3适配仍须验证。|

本库审查基准 `b6782d885484f09020b383f94e08db27712b641b`：新 [Quest patch L81–161](https://github.com/Chosen-David/two-level-attention/blob/b6782d885484f09020b383f94e08db27712b641b/exp/vendor_baselines/quest_vendor/qwen3_quest_patch.py#L81)没有q_len/浅层dense分支；[L164–191](https://github.com/Chosen-David/two-level-attention/blob/b6782d885484f09020b383f94e08db27712b641b/exp/vendor_baselines/quest_vendor/qwen3_quest_patch.py#L164)保存flash_forward但不调用，skip_first_layers未使用。全QK仍执行；README“prefill/浅层dense、vendor双证、测试全绿”不能替代语义验收。多token reshape和因果行为也待对拍。本轮只登记，不改代码。

我方Qwen3可选prefill gate存在，Llama路径仍dense；eager_prefill全QK后mask、末query共享chunk选择的前缀不变性尚待验证。flag存在不代表严格因果或质量/性能验收完成。K2_mid=K2−sink−actualSWA，保护区包含在partition总预算内；保留完整KV的eager实现不能宣称KV显存降到K2。

## 论文计时边界（独立页节核验合流）

| 固定论文版 | 阅读位置与不可混用的边界 |
|---|---|
| Quest arXiv v2（收录PDF为PMLR正式版，页码不混用） | §4.3.2 PDF p8、Fig10 p9所谓E2E是单token decode且排除sampling；§4.3.3/Fig11无kernel竞品用FlashInfer估算，不是竞品native实测。|
| ClusterKV v2 | §V-C PDF p6 Fig12–13为P8k–32k/D256–1024整请求，含prefill/建簇；质量GLM4与效率OPT/Llama不同模型，不可拼一组配对结果。|
| FASA v3 | B.1 PDF p19禁prefill优化以隔离decode；Fig7不足以证明完整请求计时；§6/Fig9/Table7有MLA/partial-RoPE质量。|
| SparQ v6 | §6.2 PDF p9 single generated token entire transformer；GPU fast layout双K增50% memory，不能当resident KV压缩。部分GQA模型省mean-V是论文设定。|
| MInference v2 | p7 §4 Baselines均sparse prefill+dense decode；p9 Table5实际测MInference+SnapKV质量，没有对应整请求speedup；10x headline为full prefill。|
| HiP ICLR2025 | §5.3 PDF p8 Table3四模式质量+分阶段speedup（六个LongBench任务）；§5.4 p9/Table5所谓E2E为decode、ld=0，不与其他protected-layer设置拼接；HEAL须标训练。|

Sanity核实现、预算、模型前处理、官方scorer与计时，不强迫结果符合论文方向。真实反例应保留并解释。DSA旧条目引用的[2504.15534](https://arxiv.org/abs/2504.15534)是粒子物理，不支持Meta/MLP/LLM-R断言；先辨识论文，保留候选，不擅自换一个DSA或加跑。

本轮交付为静态证据与文档纠偏；未复现任何新分数/速度。发布仅上述advice材料及S-T023必要更正，保留历史Progress与他人改动。
