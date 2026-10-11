# 论文对比工作与接入建议
@DeepSeek

核查日期：2026-10-11 UTC。依据本库 main `5f89b039929666e4000bd9c3cb9af7aee24de05f` 与下列作者源码。供现有 S-T023 接入流程考虑，不另起调度或扩大完整实验矩阵；本说明没有运行模型或验证收益。

## 1. 主问题与优先级
论文需要回答：在保持最终原始全维 KV attention 的前提下，共享子空间、near/far 分配和两级筛选，能否以更低的完整选择/注意力成本维持实际任务质量？保护 prefix/SWA 是共同控制条件，不能单独当创新。先静态筛选、合法参数枚举与冻结，再验收冠军加子空间；动态控制器仍暂停。

主比较优先 **Quest、FASA-C、SparQ**；**ClusterKV** 是直接相关的聚类对照，但先解决代码许可和系统设定。**MoBA** 保留机制/零训练适配参考，不把这种适配当成官方训练后方法的复现。FullKV 必须保留。KV 压缩方法和 prefill 专用方法单独分类，不混成同一种资源预算。

## 2. 论文及作者开源入口

### Quest：page-level query-aware selector
- 论文：https://arxiv.org/abs/2406.10774 （ICML 2024）
- 作者库：https://github.com/mit-han-lab/Quest
- 固定版本：`01c1623bf9395009520874e989e29f683203b357`；根 LICENSE 为 MIT。
- 无需训练，主要是 dense prefill + sparse decode，完整 KV 保留。对比它回答“两级 token 细筛相对选页本身有无收益”。
- 保留 page min/max 上界、页粒度及当前页处理。官方常见 page=16，而本库 page=64 的同 token 总额不是同一选页粒度。官方优化入口有 batch=1 / RoPE 限制；Qwen3、MLA 不能凭 Llama 示例认定支持。

### FASA-C：最直接的低维频率子空间对照
- 论文：https://arxiv.org/abs/2602.03152 ；正式页面：https://openreview.net/forum?id=FnSgecCEwg
- 作者库：https://github.com/wangyifei0047/FASA-ICLR2026
- 固定版本：`96c00c46db12ccee56104175470aa29579d66c58`；Apache-2.0。
- Training-free 但不是 calibration-free：离线选 dominant RoPE frequency chunks，记录每层/头 FC、校准集及成本，校准数据不得来自确认集。FC 数与最终 token 数是两个预算。
- FASA-C 完整 KV 在 GPU，适合作为主比较；FASA-M 的 CPU offload 属于另一系统设定。
- 官方 Qwen 代码是 Qwen2 类适配，DeepSeek-R1-Distill-Qwen 也不等于 MLA。该固定提交删除了 efficiency benchmark 目录，不能沿用 README 宣称速度评测已可复现；先验收质量路径和必要计时实现。

### SparQ：按 query 选择维度的对照
- 论文：https://arxiv.org/abs/2312.04985 （按 v3 核机制）
- 作者主实验库：https://github.com/graphcore-research/llm-inference-research
- 固定版本：`ead9b6a70e3304e4666ae887cd2c6f506525fea2`；MIT。
- 作者 GPU 系统实现：https://github.com/graphcore-research/sparq-gpt-fast
- 固定版本：`f2eeb9d10e79d3ffecc48015e55157bab0f6e269`；BSD-3-Clause，勿和主实验库混记许可。
- 无需训练，主要优化 decode；以 |q| 选 r 维，读取历史对应 K 维度筛 k 个 token。完整 KV 仍保留，须计入 S×r 粗筛读取。v3 已含 GQA，不能按早期版本说不支持。
- 原法有 mean-value reallocation 等处理。A 轨若只借 selector、统一最终 sparse softmax，应标 “SparQ selector-only”；B 轨保留完整算法才称原法复现。

### ClusterKV：语义聚类检索对照
- 论文：https://arxiv.org/abs/2412.03213 （DAC 2025）
- 作者库：https://github.com/sjtu-zhao-lab/ClusterKV
- 固定版本：`c7380819588771e8c5dadacdbd33942d7c53adea`。
- **整体 LICENSE 未核实**：第三方依赖的许可不能推导本仓代码许可。先做论文机制设计，直接 vendor 前核清许可，不默认可复制。
- 无模型训练，但有建簇/周期重聚类成本。原系统包含 CPU full-KV offload 与 GPU cluster cache；本库 GPU-resident selector 重实现须单独标记，不能冒称原系统。
- 固定聚类初始化/seed、cluster 数、更新周期、末簇预算截断；将建簇、维护和传输纳入成本。依赖环境与主环境隔离，不默认安装。

### MoBA：保留，但说明训练边界
- 论文：https://arxiv.org/abs/2502.13189
- 作者库：https://github.com/MoonshotAI/MoBA
- 固定版本：`b5d58363311d3ca946f1ec444182727c15e338b5`；MIT。
- 官方 README 明确已有模型需要 continued training；parameter-less gate 不等于 training-free。
- 不训练直接接入时标 “MoBA-inspired block-mean selector / zero-shot adaptation”，用于机制比较；不得把结果称官方 MoBA 的最终能力或失败。记录块大小、选择块数、当前因果块实际 token 数。

### 条件性扩展，不阻塞当前静态筛选
- 若论文对 prefill 提出加速结论，再接 MInference：论文 https://arxiv.org/abs/2407.02490 ，作者库 https://github.com/microsoft/MInference ，固定 `29ef1974fcb5a1440ccf92d7d7996900230fddcf`，MIT。需离线 head-pattern 配置，dense decode 固定；不能拿 prefill 加速填 decode 表。
- 沿用已有 KVCache-Factory 压缩族计划：PyramidKV 论文 https://arxiv.org/abs/2406.02069 ，作者维护的统一实现入口 https://github.com/Zefan-Cai/KVCache-Factory ，固定 `68cd9551a63fedf362ddb0edb008badd04c24996`，MIT。它不是所有收录算法的原作者实现。原生主路径为 Llama/Mistral、batch=1；与 full-KV selector 分表或做明确预算曲线，不因已有脚本就认定 Qwen3/MLA 接入完成。
- HiP 可在需要检验多级层次剪枝时后补：https://github.com/DeepAuto-AI/hip-attention/tree/5f37a7fd77abe4d362cc7b7cd4ea24a3748614c1 ，当前库混有新版本且 LICENSE 为 FSL-1.1-MIT，不能简写 MIT；本轮不额外扩矩阵。

## 3. 公平接入与验收（复用既有 A/B 轨）
1. **预算口径**：本库当前代码的 K2 是最终总预算，K2_mid=K2−sink−swa。保护 token 不参与竞争，不等于预算外免费增加。报告实际唯一 token 数、保护区、当前页、dense 豁免层、page rounding、粗筛维数/bytes 与额外索引显存。
2. **显存与计算分开**：完整 KV 保留时，降低 K2 不等于降低 resident KV 容量。分别报原始 KV、索引、临时峰值、读取量、selector/topk/gather、准备维护及完整 TTFT/TPOT/E2E。
3. **A 轨**固定模型/tokenizer revision、prompt/truncation、BOS/EOS、样本输入 hash、scorer、阶段、预算及硬件，明确每项适配。现有 Quest 按 GQA 组 mean 共享选页，不能叫与原作者逐 query-head 语义完全一致。
4. **B 轨**保留原法完整语义及独立依赖，锁定上述 SHA/许可和适配 diff。尚不支持的模型/硬件格写 unsupported；不能将论文异平台延迟填成同台实测。作者结果、本库重实现和 vendor 复现分别标注。
5. **阶段拆开**：dense/dense、dense/indexed、indexed/dense、indexed/indexed 分别评价。原法仅 decode 时先比较相同阶段；移植到 prefill 是新增适配，需单独验收。
6. **旧压缩 runner 必须先审计**：当前 `exp/trace/pred_kvcf.py` L82–109 在启用 `--split-question` 时无条件去 question 首 token；生成上限大于1时也未在首个生成 token 为 EOS 后立即停止。两处与主 runner 新协议不一致；先修齐并回归，不能直接拿旧接口输出作公平新表。
7. **Sanity 不预设结果符号**：检查输入/实现/指标和重复误差，不强制稀疏必须低于 FullKV，也不强求同原论文差值符号。小幅更高先查泄漏、评分、实现及统计波动，再按证据判断。
8. 先用已有固定小筛选集验收接入，再进入冻结的全量/独立确认。不得按确认集分数回调规则；所有失败格和最差任务退化保留。代码下载、测试数或名字相同都不等于实际接入通过。

本说明仅建议选型与验收；不启动训练/实验，不增加 GPU、付费或下载执行授权。

