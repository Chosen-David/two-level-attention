# L1-only / K2 压缩：先海选，再冻结与定向风险复核

作者：GPT（实验建议）；日期：2026-10-25。状态：待执行方 adopt/adapt/reject/defer，关联现有 TL-E155D-T6-20261021；不新增第二套任务清单。此文只提出实验顺序，不代表实验已执行、候选已晋升或论文已切换。

## 1. 当前证据与目标

核查 main `71d2d23`：当前声明几何为 pooled K1=64、near32/requested near extent4096、K2=1024、sink4/SWA128、tail32、near-avg/far-minmax、iter3。**当前 pooled K64 海选集逐任务不掉尚未证明。** E140h 的 48.10 是旧固定配额 AVG5，且 musique/gov_report/repobench 本身有负差；不能转移为当前配置的证明。T6 的三长度质量属于 K1=128。

E172 的 iter3 K128 锚 18.675/18.888/19.388 ms 与 K64 17.715/17.957/18.464 ms 对应约 5.1%/4.9%/4.8% 的进一步 TPOT 降低，不是全部融合收益。新 E174 已附 Qwen3-32B 逐臂 JSON，但计时及 Graph/eager 对拍不等于任务质量验证；128K rescue 协议须单列。

**顺序：不先冻结方法 → 小范围海选两条路径 → 在质量合格者中选最快 → 冻结获选配置 → 复核历史严重跌分任务 → 通过后再继续。** 本轮不要求每个候选跑全量，也不自动安排全套测试。

## 2. 海选：记录每次身份，不等于提前冻结策略

- 沿用 E155d 声明的五任务：`hotpotqa, musique, gov_report, qasper, repobench`（入口名 `repobench-p`），首先建立当前 pooled K64/K2=1024 的逐任务基线及匹配 FullKV。复用原海选样本清单、样本数、截断/长度与 scorer；逐项核来源，缺失就标缺失，不能从旧 AVG5 拼接。
- 有界候选：①当前 L1+L2，K2=1024；②K2=768；③若上一档质量合格再测512；④同一 L1 的全部候选页直接做原维注意力（L1-only）。768/512是1024向下的两档可行起点，不是已证甜点。K1、near/far几何与其余开关固定，不扩成几何网格。
- 每次绑定代码/实际加载模块哈希、完整有效参数、模型/tokenizer revision、precision、输入 IDs/顺序/截断/实际长度、scorer、prefill/decode阶段、seed及generation。预期页宽64、near/far32/32；tail视图与L2 same、L2 quant=false、boundary bypass=true；静态/动态far gate、static pair、async、投影等关闭。以构造后有效值和实际分支为准，不能把遗留 CLI 字段直接当运行语义。不同模型的实际维度单独记录，首轮沿既有 Qwen3-8B 海选，不借其结果认证其他模型。
- K2是包含**去重保护token**的总上限，不是额外中间token数。报告真实保护并集、边界页、候选token、pooled winners及最终unique support；短序列重叠须按实际值计算。L1-only无同样1024上限，应据实报告支持规模。
- 两路径共用同一 L1页ID，边界页/保护token处理一致；边界页只进入候选池，保护token才直接旁路。L1-only优先使用高效page-indexed主注意力，不强迫无必要的大gather；如仅有受限原型，单列实现限制，不据此宣布L2获胜。实用最快版本可另列，但不得混入固定L1的机制对照。
- 接纳实测前检查非整页物理cache capacity、valid length、尾页索引、短缓存及保护区边界；CPU局部通过不替代实际CUDA入口正确性。未修复/未验证的越界路径不得进入有效排名。

## 3. 质量资格与成本排名

海选逐任务同时报告相对匹配 FullKV、当前 K64/K2=1024 的差值；AVG5只作摘要，不能抵消单任务下降。目标为不进一步掉分；不要自行引入容忍阈值。运行前明确比较基准与资格规则，若有负差保留失败并暂不晋升，不能因 CI 跨0判等价；样本不足或身份不匹配记 inconclusive/NR，不算通过。若要采用非零非劣界限，须先取得明确决定。有限样本不能证明绝对无损。

质量合格后按同机、同输入/生成长度、隔离负载的完整decode TPOT排名；记录full selector、attention、gather/TopK/reduction等分项及未归因残差。对比双方支持的 eager/Graph 路径，优化实现差异清楚列出；不可用退化dense Graph作主基线。分别报告首次请求、prefill/准备/cache fill/capture与稳态replay，墙钟完整请求与decode间隔分开；capture-only profiler不能代替replay归因。固定重复协议，报告全部有效重复和离散度，不跨机器挑最快点。K2支持缩小不等于释放原始KV；far eviction存储实验另线。

## 4. 海选胜者冻结后：只做定向风险集，不立即全量

先从现有逐任务记录列出负差全表并固定风险清单、受影响长度、样本及比较基准，再看新候选结果，禁止事后删失败项。以下由 T6 pooled K128 与分别归档的 FullKV 分数复算，是**历史风险线索，不是当前K64失败证明**：

| 长度 | 任务 | FullKV → T6 | 差值（点） |
|---|---|---|---|
| 32K | CWE | 83.50 → 74.10 | −9.40 |
| 32K | MultiKey-2 | 76 → 71 | −5 |
| 64K | MultiKey-2 | 50 → 39 | −11 |
| 64K | NIAH-single-1 | 100 → 94 | −6 |
| 128K | MultiKey-2 | 57 → 39 | −18 |

这是各长度最突出的风险起点，不设“其余无风险”结论；同时登记其余历史负差（VT、其他NIAH/CWE等）及本轮海选新失败，按已声明风险准则补入必要检查，不自动扩成全套。源记录不完全匹配，复核时重新建立兼容 FullKV 与当前 K64/1024 对照；分别展示对 dense 的绝对损失、相对当前基线是否继续退化，不能用“不比旧低分更差”冒充恢复dense质量。已有严重dense差距未解决时单列未关闭，不能晋升为“无损”。

任何定向任务不满足预先声明准则：保留结果、不晋升、不继续广泛部署，定位/修订后复核受影响项。通过仅允许继续下一阶段，不外推未测全数据集或模型。记录重复调参次数；尽可能使用未参与海选的样本作确认，避免反复适配同一验证集；没有独立样本就明确探索性。

## 5. 最小交付

| 产物 | 必需字段 |
|---|---|
| trial manifest | trial ID、来源commit/运行模块hash、上述有效参数、输入/模型/scorer身份、实际分支 |
| quality CSV + predictions | dataset/task、length、n/缺失、FullKV/1024/candidate分数、未舍入delta、样本配对键、可用时paired CI与重采样单位 |
| timing JSON | 环境/负载、eager/Graph、setup/first/steady边界、重复层级、selector/attention/TPOT、真实unique counts、失败/OOM |
| decision | 每候选逐任务pass/fail/inconclusive、全部失败、速度排序范围、冻结胜者身份或无合格者；定向风险复核后再决定下一步 |

不改SGLang，不将该建议等同实验授权或论文切换；由执行方关联既有任务登记取舍。

## 来源（固定核查版本）

所有相对路径均可在 [main快照71d2d23](https://github.com/Chosen-David/two-level-attention/tree/71d2d230e3a5e7bba7e67c2fb43269658abc0ff6) 核对：

- [现行配置声明](../CURRENT_NARRATIVE.md)；[当前Graph几何/选择实现](../../sparse_attn/indexer/tli_indexer.py)；[E174实际treatment manifest](../../exp/trace/results/e174_iter3_k64_32b/tli_graph_k64_L32768_a1.json)。
- [海选五任务与旧网格](../../exp/trace/results/e155d_grid_verdict.json)；[E140h旧固定配额](../../exp/trace/results/e140h_k64sweet_verdict.json)；[E172](../../exp/trace/results/e172_iter3_k64_qwen_verdict.json)；[E174判决](../../exp/trace/results/e174_iter3_k64_32b_verdict.json)。
- [T6与FullKV各长度JSON及来源字段](dac_manuscript/figures/ruler32k_comparison/inputs/)：`t6_ruler_q8b_L{32768,65536,131072}_champ.json` 对 `e116e_ruler32_formal_fullkv.json` / `e119_ruler64k_formal_fullkv.json` / `e119_ruler128k_formal_fullkv.json`。
