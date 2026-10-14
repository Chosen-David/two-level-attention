# 静态冠军实现：候选内 L2 与 far 关闭净成本的窄验收建议

日期：2026-10-14 UTC。固定核查源 `f052eb916367a2b1a6eb5d689229e61841b2db22`；仅静态源码复核，没有新 CPU 张量测试、GPU计时或模型质量实测，不宣称收益。

责任范围：本方法静态设计、实现与验收归 Opus；DeepSeek 仅负责外部 baseline。本文件是仓内建议，不表示已向任何 agent 发消息或派单。用户本次明确授权日间单次 advice 发布；不改变常规夜间窗口，不追加催办。

沿用现静态冠军协议 best_mavg(.25,.125,.625)、sink128/SWA128、低频 RoPE32（16完整对）、L1/L2 compact32、refill；保留 far-on/off-refill/off-no-refill 已登记匹配控制。这里的“far-off”指 `skip_far=True`，不是关闭 gate 功能。不给新参数海选，不恢复动态 α/β/γ，不自动启动 GPU。以下优先级指后续已授权工作中的成本顺序。

## TL-L2-CANDIDATE-COST-20261014：L1 缩小支持集尚未缩小普通 L2 点积域

**事实与触发**：普通非投影、非 `dgate_short` 路径先按全长构造 fine scores：[分配/评分](https://github.com/Chosen-David/two-level-attention/blob/f052eb916367a2b1a6eb5d689229e61841b2db22/sparse_attn/indexer/tli_indexer.py#L1412-L1481)；Triton 分支同样把完整 `k_qat` 传入。随后才在 [L1 mask → softmax → GQA 聚合](https://github.com/Chosen-David/two-level-attention/blob/f052eb916367a2b1a6eb5d689229e61841b2db22/sparse_attn/indexer/tli_indexer.py#L2041-L2046) 排除非候选。因此已实现两级选择语义，但不能据 L1 保留页比例宣称 L2 点积只读候选。不是已有结果数值错误的证明。

**假设/最小对照（P0 先语义，P1 再成本）**：若候选足够稀疏，按 L1 候选页间接读或 gather 的 L2 有望少读 K；索引构造、gather、额外 launch 也可能抵消收益。仅在同一冻结配置、同一 Q/K、同一 L1 mask/缓存初态下，对拍当前“全 S 后 mask”与候选内实现；不改变表示、精度、保护区、near/far 配额、refill、最终原始全维 KV attention。覆盖短序列、尾部非整页、空区/不足候选、重叠保护与并列边界。

**必须保留的语义**：默认 GQA 的 [L1 raw-score mean](https://github.com/Chosen-David/two-level-attention/blob/f052eb916367a2b1a6eb5d689229e61841b2db22/sparse_attn/indexer/tli_indexer.py#L1838-L1845)（[avg 分支同样先做组内 mean](https://github.com/Chosen-David/two-level-attention/blob/f052eb916367a2b1a6eb5d689229e61841b2db22/sparse_attn/indexer/tli_indexer.py#L1940-L1944)）与 L2 的“每 Q-head 在其有效候选支持内 softmax，再组内 probability mean”一般不等价；不能用平均 logits 的 TopK 替代 L2，亦不能擅自把原联合归一化改为 near/far 分别归一化。此项承接 [E130 既有排序门禁](2026-10-11_phase_matched_experiment_design_by_gpt.md)，不是新增 scorer 消融。

**指标/判据**：先登记数值容差及确定性 tie 规则，比较有效候选/保护支持、最终 token IDs/实际预算和最终 attention 输出；不得只比较平均分数或吞掉 NaN。IDs 不一致须解释是否仅登记的 tie，其他差异或输出超容差即失败，停止速度主张。语义通过后才在匹配 GPU 条件、交替重复下记录 L1、候选构造/gather、L2/TopK、最终 attention、完整 selector wall 和峰值分配；包括 fallback 与冷建/更新，不只测 dot kernel。相同输入长度/生成工作量下完整成本改善达到事先登记阈值才称工程收益；未改善则保留原实现，无需扩大矩阵。

**表示口径（P0 文档/回执）**：[compact K 的实际构造](https://github.com/Chosen-David/two-level-attention/blob/f052eb916367a2b1a6eb5d689229e61841b2db22/sparse_attn/indexer/tli_indexer.py#L833-L850)确实按选中维度存 K；[min/max quant 返回浮点反量化值](https://github.com/Chosen-David/two-level-attention/blob/f052eb916367a2b1a6eb5d689229e61841b2db22/sparse_attn/indexer/tia_indexer.py#L22-L31)不是 packed int4。记录实际 shape/dtype/allocated bytes，分别报告维数缩减与数值量化，不用“4bit”推算八倍物理容量收益。不为此额外提出 packed-int4 实现任务。

## TL-FAR-OFF-NET-COST-20261014：少算 far 点积不保证净省时

**事实与触发**：[dgate_short 条件](https://github.com/Chosen-David/two-level-attention/blob/f052eb916367a2b1a6eb5d689229e61841b2db22/sparse_attn/indexer/tli_indexer.py#L1296-L1309)要求 skip_far、α/β>0、valid_length 与合法区间；命中后 [L2 先全长 repeat/float，再切保留段](https://github.com/Chosen-David/two-level-attention/blob/f052eb916367a2b1a6eb5d689229e61841b2db22/sparse_attn/indexer/tli_indexer.py#L1436-L1450) 仍展开完整 K 到 Q-head 并转 float32。与普通 [Triton GQA 复用路径](https://github.com/Chosen-David/two-level-attention/blob/f052eb916367a2b1a6eb5d689229e61841b2db22/sparse_attn/indexer/tli_indexer.py#L1451-L1475) 比较时，分支/内存成本已经改变，不能以减少 far FLOPs 推净收益；本次没有测出它更慢或更快。

**最小对照/判据（P0 静态记录，P1 匹配计时）**：先输出实际路由、far 区间、选中 ID、dtype/shape、repeat/转换字节及峰值。若实现“先切保留段再展开”或间接读取，只在同一个 far-off/refill 政策内对拍 IDs 与输出，按上一项容差/tie 门禁通过后才测成本；不要混入 far-on 与 far-off 的算法质量差异。沿用原三种 far/refill 对照，记录 selector 总 wall、TPOT、TTFT 与请求 E2E、实际生成长度，保留负收益。新优化只有含物化/索引/同步的完整成本达到预登记改善阈值才通过；否则仅能说少算点积。

**标定成本承接 E137（不新开 gate 实验）**：[全 D float 标定、GQA 展开与 float 标量读取](https://github.com/Chosen-David/two-level-attention/blob/f052eb916367a2b1a6eb5d689229e61841b2db22/sparse_attn/indexer/tli_indexer.py#L442-L458)中的 GPU tensor→Python float 会同步；[prefill 调标定](https://github.com/Chosen-David/two-level-attention/blob/f052eb916367a2b1a6eb5d689229e61841b2db22/sparse_attn/patches/qwen3_attn_patch.py#L63-L75)位于首 token 前。若既有 gate 候选被评测，须把全量标定、临时内存和 host 同步计入 TTFT/请求 wall，不以热态 decode 收益抵扣后漏账。质量、实际问题可见性、freeze/clear 生命周期继续归 [E137 现有建议](2026-10-13_1040_E137_gate_composition_scope_by_gpt.md)；`TL-E137-PREFILL-GATE-145` 已修的 pending→decode freeze 不重新列 bug。

## 已覆盖事项：仅更新当前源码锚点，不另建任务

- **Prefill**：沿用 [阶段建议 §3/§5](2026-10-11_phase_matched_experiment_design_by_gpt.md)。[当前 eager 全 prefix QK/PV](https://github.com/Chosen-David/two-level-attention/blob/f052eb916367a2b1a6eb5d689229e61841b2db22/sparse_attn/ops/eager_prefill.py#L73-L106)仍是 dense 算术后 mask；[chunk 末 query 选集](https://github.com/Chosen-David/two-level-attention/blob/f052eb916367a2b1a6eb5d689229e61841b2db22/sparse_attn/ops/eager_prefill.py#L147-L172)须完成已登记的“固定当前 query/前缀，只改未来 suffix”门禁。严格前缀主张要求候选/输出在声明容差内不变；若选择 whole-prompt 近似协议则明确标注，不能称在线前缀等价。[入口限制](https://github.com/Chosen-David/two-level-attention/blob/f052eb916367a2b1a6eb5d689229e61841b2db22/sparse_attn/patches/qwen3_attn_patch.py#L76-L82)仍为 flag+B1+无 mask；未通过既有路由/语义/成本验收前，不计 prefill 稀疏提速。行级 SWA 修复不等于上述前缀选集问题已闭合。
- **质量—成本身份**：复用 [阶段建议 E0/E6/§5](2026-10-11_phase_matched_experiment_design_by_gpt.md)与 E137，先找现存回执，不盲目重跑。每个质量—成本点闭合同一冻结配置/代码、输入与模型 revision、actual_route、dtype/聚合/量化/预算、gate/refill、预测与 scorer、计时原始样本/硬件/同步边界；计时重复可以独立 run，但须可追溯到同一冻结实验身份。自然生成质量/请求成本与固定 token 回放诊断分别标注。已有 E134 稳态微基准不替代该闭包；未闭合只报告局部证据，不宣布双32+gate联合质量/成本验收通过。容忍δ/改善阈值沿既有预登记；缺失时先补声明，不能事后按结果挑阈值。

去重范围：已检索当前 active/archive 建议与回复，并复核上述相关正文/源码。新稳定 ID 仅为候选内 L2 成本与 far-off 净成本两项；GQA/prefill/质量配对/E137继续使用原建议和既有 ID。MInference 正式入口修复不重列缺陷；FASA sha8 条件已有外部审计，本次不重复发布。不清理、归档或修改其他 advice。
