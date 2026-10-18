# 论文证据缺口：当前验收入口

按用户即时要求整理（2026-10-18）；这是既有任务的验收索引，不是新增 GPU、矩阵或重复运行派单。核对基线 main `4d51efc9ca130f08f93a90dd443dd91f154a114e`。先归集已有 JSONL、manifest、receipt 与完整成本日志，缺失则保持相应主张未验；不据仓库记录推断宿主作业已读、已启动或已停止。

| 优先级 / 论文主张 | 匹配对照与现有证据 | 仍缺什么、最小补件及关闭条件 |
|---|---|---|
| P0 公平 baseline / 跨模型质量：部分支持 | 沿用 [S-T023](../task/task_details/S-T023.md)。原生阶段/保护/层策略保留并披露，策略匹配适配器另列；同模型、输入、scorer、生成条件比较。E143a 32B RULER 两臂已有匹配 manifest 正证据（下述），不能再称完全无匹配清单。 | 承接 [remaining bounds 第4项及回复](2026-10-17_e143a_e147_remaining_bounds_by_gpt.md)：优先取回清单绑定的两臂预测、同代 receipt、输入/sample ID、权重与 tokenizer revision、scorer 原件；哈希核对、样本配对及原始重评分通过后才升级。32B LB 13任务作者 AVG 43.20 尚无同配置32B FullKV锚，8B 51.25不能替代；先查既有产物，不由本索引启动补跑。竞品还须各自 faithful-port / 原生路径验收，不拼历史配置排行榜。 |
| P0 当前质量配置的完整请求收益：未支持 | 沿用 [TL-E152-TIMING-20261016](../task/task_details/TL-E152-TIMING-20261016.md)、其 CODE-T002 / E2 后续及 [测量设计 P0-5](2026-10-15_paper_measurement_experiment_design_by_gpt.md)。CPU harness 修复不是性能结果；E152v2另一配置的作者负结果须保留。 | 先取回与质量点同实现/参数、同硬件与工作量绑定的计时原件及真实路由日志；核连续 wall/TTFT/TPOT、输出长度，计入初建、选择、更新、搬运和同步，区分冷/热与常驻/峰值显存。绑定不齐则只报告该历史配置，不能用另一质量点搭配其成本；完整请求原件和独立复算通过才关闭。 |
| P0 区域配额单独因果贡献：未支持 | 沿用 [A3](../guide/paper_figs/ablation/ablation.md) 与测量设计 §2.3。mavg/aavg 现成比较同时改变 scorer 与配额，只支持整体配置描述；aavg仍有L1，亦非“无L1”。 | 先检查已有受控原件是否固定统一 page scorer、K1/page/L2、保护/表示/精度/共享规则，并记录实际 unique 支持。只有干预及其余固定量可核验时才谈配额效应；支持量改变须相应缩小归因。缺件继续列未验，不另开实验；保留128K aavg−mavg 的 MV +26.5、MQ +7、MK2 +6 反向变化。 |
| P0 确认性泛化 / 非劣：未支持 | 沿用 [测量设计 P0-2](2026-10-15_paper_measurement_experiment_design_by_gpt.md) 与 [claim boundary 及回复](2026-10-17_e145c_e146c_claim_boundary_by_gpt.md)。用户固定任务集算术 AVG 仍是主指标，逐任务用于解释，不新增逐项零损门槛。 | 优先恢复既有 document 划分、参数选择历史、确认前冻结的容差/统计规则与配对预测；核开发/确认不重叠及选择时间。benchmark任务全覆盖、AVG接近或同分复制均不能替代独立确认；无法恢复则只作描述，不事后预注册。 |

## 本轮新增的验收定位

**32B 的积极匹配证据及边界。** `exp/trace/results/e143a_r32k_{fullkv,tli}_formal.json.manifest.json` 的 blob 分别为 `9a2e063552212c6349ae77779afa75e29e6b015e`、`49f3b28413900bc7a6a68285ee68609bccc67f7d`：两臂各11×100、11份源数据 hash 相同、max_gen=64、no-YaRN；共同 producer SHA256 `ab5c7d6307a14e4d69c51271f69358271aaac4adb46347a5239766504e26524b`、model-config SHA256 `97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb`、scorer SHA256 `e5139e2dd914d92e87e036cdff3635a5136e10f79ae3efe0a3d13cd702c7319b`，并声明 v2 同代预测绑定。这是 manifest 层匹配，不能代替实际权重 revision、JSONL/sample ID 配对与跨机原件验证；当前 main 树未见 e143a/e144a/e146c 的 JSONL，不等于宿主没有。32B LB 边界来源 `e143a_lben32b_verdict.json` blob `d4962f37bc48543faf4cc2c56ebf652566e4f496`。

**P1 旧符号建议的精确残留。** [10-17原建议及接受回复](2026-10-17_e145d_e146ca_contrast_signs_by_gpt.md) 已处理方向问题，本次不另起同任务。当前 `e146ca_series_verdict.json` blob `7ba0e60051e4ad207e5f5d2a94ed7d1621acf383` 的 `interpretation` 统一“mavg−aavg”链仍将32K写成 `+0.07`。从两份11任务 formal 数组十进制复算：58.3−58.365454545…=−0.065454545…，应显示 **−0.07分数点**。只修该方向的 prose；结构化 **aavg−mavg=+0.0655** 正确，原分数及负例保持。此为汇总算术复核，不是 raw 重评分；关闭条件是原建议的残留位置修正且方向检查通过，不由此证明区域机制。

本文不把新工程提交当新质量/GPU证据；不改原始结果、不清理旧 advice、不重复派单。原任务与回复继续作为执行入口，以上状态仅针对各项论文主张的证据闭合程度。
