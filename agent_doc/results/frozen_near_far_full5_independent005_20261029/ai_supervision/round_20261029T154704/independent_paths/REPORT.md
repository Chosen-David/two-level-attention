**结论：已独立验收的 1227 配对输入、3681 个输出及对应门控／缓存回执为 `usable-with-scope`；1300 配对全量验收仍为 `pending`。** 本轮未重评分，不签发整体实验完成。

截止 **2026-10-30 00:06:56 CST**，结束盘点观察到 1230 条 complete，其中新增的 GovReport **107、132、195** 尚未完成本轮输入／路径验收；另有 70 配对尚未 complete。共 64 个 manifest、57 个完成分片标记，未见 `FINISHED.json`、最终 summary/validation 或 `COMPLETE.json`。

**实际验收覆盖**

| 数据集 | 已验收配对／目标 | 已解码输出 | 层—请求／臂 | 尚未验收配对 |
|---|---:|---:|---:|---:|
| Qasper | 200/200 | 600 | 7200 | 0 |
| HotpotQA | 200/200 | 600 | 7200 | 0 |
| Musique | 200/200 | 600 | 7200 | 0 |
| GovReport | 183/200 | 549 | 6588 | 17 |
| RepoBench-p | 444/500 | 1332 | 15984 | 56 |
| **合计** | **1227/1300** | **3681/3900** | **44172** | **73** |

七批覆盖为 **1100＋14＋17＋17＋46＋3＋30**，无重复计数。结束前重新读取全部已验收请求，**1227 个文件哈希全部保持一致**；另复核 204 项源码、资产及分片绑定，无变化。

| 检查 | 实际结果 |
|---|---|
| 冻结源码 | HEAD=`3e4a6885604ac3e719ee92a4906f87b532535314`；72 文件同时匹配协议 SHA256 和该提交 Git blob |
| 输入重建 | 1227/1227 的原始长度、最终长度、截断标志、token SHA256 一致；答案匹配原始行 |
| 截断 | GovReport 3 条、RepoBench-p 6 条；输入加生成预算均不超过 40960 |
| 输出解码 | 3681/3681 严格匹配；1983 臂以 EOS 结束，1698 臂达到上限；无 EOS 后继续生成 |
| 配对一致性 | 1227 请求三臂首 token 一致，三臂 prefill profiles 一致 |
| 门控与回执 | 132516 条末步 TLI 回执通过；每条 8 个 KV head，选择数范围 132–1024 |
| 缓存 | 397548 个层快照通过独立数量、容量字节、节点顺序、可见性及守恒检查 |
| Far 决策一致性 | Dual 与 Far 臂 **44172/44172** 层—请求一致 |
| native 标记 | 累计检查到 64 条 Full 标记；未独立重跑其生成比较 |

**每任务、每臂门控覆盖**

单元格为 **层—请求数 / 涉及请求数**。层分母和请求分母见前表；同一请求可涉及多个类别，因此请求数不能横向相加。Near、Far 分别表示 Near-only、Far-only。

| 数据集 | 臂 | Keep | Near | Far | Both |
|---|---|---:|---:|---:|---:|
| Qasper | Full | 7200/200 | 0/0 | 0/0 | 0/0 |
| | Far | 6299/196 | 0/0 | 901/113 | 0/0 |
| | Dual | 6298/196 | 1/1 | 901/113 | 0/0 |
| HotpotQA | Full | 7200/200 | 0/0 | 0/0 | 0/0 |
| | Far | 6618/200 | 0/0 | 582/189 | 0/0 |
| | Dual | 6601/200 | 17/3 | 581/189 | 1/1 |
| Musique | Full | 7200/200 | 0/0 | 0/0 | 0/0 |
| | Far | 6766/200 | 0/0 | 434/194 | 0/0 |
| | Dual | 6760/200 | 6/6 | 433/194 | 1/1 |
| GovReport | Full | 6588/183 | 0/0 | 0/0 | 0/0 |
| | Far | 5579/181 | 0/0 | 1009/138 | 0/0 |
| | Dual | 5578/181 | 1/1 | 1009/138 | 0/0 |
| RepoBench-p | Full | 15984/444 | 0/0 | 0/0 | 0/0 |
| | Far | 13512/438 | 0/0 | 2472/350 | 0/0 |
| | Dual | 13504/438 | 8/8 | 2472/350 | 0/0 |

Dual 合计 Near-only **33/44172 层—请求，涉及 19/1227 请求**；Both **2/44172，涉及 2/1227 请求**。

**机制与实际路径**

源码核查支持：prefill 使用末 `min(256,L)` 个 query、因果 attention、head mean 后再对 query 求 mean；Near/Far 分别严格比较 `<0.05`。冻结后禁止改变决策，decode 不重新标定。原始 Q/K 未保存，因此本轮重算的是保存 mean 对应的决策，未从真实 GPU 张量重算 mean。

实际调用链为冻结源中的：

- `prefill_frozen_skip.py:95–104`：采集 profiles、逐层决定并冻结；
- `qwen3_attn_patch.py:294–300`：分区 cache 的 decode 路由；
- `frozen_region_tli.py:69–82`：更新 cache，使用可见 K/V，调用真实 TLI scorer/selector；
- `frozen_region_cache.py:114、167、194`：冻结、老化迁移、排除不可见 staging。

pc02 活跃 worker 的命令、cwd、PYTHONPATH 均指向 dispatch 指定冻结源和输出路径。tail32 选择维度为 **48–63、112–127**，最终 attention 消费完整 **128 维 K/V**。L1 配置为总预算 128 页、Near 32 页、Far 至多 96 页，另有边界页 bypass；L2 总预算 1024，Near token 配额关闭，采用自由竞争。

Near-only 保留不可见 staging；Far-only 物理删除 Far，后续 Near 老化进入被删除区域时永久淘汰；Both 只保留 sink/SWA。已验收记录中 **11 个 Near-only 层—请求**出现 Far 增长：GovReport 98 增加 512，其余 10 个各增加 64。

**指定质量关联，未重评分**

| 记录 | 本轮独立核查 |
|---|---|
| Musique 56，层 11 Both | Near/Far mean=`0.0462979525/0.0344907008`；可见 token 保持132，永久丢弃 `14493→14496`。三臂预测完全相同，保存分数均0；不支持将该样本质量差归因于 Both |
| GovReport 98，层30 Near-only | Near/Far mean=`0.0436926708/0.5672893524`；511次 decode 更新中，Far `23356→23868`，staging `4142→4141`，无永久丢弃，支持8页老化迁移 |
| GovReport 98 输出关联 | Full/Far预测相同，Dual不同；保存分数分别 `0.3039073761 / 0.3039073761 / 0.2739322489`，Dual低 **2.997513个百分点**。这是记录关联，不能单凭此证明因果 |
| HotpotQA 135，层7 Both | 两区域 mean 均低于0.05；可见132，永久丢弃 `12295→12298`，补充第二个真实 Both 请求覆盖 |

**六项 verdict**

| 检查 | verdict | 证据范围与待办 |
|---|---|---|
| implementation | **pass，限定范围** | 冻结72源码、调用链、已查记录的门控／缓存／回执一致性；不认证未记录的具体 kernel 分支 |
| reference_boundary | **pass，限定复用** | 核对旧报告、脚本、结果和独立执行日志哈希；复用396缓存用例、16236快照、400门限组合、20 mean参考对照；本轮未重跑 |
| data_integrity | **pass：1227；全量 inconclusive** | 已查输入／输出／身份／字节稳定；剩73配对待验收 |
| numerical_sanity | **inconclusive，评分待另一会话** | 已查 profiles 有限值、决策和缓存数量约束通过；未重算官方评分、AVG5、CI或最终汇总 |
| measurement_validity | **pass，协议范围** | 已查请求三臂同输入、同任务生成上限；KV字节仅节点容量，attention tokens仅工作量代理，不支持速度或峰值显存结论 |
| reproducibility | **pass：CPU重建；GPU inconclusive** | 独立输入重建、输出解码、门控／缓存交叉检查通过；未独立复现GPU生成 |

现有 `verification_plan.json` 是实质要求清单，**并非完整形式化 schema 合同**；本轮没有补造合同，也不因缺少完整历史权重哈希否定上述已实际验证的限定范围。

**最终哈希绑定**

请求摘要算法：将 `requests/<dataset>/<四位行号>.json → 文件SHA256` 映射按键排序，以 `json.dumps(..., sort_keys=True, separators=(',', ':'))` 编码后计算 SHA256。

| 绑定对象 | SHA256 |
|---|---|
| 已验收1227请求集合 | `7980536634d2ba02adb6689d680cd8477bb24b41693a8ddfc3eda68b5b79aba9` |
| 结束复核204项绑定映射 | `04c63c2e3a67850250baa8cf796de4da2571eeae21ca867f0285214b2c140de9` |
| 72源码相对路径哈希映射 | `b7c4b4401f9aacfb3b8ce401729b7b2fea22a45a07c140e5e2907d5c08468199` |
| `manifest_rank0.json` | `b29e2989b1f8d379102079d6d1e9020e5aa9620f70be29b16e27f1b03e3d576a` |
| `verification_plan.json` | `6bd9754a06f7c820645e60da2df4ce197c36b00da9e0ccd34c80e74f69268fc9` |
| 当前交接文件 | `d956c259e3b86a0f6c5e5199116349fad22a0a0543c1154111bb7382c420d9b0` |
| 第二轮边界报告 | `120af0e48127145fd95380d2e9e1ae833a7037f27d145095de408b542f3d5b76` |
| 第二轮独立执行日志 | `6a609790816815e3c1e66cc219ce3319a3503cb470ac6eda270a868ec23b70ae` |
| 第三轮输入报告 | `5c0511f106dfa72f06a53ca78aea19d6aedfb939e59d867b213abc3aeec1878a` |

五个数据文件、模型 config、tokenizer config 均匹配协议；tokenizer.json、vocab、merges、generation config 当前哈希也匹配第三轮记录。第三轮680子集的 **746项文件绑定全部匹配当前字节**。完整资产哈希、执行代码和批次结果保留在工具日志。

**保留局限与 pending：**未独立GPU生成复现；未验证原始张量、逐步实际选择集合及迁移后的GPU数值；完整模型权重历史绑定和部分kernel环境仍缺。`native_full_parity` **不等于 dense parity**。后续须补验剩余73配对，并在同一最终1300集合及哈希上完成评分与汇总验收。

所有 exec 使用 `login=false`；CPU tokenizer 检查禁用模型框架、离线加载并设置 `PYTHONDONTWRITEBYTECODE=1`、`python -B`。未写文件、加载模型、运行GPU、安装依赖、修改生产或创建调度器；报告交外层保存。