**结论：本轮 680 配对的输入 token、输出解码和末步 TLI 回执一致性为 `usable-with-scope`；整体实验验收仍为 `pending`。** 未重算评分，未重新生成模型输出，不外推全量 1300 配对。

范围固定为 `ai_supervision/round_20261029T151704/integrity_initial.json`，SHA256：
`7dc0117c4237d9383d8ab7ad4777d651cf759161958b6c9698857ef9b2a373a3`。

**实际完成的检查**

| 检查 | 结果 |
|---|---|
| 固定文件绑定 | 本地 746 文件匹配；pc02 检查前后均为 746/746 匹配 |
| 冻结源码 | HEAD=`3e4a6885604ac3e719ee92a4906f87b532535314`；72 个源码文件同时匹配 manifest 哈希和该提交 Git blob，结束复核通过 |
| 原始数据及配置 | 五个数据文件、模型 config、tokenizer config 均匹配冻结协议 |
| 独立输入重建 | **680/680** 的 `raw_tokens`、`input_tokens`、`truncated`、`input_sha256` 全部一致 |
| 输出解码 | **2040/2040** 的 `generated_ids` 解码严格等于 `prediction`，长度等于 `generated_tokens` |
| 停止条件 | 1077 臂以 EOS 结束；963 臂达到生成上限且末 token 非 EOS；没有提前出现 EOS 后继续生成 |
| 配对首 token | **680/680** 三臂首 token 相同 |
| TLI receipts | **73,440/73,440** 条末步层回执通过所查字段和数量约束 |
| native parity 字段 | 仅 **33 条 Full** 存在 `native_full_parity=true`，恰好对应 33 个完成分片各自首条请求 |

输入重建在 pc02 `/usr/bin/python` 执行：Python 3.9.9、Transformers 4.56.2、Tokenizers 0.22.2。独立代码经 SSH stdin 执行，没有调用生产 `prepare_ids` 或 `validate(out)`。

重建顺序为：原始行 → 冻结 prompt 格式化 → 无特殊 token 编码 → 超过 31500 时首尾各段解码拼接 → 非 RepoBench-p 应用 chat template（`enable_thinking=False`）→ 再编码及 SHA256。覆盖 Qasper、HotpotQA、Musique、GovReport 各 107 条，RepoBench-p 252 条；其中 GovReport 2 条、RepoBench-p 4 条截断。所有输入加生成预算均未超过模型配置的 40960 上限。EOS 按当前 `generation_config.json` 为 `[151645,151643]`。

**实际路径与预算解释**

冻结源码的调用链为：

- `qwen3_attn_patch.py:294`：分区 cache 在 decode 时进入 `frozen_region_tli.decode`。
- `frozen_region_tli.py:69`：更新 cache，只取得可见完整维度 K/V；调用 `region_mask`。
- `region_mask`：用 tail 32 维构建逻辑位置投影，调用真实 `TLIIndexer.compute_score`、`compute_mask`；不可见 token 的 fine score 置为 `-inf`，最终 mask 再投影至可见位置。
- `eager_decoding_attn`：消费上述 mask 和完整 128 维 K/V。**32 维用于选择，不代表最终 attention 只有 32 维。**

全部回执记录的维度均为 `48–63、112–127`，每条含 8 个 KV head 的选择数；实际数量范围 **132–1024**，均不超过可见 token 数或 K2=1024。逻辑长度均等于 `input_tokens + generated_tokens − 1`，可见长度与最终 cache 报告一致，skip 标志一致。

协议与源码规定 L1 总预算 128 页、Near 32 页、Far 至多 96 页，页宽 64；边界页另有 bypass。L2 为自由竞争，sink=4、SWA=128 计入总预算。**回执没有记录实际 L1 页索引、选择 token 索引或逐步预算轨迹；本轮验证了配置、源码消费及末步数量，未验证每一步的实际选择集合。**

`native_full_parity` 是生产者执行比较后保存的布尔值，不含独立 native token 序列。源码比较对象是**仍挂载 TLI patch 的 DynamicCache 路径**，不能解释为与 dense full-attention 生成一致；本轮也未重新执行该比较。

**指定记录**

层号均为 0 起始。

| 记录 | 独立复核结果 |
|---|---|
| Musique 56，层 11，Both | Near/Far mean=`0.0462979525 / 0.0344907008`，均低于 0.05。输入 14625 token，生成 4 token；冻结及最终均仅 sink/SWA 共 132 可见 token；永久丢弃数 `14493→14496`，符合 3 次 decode 更新。 |
| GovReport 98，层 30，Near-only | mean=`0.0436926708 / 0.5672893524`。生成 512 token、511 次更新；Far `23356→23868`，增加 **512**；不可见 staging `4142→4141`；可见数 `23488→24000`；永久丢弃为 0。记录支持发生 **8 个 64-token 页的老化迁移**。 |
| HotpotQA 69，层 23、25–33，Near-only | 10 层均满足 Near `<0.05`、Far `>0.05`。生成 2 token，仅一次更新；staging `4140→4141`，Far 保持 12860，可见数保持 12992；**没有跨页迁移覆盖**。 |

源码 `frozen_region_cache.py:114、167、194` 分别实现冻结隐藏 staging、按新 Far 边界迁移、gather 排除不可见节点。迁移后 token 进入可见 Far；Near-only 不永久丢弃它们。

固定子集共 **7 个 Near-only 层—请求**有 Far 增长记录：GovReport 98 增加 512；Musique 81，以及 RepoBench-p 56、145、237、298、496 各增加 64。它们补充了旧报告的真实记录覆盖，**没有关闭 GPU K/V 内容及迁移数值的独立复现缺口**。

**报告与资产绑定**

已读取两份报告，并在结束前再次确认字节哈希未变：

| 相对结果根路径 | SHA256 |
|---|---|
| `ai_supervision/round_20261029T151704/independent_partial/REPORT.md` | `1022122daa08492ab77288bd9d583fc20b9d3b6444509b4753306d2ea8029feb` |
| `ai_supervision/round_20261029T144704/independent_boundary/REPORT.md` | `120af0e48127145fd95380d2e9e1ae833a7037f27d145095de408b542f3d5b76` |

旧边界报告引用的参考脚本及结果哈希也匹配：`f482fcf3…cd039`、`51e6bb42…72e0d`。按用户要求保留其限定 CPU 性质结论，未重复执行。

本次另计算并前后复核了以下当前资产：

| 资产 | SHA256 |
|---|---|
| `generation_config.json` | `2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2` |
| `tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `merges.txt` | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |

这些额外资产未被原 manifest/preflight 完整绑定；当前输入全部重建一致，但不能据此补造其历史哈希证明。

**保留限制：**未计算完整模型权重哈希、未加载权重、未复现生成或 native parity、未检查原始 GPU Q/K 和 attention 数值。回执只保存末步摘要，且未完整记录 `TLI_E127_FAST` 等消费者环境，因此不认证具体 gather/kernel 后端或性能。全量覆盖、最终 AVG5 与整体实验验收继续 `pending`。

所有 exec 均 `login=false`；使用 `PYTHONDONTWRITEBYTECODE=1`、`python -B` 和 CPU/离线 tokenizer 设置。未自行写文件、安装依赖、访问 GPU、修改生产或创建调度器。