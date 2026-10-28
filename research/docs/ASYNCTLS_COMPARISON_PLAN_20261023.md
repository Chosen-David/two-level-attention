> Historical design from PR #5. Preserved for review, not current implementation status.
> See agent_doc/results/pr-unification-20261027/validation.md for current disposition.

# AsyncTLS 对标：分阶段测试设计（2026-10-23）

状态：设计 draft；没有启动 GPU、下载权重、扩大现有实验矩阵或产生性能结果。
代码基线：`a7bcdddf727aa99196d925b9f106d785ce9273d4`。
本方案接续 S-T023、2026-10-13 阅读方案和 PR #4 的 Graph 设计；不是重复实现计划。
长度门禁修复不在本 PR。执行每一硬件阶段仍需已有资源授权及相应前置验收。

## 1. 本轮重新核验及接入缺口

原文固定为 [AsyncTLS v1](https://arxiv.org/html/2604.07815v1)，实际重读 §4.1、§4.2、§5。
§4.1 为符号分解的块打分、校准通道与量化索引、逐 Q head softmax 后 GQA 平均、最终原维 attention。
§4.2 使用上一时刻候选并增量预取，因此不能沿用旧方案“offload 不影响输出”的假定。
§5 配置 B=64、128 块、GQA 索引32维、INT4、token预算512/1024/2048；未明确 CUDA Graph。
作者无 offload 延迟报告为 Qwen/GLM 2.3/2.7倍；96K offload 吞吐1.84/4.70倍使用 bs6 对 bs1，不能移作同批延迟结论。
保护集、前 dense 层、精确校准集和全部计时边界未由这些段落闭合，保持未知。

截至本轮，对论文链接、arXiv/Hugging Face 页面及精确名称代码检索，未核得作者认可的官方仓库、固定 commit 或代码许可。
这不证明官方代码不存在。论文 CC BY 4.0 不自动授予未知代码许可。
如后续取得官方仓库，先记 repo/commit/LICENSE/依赖/原生模型支持，再决定复用；
在此之前只能称 **AsyncTLS-style 论文机制重实现**，不得称官方复现。

本地仓库事实（上述 SHA）：

- `benchmark/RULER/pred_ruler.py` 的 asynctls 分支导入
  `exp/vendor_baselines/asynctls_vendor/qwen3_asynctls_patch.py`；
  LongBench 同样导入；该目录及 `identify_channel.py` **不在提交树**。
- `sparse_attn/arguments.py` 有 channel_file、token_budget、block_size、top_kb 参数；
  参数存在不等于可运行实现。不得从其他机器未冻结的 PYTHONPATH 静默补足。
- `tls_attn/` 有 generator、cache、TileLang风格 ops 和 offload 路由，
  但本轮未确认其作者来源、许可、性能入口与 asynctls_vendor 的绑定；不冒称没有任何相关代码，也不直接当官方实现。
- `benchmark/RULER/pred_ruler_batch.py` 的 B>1 方法集合仅 none/fasa_vendor/tli；
  AsyncTLS 不在其中。E152v4 只接受 none/tli 且 B=1。
- 当前 open PR #1–#4 未包含这份具体对标计划；#4 的通用 Graph 计划继续保留。
- 检索 `agent_doc/results/` 相关 baseline/graph/async 路径，未发现可直接满足本方案的独立三方匹配验收。
  旧结果保留为历史线索，不因本次设计失效或被删除。

旧阅读方案中的比例分区叙事、无源码未来式及“offload不影响输出”均不用于当前方法定义。
旧方案保留历史；本文件给出新增的可执行性、质量配对与 Graph 公平门禁。

## 2. 冻结比较对象：两条比较口径不可混表

### R：GPU resident 公平主比较

FullKV / LoDA-SA / AsyncTLS-style，同模型、权重、tokenizer、dtype、设备与软件栈；全部 KV resident，offload=false。R 的 AsyncTLS-style明确使用当前步 M_t 同步筛选；
上一时刻 M_{t-1} 的选择滞后是单独适配轴，不能仅凭offload=false判为已关闭。O轨再核论文lag与预取。
优先复用已注册 Qwen3-8B、B=1、32/64/128K 中经容量核实的两个长度锚点，最终 K=1024。
这是分阶段设计，未选择或启动新矩阵。不能把 requested L 当实际 token 数；每请求记录真实长度。
较大 batch 仅在三方支持并通过 ragged/padding/Graph 门禁后追加，不能用我方单独支持的 B>1 外推三方结论。

我方冻结对象来自当前 E152v4 treatment 记录：page64、K1=128、K2=1024、nl=4096、nb=32、
near=avg、far=minmax、tail32、cmp4、nt=off、sink4/SWA128、boundary bypass on、gate/layer-skip/kmeans off、
native dense prefill。这是**本方案参考配置身份**，不改默认、不把历史其它配置重标。
实际运行还须绑定解析后 manifest、已导入模块字节及前 dense 层；任何 null/缺省未解析值阻止“配置相同”结论。
nt=off 是兼容接口哨兵，不恢复当前方法的 nt/gamma 调参。

- **P（paper-style）**：保持可核的 AsyncTLS 机制和原生保护/层策略；
  论文未写清者标 adaptation/unknown，不能为匹配我方而悄改后称严格复现。
- **N（normalized）**：显式统一 sink4/SWA128、前 dense 层及最终去重 K=1024；
  标为 controlled adaptation。P 与 N 分开标识，不将 N 当论文作者原设置。
- FullKV 保留全部因果可见 token，不强迫其 K=1024。
- 粗筛 page size、各自 K1/kb、候选页数和候选 token 数分别披露，不强制相同粗筛预算。
- 最终预算仅在 sparse-active 层逐请求/KV-group记录 `|unique(selected ∪ protected)| <= min(K, visible)`；
  保护重叠去重、短序列缩窗、边界页只获得候选资格。保护 token 是否再次占 top-k 槽必须实测披露，不宣称恰好填满 K。
  前dense层单列all-visible，不套稀疏K上限，也不能用dense层均值掩盖稀疏层预算。
- 我方 L2 当前全历史打分再 mask，不把候选缩小画成已减少所有 L2 logits 计算。

### O：offload／容量吞吐（本轮不执行、不阻塞 R）

另开结果轨，记录 CPU pinned 内存、PCIe/NVLink、prefetch lag、传输字节、stream/event等待、
命中/失配、冷启动、重捕获及 host staging。相同 batch 延迟与各自最大可容纳 batch 吞吐分表。
bs6/bs1 可说明容量约束下系统吞吐，不能说明同批延迟；OOM作为容量结果保留，不删失败行。
未取得可验证引擎、许可或正确性时保持 unavailable，不能用“关 offload 的 torch mask”代替该系统轨。

## 3. 最小分阶段判别清单

所有阶段共同绑定下一节的冻结 manifest。前置失败停止依赖阶段；保存失败原件，不只留通过摘要。
CPU 测试为本地小张量，不用模型权重；GPU/质量任务只列设计，不能由此文件自动启动。

| ID | 目的、冻结输入/预算 | 入口与输出 | 验收及失败隔离 | 依赖/资源 |
|---|---|---|---|---|
| A0 来源/可运行性 | 固定源码、许可、校准文件哈希及 R/P/N身份 | 下节只读 inventory 命令；JSON记录 missing_paths、source/dirty/module hashes、支持模式 | 缺 adapter/校准/许可不以成功状态派发；未知官方来源标 style | 无前置；CPU/磁盘只读 |
| A1 选择语义 | 合成 Q/K/V；含正负Q、GQA head冲突、末页、短序列、候选<K、重复ID、保护交叠、padding | **新三方语义 oracle 待实现**；每case存 logits/probabilities/pageIDs/tokenIDs/unique预算及最大误差 | 符号分解对直接max表达式；mean-softmax与softmax-mean必须有产生不同排名的负例；pad/未来token永不选；同分允许同分集合而非任意强求稳定排序 | A0；CPU真实张量；没有 GPU |
| A2 attention与缓存 | 先固定同一选集隔离筛选误差；再全候选且K>=visible退化dense；多请求不同长/死亡行/连续两请求 | **跨方法集成fixture待实现**；输出/缓存长度/route计数/最大误差/每层mask | fp32 oracle预设atol=1e-5, rtol=1e-4；bf16候选atol=1e-2, rtol=1e-2须先在dense重复误差上确认；严禁据稀疏对dense输出差判代码bug | A1；先CPU tiny模型；GPU另授权 |
| A3 eager/Graph等价 | 同方法同输入、种子、cache和预算；值变/长度增长/越bucket/请求重用/死亡行 | 既有E152v4仅none/tli B1；三方/batch Graph harness **待实现**；逐步选集/输出/token及graph事件 | eager对graph同方法对拍；requested/enabled/captured/replayed/fallback均有实际证据；错误配置fail closed；不支持单列，不静默fallback混入Graph均值 | A2；最终需CUDA，当前未执行 |
| A4 resident成本锚点 | 三方同一device，B1，同真实L、固定生成128且无early stop，先两个已批准resident锚点 | 现有E152v4可作none/tli诊断，**三方完整请求计时入口待扩展**；schema见§5 | eager/eager与graph/graph各自匹配；缺任何一臂模式不给同模式比值；交错顺序、独占/共租记录；准备/维护计入完整wall；负加速保留 | A3；待独立资源/矩阵授权后单卡串行，先容量预检；无新GPU启动 |
| A5 质量smoke | 已有固定任务ID小子集，三方完全相同token输入；固定generation | 现有RULER/LongBench入口见§4；AsyncTLS依赖缺失先blocked；逐例预测+回执+scorer hash | 用于排崩溃、silent dense fallback、身份错配；不能以smoke通过称无损 | A2、A0恢复；GPU另授权 |
| A6 正式质量确认 | 沿既有全任务集合，固定开发集/确认集，禁止用同一格选冠军再证明非劣 | 复用formal scorer；**配对document-cluster CI跨方法消费者需接线/核实**；逐任务/宏均值/最差task/CI | 事先冻结容差、检验方向与多重比较策略；未过非劣不等已证退化；保留所有失败任务和缺失率 | A5；只有原矩阵获准且输入闭合才跑 |
| A7 联合判决 | 质量与计时同配置/实现/模型身份；输入集差异明确 | 三方report（实现状态、质量、TTFT、TPOT、wall、吞吐、显存） | 不将跨配置最好质量与最好速度拼一个点；只有有效匹配对照可比；独立审查签收 | A4+A6；CPU汇总 |
| X 扩展候选 | 512/2048预算、第三长度、B2/4/8、全候选消融、MLA、offload | **未注册执行命令** | 按具体未解主张申请，不自动全笛卡尔积 | 主轨结束后另定；本轮不执行 |

A1另查：Q/K norm、post-RoPE位置及YaRN、attention scale按原文sqrt(d)冻结并核实现，不悄换子空间dim；AsyncTLS L2 softmax归一化域明确为候选内，不能偷换全历史域；
量化校准/clip/scale/zero-point/packing/dequant/尾部补位；索引只读K与最终原始K/V分离。
不把 DS 校准通道替换为我方低频尾维。检查前dense层及prefill/decode的实际patch计数，不能只读参数名。
A2的padding、cache ownership和长度检查需真张量，测试 double 只可证明控制流。
A3必须包含 stale buffer/storage地址、重捕获计数和配置签名失效；同一graph=true布尔不构成证据。

## 4. 真实入口与明确待实现部分

以下都是源码核对得到的入口模板，**本 PR 未执行模型命令**。
先设置本地已存在 `MODEL/DATA/CHANNEL/OUT/RUN`，不下载权重；
`OUT` 必须新目录，`RUN` 唯一，禁止复用历史预测名。旧shell有宿主硬编码路径，不直接派发。

安全只读检查（不import模型、不连GPU）：

```bash
git rev-parse HEAD
git status --short
git ls-tree -r --name-only HEAD -- exp/vendor_baselines/asynctls_vendor tls_attn
python - <<'PY'
from pathlib import Path
for p in (
    "exp/vendor_baselines/asynctls_vendor/qwen3_asynctls_patch.py",
    "exp/vendor_baselines/asynctls_vendor/identify_channel.py",
):
    print(p, "present" if Path(p).is_file() else "BLOCKED: absent")
PY
```

当前 RULER 入口存在，但 AsyncTLS adapter/channel依赖未闭合：
```bash
python -m benchmark.RULER.pred_ruler --model Qwen3-8B --model_path "$MODEL" \
  --task niah_single_1 --context_length 32768 --method asynctls_vendor \
  --data-root "$DATA" --output-dir "$OUT" --t "$RUN" --max-num 100 \
  --asynctls_channel_file "$CHANNEL" --asynctls_token_budget 1024 \
  --asynctls_block_size 64 --asynctls_top_kb 128
```
这只是 N候选 K1024入口，不宣称已落实N保护/层策略。P历史RULER使用K512，必须另标签。
64/128K须三方同一已核YaRN与截断方案，不擅自延伸以上32K命令。

LongBench现有入口：
```bash
python -m benchmark.LongBench.pred --model Qwen3-8B --model_path "$MODEL" \
  --task hotpotqa --method asynctls_vendor --t "$RUN" \
  --config-path benchmark/LongBench/config --dataset-path "$DATA" --output-dir "$OUT" \
  --asynctls_channel_file "$CHANNEL" --asynctls_token_budget 1024 \
  --asynctls_block_size 64 --asynctls_top_kb 128
```
校准生成器缺失，无可给出的真实校准运行命令；不能虚构 `--offload` 或 AsyncTLS Graph CLI。

现有 B1 Graph诊断入口（只none/tli）：
```bash
TLI_SPARSE_PREFILL=0 python exp/bench/e152v4_graph_latency.py \
  --method none --model-path "$MODEL" -L 32768 --graph-mode graph \
  --max-gen 128 --warmup 1 --repeats 3 --out "$OUT/dense_graph.json"
```
eager用相同入口的 `--graph-mode eager`，也必须保留独立输出名。该producer将eager声明为仅对拍，不能把其输出直接当A4公平eager延迟；完整三方计时仍待实现。
我方参数须由冻结manifest映射现有真实CLI逐项核对，不在这里给未验完整拼接命令。
这个producer的TPOT排除了fill_static/capture；不是A4完整请求harness，不能直接标A4通过。
AsyncTLS三方Graph、同输入真实wall instrumentation、preflight fail-before-load、预算observer、
配对CI消费者均列为实现缺口，不因文档存在而宣称就绪。

## 5. 冻结数据与输出契约（设计 schema，尚非现有producer已实现字段）

每行 JSONL：`schema=asynctls-comparison-v1`、run_id、stage、status(pass/fail/blocked/not_supported)、
source_commit、dirty_patch_sha256、producer_path/hash、actual_imported_module_paths/hashes、
upstream_repo/commit/license或style_identity、model/config/tokenizer hashes、software/hardware、
method、P_or_N、offload、resolved_manifest/hash、calibration_ids/hash、seed。

输入身份：dataset/revision、ordered_source_ids、template hash、post-truncation/BOS完整token hash、
实际prompt_length、generation参数（max_new_tokens、EOS/min_length、greedy/temperature/top-p）、
scorer/hash、prediction/hash；相同sample_id不替代token相等。质量与性能输入集各自标识。
公开结果只发布仓库相对路径、非敏感配置和hash；宿主绝对路径、输入原文/受限ID、dirty补丁正文和个人反馈留受限证据，不随设计PR公开。
校准/开发/确认集的文档组不相交；对可能共享源文档的数据按document cluster抽样。

预算：per-layer/KV-group `visible_count, protected_unique, selected_unique, union_unique,
K_requested, K_realized, page_size, coarse_requested, coarse_realized, boundary_extra_pages,
candidate_unique_tokens, dense_layer, fallback_reason`。平均值不得掩盖越预算行。
FullKV的K字段用all-visible语义，不伪装固定K。

Graph：`requested, supported, enabled, capture_count, replay_count, recapture_reason,
bucket, input_update_verified, fallback_count, fallback_reason`，加实际trace/event证据路径hash。
Eager明确capture/replay=0；unknown不能写false充当验证。

时间单位ms：单独记录load/tokenization/H2D（主wall之外并披露）、prefill/index_build、
fill_static、warmup、capture、recapture、每步input_update/metadata_update/selector/topk/gather/
attention/host_wait/sync，以及first_output/last_output与外层同步wall。
完整请求wall从输入resident且同步后到末token/同步，**包括该请求发生的所有index/cache准备与维护**。
冷请求含首次capture；warm steady态只在明确预先存在可复用cache/graph时摊销，并报告摊销假设。
阶段CUDA events可能重叠，不能相加冒充wall。TPOT分母为首token后实际有效生成token数；
batch throughput为全批有效输出token/批wall，不计死亡行padding。另报请求数/s、TTFT、TPOT p50/p95、
peak allocated/reserved GiB、host pinned GiB、OOM和共租状态。

质量统一到0–100分但保留原metric和分母；报告每任务N/缺失/均值/delta、宏均值、
最差任务delta、最差样本组、paired CI。时间重复以独立进程/请求为单位，不能把127个相邻step
当127个独立模型实验。配对进程顺序随机/交错；每进程5次热身、至少3个独立进程且各3次测量为**候选协议**，
依预算冻结后运行，不把已有2进程记录说成满足。

A6候选预注册：delta=method−FullKV，宏均值非劣margin0.3分、逐任务1.0分；
单侧95%配对document-cluster下界必须 **严格大于** −margin，等于为未通过；
逐任务用同时区间或Holm控制family-wise错误。此为待采用设计阈值，不冒称用户/论文既定标准，
不得看确认集结果后改阈值。独立确认集不足时报告探索结果，不宣布无损。
缺失预测/超时/OOM不做pairwise deletion美化均值：完整性门禁失败，报告原分母、缺失数与失败原因，补齐前不出确认性非劣结论。
跨任务共享源文档用全局文档group联合重采样，任务权重固定；无法构建group则报告聚类身份缺口，不把逐行bootstrap称document-cluster CI。
样本数需在冻结的开发集方差下确定功效；固定100条不是自动充分。
“优于dense”另要求预注册检验与CI，不从正均值直接推断。

## 6. 历史记录复用范围与隔离

- `exp/trace/results/asynctls_r32k_formal.{json,md}`：
  现有11任务、每任务100条的K512/b64/kb128语义复现记录，AVG50.09；
  不是论文RULER10任务表，也不是本次K1024规范化三方精度。manifest/receipt可作身份追溯起点，
  原预测/adapter来源与实际导入字节仍需绑定；不删除、不改标、不将其均值当本轮结果。
- E152v4 `exp/trace/results/e152v4_formal_pc40/`：
  32/64/128K作者归档TPOT可CPU复算为1.064/1.448/2.067倍。
  六份dense文件是v2/DynamicCache记录，TLI是v4 Graph；提交证据**不支持graph-vs-graph**。
  不据此断言宿主没有隐藏Graph，也不替宿主补证。运行commit字段为404feb17cf14d536dec943aa433d6d21d610d172；
  新main的producer不自动等于当时实际运行模块。B1及TPOT排除fill/capture边界必须保留。
- 当前T6质量按其自身manifest复用，只有model/config/input/prediction/scorer全部闭合才能与本设计配对；
  当前方法设计选择不把旧nt448/比例配置质量改标为pooled。
- 旧32/64/128K质量、MLA诊断、mass及单kernel微基准保留各自证据范围，不能填补新的三方完整请求表。

## 7. 完成定义与发布

本设计PR验收：路径/参数静态核对、独立科学与工程审查、Markdown/代码块检查；
不意味着任一GPU阶段通过。执行结果须由独立验证者核原始artifact/源码/配置/错误行，
状态仅为usable-with-scope。失败隔离按run_id保留，不覆盖旧结果，不停止未绑定的他人任务。
只向Chosen-David修复/设计分支提交draft PR，不合main。
相关建议：[advice](../../agent_doc/advice/2026-10-23_asynctls_comparison_design_by_gpt.md)。

