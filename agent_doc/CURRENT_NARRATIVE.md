# CURRENT_NARRATIVE — 当前叙事锚点（防误读单一事实源）

> **本文件是仓库叙事的唯一现行锚点**（2026-10-23 建立）。所有结论只允许引用
> `exp/trace/results/` 下的判决 JSON 文件名，不复制过时报告的数字。
> `agent_doc/advice/archive/` 内的报告已被后续判决推翻或已被吸收，**不得引用**。
> 每次新判决落库时更新第 5 节索引（写入收官流程）。

## 1. 生产口径（champion geometry）与参数陷阱

主方法 = TLI（论文显示名 LoDA-SA）两级索引 decode 稀疏注意力。生产口径的
**全部显式 flags**（任何主表/判决/复现脚本必须逐字传齐）：

```
--method tli
--tli_far_method minmax --tli_near_method avg
--tli_near_pages 32 --tli_near_tokens off --tli_near_len 4096
--tli_subspace tail --tia_level2_cmp_ratio 4
--tli_sink_tokens 4 --tli_swa_size 128
--tia_level1_topk 64 --tia_level2_topk 1024
```

几何口径 = **nb32 / nt-off（L2 自由竞争）/ nl4096**，tail32 子空间，
minmax+avg 粗筛，sink4+swa128 正交强制区，K1=64 块 / K2=1024 token。
依据判决：`e155d_grid_verdict.json`（网格选举，冠军 nb32ntoffnl4096）+
`e152v4_formal_verdict.json`（K1=128 时代主表）+ `e140b_k1_downscan_verdict.json`
+ `e140h_k64sweet_verdict.json`（K1=64 甜点）+ `e168_k64_e2e_verdict.json` /
`e169_k64_llama_verdict.json`（K1=64 双模型族 e2e；2026-10-25 起生产口径
champion flags = K1=64）。

**参数陷阱表**（历史教训，2026-10-23 开源净化手术已结构性消除大部分）：

| 陷阱 | 历史行为 | 现状 |
|---|---|
| TLI 的 batch 能力只在 attention 消费端，indexer 必须逐行执行 | PR #7 的 `TL-BATCH-INDEXER-PARALLEL-20261026` 增加完整 batch 候选实现；`docs/tli_batch_indexer.md` 与 `agent_doc/results/tli_batch_parallel_20261026/` 记录 CPU/解释器/离线编译，目标 GPU、旧 Graph 对照和完整模型质量仍待验，默认仍走 legacy。此项不修改历史 GPU 主表，也不代表 compact KV 淘汰已支持 B>1。 后续五轮见 `docs/tli_batch_five_rounds.md`，包含按需摘要、Q 复用及可选 movement 融合；本地证据不构成 GPU 提速判决。 |---|
| `--tli_enable_layer_skip` 默认 true | 漏传 = 静默跑成 D′ gate 口径（E129 起协议强制显式传） | **接口保留**（2026-10-25 用户指令：far-skip 转向 KV cache 节省叙事线，见第 5 节）；主表口径仍显式 false |
| `--tli_subspace` 默认 full | 生产必须显式 tail | 默认仍 full，**必须显式传 tail** |
| `--tia_level2_cmp_ratio` 默认 2 | 生产口径 4 | **必须显式传 4** |
| α/β/γ 比例接口（E64 时代） | 与显式几何混配事故面 | **CLI 接口已删除**（冠军几何运行时内部状态 α=β=1.0、γ=None，由 nb/nt/nl 归一推导） |
| kmeans/ccluster/sim_greedy 分支（E110） | `--tli_enable_kmeans` 默认 true | **已删除**（冠军口径 4bit 细筛，E4c 判决无一致优势） |

不传任何几何 flags 的 legacy 缺省 = 单池老逻辑（非主表口径），仅作向后
兼容保留；主表/论文数字一律以冠军 flags 口径为准。

## 2. 主表数字（e2e decode 加速比，TLI-graph vs dense-eager）

统一口径：整步 CUDA graph 化 TLI decode vs dense eager decode，
TPOT 中位数，bit-exact 门禁（graph vs eager gen_tokens 逐位一致）。

> **【2026-10-27 刷新】下表已从 K1=128/iter2 口径切换为 K1=64/iter3 口径**
>（E172/E173/E174 三模型族判决）。iter22 旧口径段落保留在文档下方作历史参考。

| 模型 | 32K | 64K | 128K | 判决文件 |
|---|---|---|---|---|
| Qwen3-8B | **1.451×** | **2.216×** | **3.636×** | `e172_iter3_k64_qwen_verdict.json` |
| Llama-3.1-8B | **1.428×** | **2.192×** | **3.643×** | `e173_iter3_k64_llama_verdict.json` |
| Qwen3-32B | **1.338×** | **1.924×** | **3.015×** | `e174_iter3_k64_32b_verdict.json` |

- **加速比随上下文单调增长**：三档 1.34-1.45× / 1.92-2.22× / 3.02-3.64×。
- **context 退化压缩**：dense 32K→128K +161~165% vs TLI +3.3~4.2%。
- **iter3 vs iter22 增量**：K1=64（~3-5%）+ iter3 融合链（~24-40% TPOT 下降）。
- **完整 e2e（含 prefill）**（`e178_full_e2e_with_prefill_verdict.json`）：
  TPOT 1.28/2.39/3.69× + E2E（含 prefill）1.17/1.23/1.49×（32/64/128K）；
  prefill 两臂一致（原生 SDPA dense）。
- 32B@128K 行协议披露（见 `e160b_128k_expandable_verdict.json`）：
  ①环境 expandable_segments:True（32/64K 行为默认分配器）；②A 臂
  4×单 run 进程级 repeat（graph 私有显存池跨 run 残留，显存贴满时
  同进程多 prefill 不可行）；峰值 141.7G = 容量 98.6%。
- 32B 行跨机复现：`e160r_32b_replication_verdict.json`（pc31 复刻，
  加速比 dev +0.10%/+0.22%）。
- **graph-vs-graph 口径警告**（`e159_llama_e2e_latency_verdict.json`
  附注）：整步 capture 下 dense SDPA 走 4D additive-mask 慢路径，
  graph-vs-graph 数字（如 Llama 4.97/8.64/14.28×）是「graph 降级
  dense 基线」口径，只能作 graph-friendly 叙事点，**不得进主表**；
  主表一律 TLI-graph vs dense-eager。
- 质量主表：RULER 32K T6 champion 59.02 vs FullKV 59.38（保留率
  99.4%）等长度梯度见 `e147c_series_verdict.json` /
  `e147h_*_verdict.json` 族；LongBench 数字见 `e145d_lbv2_verdict.json`
  / `e145c_*_verdict.json` 族。
- **【已合入 2026-10-25】iter2+iter3 优化链晋升主树**（3 文件 delta：
  `indexer/tli_indexer.py` md5 854f2567 / `ops/eager_decoding.py` f23b7003 /
  `patches/llama3_attn_patch.py` a79dbfa4，源自 pc40 `~/tli_ako/solution_iter3`
  影子包，与 e165-e167b 系判决逐位同源；晋升后 E127 门禁 15/15 PASS）。
  判决链：`e165_kernel_census_verdict.json` + `iter2a/2b/2c_verdict.json` +
  `e166a/b/e/d2_verdict.json`。**iter3@K1=128 TPOT 18.675/18.888/19.388ms，
  加速比 1.407/2.100/3.458×**（e2e bitexact 三长度 + pc02 跨机 ≤0.17% +
  RULER 质量门 md5 逐字节一致三版同源闭合）。
- **【已收官 2026-10-25】E170 创新点收益独立验证**（exp/trace/results/e170_innovation_verification/）：
  代码是否吃到收益的独立答案 = **是**。
  (a) 子空间（tail32）：microbench 隔离 L1 amin/amax **3.38×**，紧凑 buffer
  **4× 节省**（64 MB / layer @ 128K），decode K2 候选消除 **15.6×** 全宽工作量，
  decode step KV traffic 仅 full 的 **25%**；
  (b) far skip（D' gate）：选中 KV 流量仅 dense 的 **4%**（128K），紧凑显存
  dense 的 **4%**，far 段占选中流量 **19.5%**（三长度恒定，e163 一致）；
  (c) op 归因：kernel count 与 e166g 完全对齐（launch=15132、memset=1314），
  elementwise 弥散群无单一热点；
  (d) bitexact：三长度 md5 一致（d8c8649...）+ TPOT 平坦化 18.73/19.06/19.39ms；
  (e) GPT runtime-review 补丁：`input_signature` 门控 + `fb` 修正 + 4×
  `new_zeros→new_empty` → 已全部在代码面（可追溯至 iter2 合入前的 e160+ 构建）。
  verdict 文件：e170{bitexact,microbench,farskip,opattr}_*.json；
  报告：agent_doc/advice/e170_subspace-farskip-verification/。
- **【已收官 2026-10-25】E168/E169：K1=64 iter22 代码面双模型族判决**：
  Qwen3-8B `e168_k64_e2e_verdict.json`（1.050/1.467/2.084×，K1=128 复验格
  24.697ms 逐位复现，bitexact 2/3——64K step-55 bf16 tie-break 单点分歧披露）；
  Llama-3.1-8B `e169_k64_llama_verdict.json`（1.011/1.431/1.968×，bitexact 3/3）。
  双模型族 TLI 臂 -0.45~-1.08% vs K1=128，K1=64 口径与 e140b/e140h 质量判决闭合。
  **【已收官 2026-10-25】iter3+K1=64 主表三模型族判决全齐**（论文最终主表）：
  **E172 Qwen3-8B**（`e172_iter3_k64_qwen_verdict.json`，pc40 GPU0）=
  **1.451/2.216/3.636×**（17.715/17.957/18.464 vs dense 25.712/39.797/
  67.139ms）；K1=128 复验格 18.668ms 逐位复现 iter3 18.675ms。**E173
  Llama-3.1-8B**（`e173_iter3_k64_llama_verdict.json`，pc40 GPU1）=
  **1.428/2.192/3.643×**（14.971/15.188/15.590 vs 21.385/33.287/56.801ms），
  跨模型族对称 ≤0.024/档。**E174 Qwen3-32B**（`e174_iter3_k64_32b_verdict.json`，
  pc02 GPU0，128K 行走 E160b 同款单 run/进程救援协议）= **1.338/1.924/3.015×**
  （41.295/41.774/42.670 vs dense 55.265/80.356/128.670ms），**十项词元身份门
  全部 bit-exact**（含 k128 复验臂逐位）；k128 复验臂 32K 43.28 / 128K 44.58
  均慢于 K1=64 → K1 减半三模型一致正向。取代 iter22 口径（8B 1.064/1.448/
  2.067×、Llama 1.015/1.41/1.961×、32B 1.015/1.338/1.841×）。
- **iter3 机制判决链**（构成：掩码装配融合 `e166a_iter3a_verdict.json` +
  候选压缩融合 `e166b_iter3b_verdict.json` + compact launch 调优
  `e166e_compact_cfg_verdict.json` + 摘要维护/L1 双池打分/候选装配三
  kernel `e166d2_iter3d_verdict.json`，关键坑：生产 KV 为 head-major
  置换视图，门控须查 stride(2)==1）：
  双口径闭合 `e166h_census_iter3d_verdict.json` busy 22.27→19.34ms
  （copy −1.74/other −1.19/gemm 贴地板不动），busy/TPOT=99.7%；
  op 级差分 `e166g_opattr_verdict.json`（sort 全家含 6 cub RadixSort
  整体消失，launch −5004）。残余 = topk 家族 ~3.3ms（质量门口径留
  用户）+ copy/elementwise 弥散群；**bitexact 兼容杠杆已收割完毕**。
  **长度平坦化机制曲线三点闭合**
  （`e166i_census_32k_flatness_verdict.json` + `e166j_census_64k_flatness_verdict.json`
  + `e166h`）：32/64/128K busy 18.674/18.866/19.340ms，gemm/attn/copy
  三类 L 无关成本三点全平直，唯一增长段摘要-索引维护近线性
  （+0.7ms@4×）；e2e-census 双口径逐段闭合（busy/TPOT
  99.9/99.9/99.7%）。topk 内部 sort 三长度恒定（~0.85ms，与被消除的
  候选域 cub sort 家族不同）。**批档包络**（`e167_bs_scaling_verdict.json`，
  exp/bench/e167_bs_latency.py，2026-10-25）：B=1 复现主表口径（1.40×/
  3.48×，偏差 <1%）；128K 批档 tli 领先至可行上限 B=2（82.6 vs
  120.8ms=1.46×，dense 被 KV 流量主导批摊销失效）；32K B≥4 dense 反超
  （GEMM 批摊销 1.31-1.49×）；TLI 批路径=行串行（吞吐不增 53.8→44.5
  tok/s，跨行融合列后续工程）；显存包络 tli 静态池宽于 dense（B=16@32K
  tli 可跑 vs dense OOM）；bs=32 全长度不可行。serving 叙事限定 bs=1
  decode 与长上下文批档，32K 批档如实披露。**64K 补点三点闭合**
  （`e167b_bs_64k_crossover_verdict.json`）：交叉点随长度单调右移
  （32K:B∈(1,4) → 64K:B≈4 过渡带 0.96× → 128K:可行区间内不交叉）；
  B=1 复现主表 2.12×（偏差 <1%）；机制=dense 批摊销被 KV 流量稀释
  vs tli 每行成本平坦（平坦化机制的批维度红利）；tli 行串行代价随 L
  增大（每行 +19%→+66%→+114%），跨行融合工程动机数据。
- **【iter3 已合入·跨模型族】E168 Llama iter3 复制**
  （`e168_llama_iter3_verdict.json`，pc02 四卡三臂）：iter3 融合链在
  Llama-3.1-8B 完整复现——bitexact 三长度 PASS、TPOT 增益
  −25.2/−31.4/−43.3%（vs Qwen3 −24.3/−31.1/−40.2%）、加速比
  1.353/2.073/3.456×（vs Qwen3 1.407/2.100/3.458×）、长度平坦化
  +4.0%@4×ctx（vs Qwen3 +3.8%）；e159 主表口径跨机复现偏差 <0.5%。
  融合链与平坦化机制获跨模型族（Qwen3/Llama3）泛化证据。
  **e168b 补 census 机制分解**（`e168b_llama_census_verdict.json`）：
  Llama iter3 busy 15.874/16.069/16.463ms（+3.7%@4×），gemm 4.36-4.39/
  attn 0.735/copy 5.2-5.3 三类平直，other 5.48→6.06 唯一增长——与
  Qwen3 e166i/j/h 四类结构逐类同型，平坦化机制证据升至 kernel 类级
  双模型族口径。**e168c Llama RULER 质量门 PASS**
  （`e168c_llama_quality_gate_ruler.json`）：120 样本 × 3 任务
  （niah_single_1/niah_multikey_1/vt）orig vs iter3 双臂输出 md5 逐字节
  一致——Llama 证据链三件齐（e2e bitexact + census + 质量门）。口径注：
  本系列为 K1=128（历史主表口径）；K1=64 生产口径由 e171/e172-e174
  系列另行闭合。**e168d Llama 批档包络（K1=64 生产口径）**
  （`e168d_llama_bs_verdict.json`）：交叉点随长度单调右移跨模型成立
  （Llama 32K:B≈2 → 64K:B≈4 → 128K:不交叉；与 Qwen 形态同构）；
  B1 1.461/2.234/3.741×、平坦化 +4.3%@4× 保持；行串行代价随 L 增大
  （每行 +17%→+55%→+131%）同 Qwen 形态；128K B4 双臂 OOM 同 Qwen。
  批维度『长度轴是分离轴』双模型族口径；Llama 32K B2 即反超（比
  Qwen 早），论文口径 tli 领先区间 Llama = 128K×B≤2、64K×B≤2。
  **e168e Qwen3-8B K1=64 census 平坦化**（`e168e_qwen_k64_census_verdict.json`，
  pc40 主树 7bc1788）：busy 17.719/17.939/18.380ms（+3.7%@4×），三类
  平直 + other 唯一增长（+0.70ms@4×），与 K1=128 口径结构完全同型；
  K1=64 额外收益 = other 类一致降 ~0.95ms（L1 块数减半）；busy 与
  E172 TPOT 18.46ms@128K 自洽（99.6%）。主表切 K1=64 后平坦化主张
  证据链无缝衔接。**e168f Qwen3-8B K1=64 批档包络**
  （`e168f_qwen_k64_bs_verdict.json`，pc40 GPU1）：B1 1.470/2.206/
  3.644×（与 E172 主表 + e168e census 三口径闭合）；交叉点 32K:B≈2 →
  64K:B≈4 → 128K:不交叉，与 Llama e168d 三点完全对齐——旧口径
  （e167）下 Qwen 32K B∈(1,4) 与 Llama B≈2 的模型族差异随口径统一
  消失，批档交叉点形态双模型族统一为『随长度单调右移』；行串行
  +14%→+50%→+120% 跨口径跨模型同形态；128K B4 双臂 OOM。论文批档
  叙事建议统一用新口径双模型族数据（e168f+e168d）。
  **e168i Llama-3.1-8B K1=64 census 平坦化**
  （`e168i_llama_k64_census_verdict.json`，pc40 GPU1，worktree@7bc1788
  与 e175 工作分支隔离）：busy 15.022/15.223/15.795ms（+5.1%@4×），
  三类平直 + other 唯一增长（+0.59ms@4×），逐类同型于 e168b（K1=128）
  与 e168e（Qwen K1=64）；K1=64 额外收益 = other 类降 0.67-0.85ms
  （L1 减半）；first_tok/gen5 与 e168b 逐 token 一致；busy@128K 与
  E173 TPOT 15.590ms 自洽（CUPTI 微膨胀 ≤1.5%）。**新口径平坦化
  证据链双模型族闭合（e168e + e168i）**。注：首轮 GPU0 采集被并行
  e175 任务抢占污染（gemm 非单调）已废弃，正式数字为 GPU1 独占重跑。
  **e168j Qwen3-32B K1=64 census 平坦化**
  （`e168j_32b_k64_census_verdict.json`，pc02 GPU1，iter3 shadow +
  expandable_segments + 单 run 单进程）：busy 41.637/41.980/42.862ms
  （+2.9%@4×），gemm 17.31/17.30/17.37、attn 1.42×3、copy 12.0×3 三类
  平直，other 10.86→12.03 唯一增长（+1.17ms@4×）；与 E174 TPOT
  41.295/41.774/42.670ms 双口径闭合（CUPTI ≤0.8%）；跨模型结构：gemm
  随参数量 ≈4×（vs 8B）、other 随层数近线性。**新口径平坦化证据链
  三模型族闭合（e168e + e168i + e168j），论文平坦化主张可升级为
  三模型族口径**。
  **e168g Qwen3-32B K1=64 批档包络**
  （`e168g_32b_bs_verdict.json`，pc02，repo@7bc1788，B∈{1,2}×
  三长度）：B1 1.340/1.925/3.023×（与 E174 主表 <0.3% 闭合）；32K
  交叉点 B≈2（0.969×，与 8B 双模型族 0.985/0.956 三点对齐）、64K
  B2 tli 领先 1.205×、128K B2 双臂 OOM（2×68.7GB KV 物理上限）。
  32B 特有：可行批档包络随 L 收缩。行串行 32K −0.5%/64K +26.9%
  （形态与 8B 一致）。**批档叙事三模型族口径收官（e168f+e168d+
  e168g）**。协议披露：①e167 采集器 tli 臂 prefill 漏 logits_to_keep=1
  （E167 坑只修 dense），32B B2×64K 19GB logits OOM——补丁已同步
  pc02+本地仓库（bit-neutral）；②tli 128K B1 复现 E160b 根因（同进程
  二次 prefill+graph 池残留），单 run 单进程 ×3 = 42.56/42.57/42.57ms；
  ③dense 128K B1 同进程 4 runs PASS（DynamicCache 无池残留）。
- **【已合入 perf/gpt-kv-eviction·未合 main 2026-10-25】E175：codex far-skip-kv-eviction
  物理蒸发 + cyclic compact cache 评测**：ring buffer 几何区宽恒定
  （sink4+near4096+window128+block64−1=4551），evict 触发后每层物理 KV
  恒 ~8.62 MB（与序列长度解耦）；单层 synthetic bench 8K→128K 节省
  46.6/86.6/93.3/96.8%（`far_skip_decode_single_layer.json`）；全模型
  eager 13/36 层静态 sk4 mask → footprint 32K −1.01 / 64K −2.24 / 128K
  −4.70 GiB = 3.4/5.1/6.5%（`e175_m1_footprint_verdict.json`）。**不合
  main**：eager-only（CUDA Graph capturing 显式 raise）/ B=1 / 非 4bit
  bitwise 等价 / RULER 质量门 M2 未跑。定位 supplement S8 implementation
  ablation。报告+LaTeX 段：`agent_doc/advice/e175_gpt_far_skip_eviction/`。
  代码 324073d 合入 `perf/gpt-kv-eviction` 作实验记录（分支保留）。

- **E176 batch indexer 并行化评测（eager 路径，128K）**：legacy vs
  `TLI_BATCH_INDEXER=parallel` 对比——B=1 125.33 vs 125.38ms
  (+0.04%)，B=2 196.15 vs 195.93ms (−0.11%)，B=4 371.35 vs 370.98ms
  (−0.10%) → **全部在噪声带内，零收益**。根因：eager 路径 indexer 开销
  (~3.3ms topk + ~8.5ms copy) 在总 decode 125ms 中占比已小，batch
  Triton kernel 并行优势被 Python 逐行循环开销淹没。代码已合入 main
  （opt-in `TLI_BATCH_INDEXER` env 开关），但**不推荐作为性能优化手段**。
  载体：`sparse_attn/indexer/tli_batch*.py` + `qwen3_attn_patch.py` 路由。

- **【已合入 2026-10-27】E177：HybridCache request-local hybrid batch**
  （`codex/far-skip-hybrid-cache-20261026` cherry-pick 2 commits + test
  修复，fast-forward merge main b0a378c）：
  `hybrid_batch_cache.py` + `hybrid_batch_runner.py` 提供按请求×层
  ring/linear 混合策略，紧凑层不做全宽分配。
  **KV 节省**（`e177_hybrid_cache_kv_savings_verdict.json`）：128K 70%
  skip 层 **67.2% 节省**（18→5.9 GiB），三模型族闭合（8B/Llama/32B ≈67%）。

- **【已落库 2026-10-27】E178：完整 e2e（prefill+decode）性能**
  （`e178_full_e2e_with_prefill_verdict.json`，pc33 双卡 Qwen3-8B）：
  TPOT 加速 1.28/2.39/3.69×（32/64/128K）；E2E 加速（含 prefill）
  1.17/1.23/1.49×；prefill 两臂一致（原生 SDPA dense）。
  TLI decode 长度平坦化: +29.5%@4×ctx vs dense +273%。

## 3. 对比口径（vendor 基线）

- 对比基线（Quest/MInference/SparQ/MoBA/ClusterKV/AsyncTLS/HISA/JSSA/
  FASA）全部经 `sparse_attn/patches/` + `exp/vendor_baselines/` 统一
  harness 跑批，数字只与判决 JSON 对齐，**不引用 advice 报告**。
- **AsyncTLS 三口径拆解**（无官方开源仓库，C 轨语义复现，实现头注见
  `exp/vendor_baselines/asynctls_vendor/`）：kernel 级 1.7-6.2× /
  e2e 2.3×@96K / offload-throughput 1.84×/4.70×（各自论文口径下）。
  复现锚点：`asynctls_r32k_reverify_verdict.json`。
- 主表口径下 vendor 对齐原则：token 预算统一 K2=1024（quest/sparq/
  fasa/moba/jssa budget-normalized，见 `benchmark/` 各 runner 头注）。
- FASA RULER 复现的崩溃诊断与口径（11 任务 × n=100）见
  `e119_ruler32768_formal_fasa_20261021.json` 族（重跑判决落库后更新）。

## 4. MLA 线（DeepSeek-V2-Lite，子空间消融链）

叙事链五判决，全部在库：

1. `e158_mla_subspace_verdict.json`（trace）：rope 低频主导
   （lowfreq32 0.865 ≫ highfreq32 0.258）；仅 nope 不行（0.672）；
   最优 = rope64+nope32（96 维）0.905，保持 full192 的 96.4%。
2. `e158b_mla_rope_nope32_e2e_verdict.json`（96 维 e2e 兑现）：
   三臂统计不可分（配对 bootstrap CI 跨零）——**trace→e2e 饱和映射**：
   两级预算 K1=64/K2=1024 冗余吸收子空间增益。
3. `e158c_64dim_combo_verdict.json`（64 维 trace）：lofreq32+nope32
   0.905-0.908 与 96 维持平 = 免费降维。
4. `e158e_mla_subspace_perf_verdict.json`（性能三口径）：64 维
   indexer 税 -17.6%（vs full192）；但 eager 下稀疏臂全线慢于 dense
   2.3-2.5×（CPU dispatch 税 50-60ms/步 = TPOT 40-45%）——**MLA 性能
   主张必须靠 graph 化**，DSV2 graph port（DSV2StaticCache + vendored
   rope 张量化 7 文件级改造）列 follow-up。

**结论：MLA 生产口径 = rope-only 64 维**（`e142a_mla_e2e_verdict.json`：
rope-only 与 full192 e2e 持平）；96/64 维组合降级为消融臂。
MLA D=192 不满足增量缓存门（k.shape[-1]==128）→ prepare 全量重建
（inc=0），系生产 eager 路径真实行为。

5. **E158h 四臂补全收官**（`e158h_mla_nope_e2e_verdict.json`，2026-10-25）：
   nope64（24.53/12.46）/nope32（24.93/12.28）名义反超 fullkv 锚
   （23.91/11.48），但多重比较校正后全臂与 fullkv 统计不可分
   （nope32 池化 p=0.049 名义显著 / lowfreq32 池化 -1.20 p=0.027 显著
   不足）——E158 trace 层「仅 nope 不行（0.672）」在 e2e 不成立，
   **饱和映射三证据链闭合**（96 维 E158b / 64 维 nope / 32 维 nope）；
   lowfreq32 显著掉分背书 E158g「生产维持 rope64」。

## 5. 已被推翻叙事索引（防误读核心——旧结论 → 推翻它的判决）

**2026-10-27 统一审查入口**：[12 PR来源与冲突清单](results/pr-unification-20261027/validation.md)。保留main与既有融合；#8稿件/#12耦合运行时未强行覆盖。LongBench full/split明确分离，旧质量不可直接继承；无新增GPU收益结论。
**2026-10-27 实现审查补充（非质量/性能新判决）**：
[五轮运行链路审查](results/runtime-audit-20261027/README.md) 修复 batch
live length、请求状态/错误回执与长 prompt logits 问题，并将 LongBench
完整 prompt prefill 与历史 split 协议显式隔离。新协议、compact/evict
与 legacy 的选择范围差异均需独立质量复验；本次 CPU 回归不继承旧表结论，
CUDA 必跑门禁仍待有 GPU 环境验收。


**2026-10-27 子空间 padding 实现补充（待 GPU 验证）**：
[证据](results/subspace-pad-20261027/validation.md)：先选维再按原边界补零，
消除统计前的全维 padding 临时量；MoBA 保留全维路径。201 CPU通过、
24 CUDA跳过、30解释器对拍。CPU容量/128K公式不代表 GPU峰值或 KV节省；
Qwen为初始化收益候选，MLA eager仍每步全量重建，实际收益待配对。

2026-10-27 几何共享复核：main/PR10 原九条目全局缓存导致 B>=10 hybrid eager 跨层复用失效；PR10 改为请求级有界缓存，B32 两层实际构建64→32次/步，56 CPU通过/15 CUDA跳过、配对logits逐位一致。保留更多大batch几何条目是内存/重算取舍；capture仍逐层构建，不能宣称Graph或GPU实测加速。见 `agent_doc/results/geometry-cache-20261027/validation.md`。

2026-10-27 输出组装复核：B1 hybrid decode 的唯一attention输出仍经cat复制；PR #10 直接复用输出，B>1保持原路径。16 CPU通过/8 CUDA跳过，未测GPU速度；EOS行selector/attention仍执行。见 `agent_doc/results/single-output-20261027/validation.md`。

2026-10-27 prefill 选维复核：static_pair 可选路径此前将全长 Q 转FP32后只用尾256行；PR #10 改为先切片再转换，精确选维及请求重置CPU通过。61 CPU/30 CUDA跳过；仅消除标定临时转换，不代表decode加速或KV节省，动态gate展开尚未优化。见 `agent_doc/results/prefill-tail-20261027/validation.md`。

2026-10-27 workspace 容量复核：此前 linear gather 的4.52MiB是有效数据量，几何扩容后的底层缓冲实际8MiB；PR #10 对固定形状 cat/scratch 改为按需分配，CPU存储配对8→4.515625MiB，GPU allocator/峰值/速度未测。见 `agent_doc/results/workspace-capacity-20261027/validation.md`。

2026-10-27 runner 输入维护复核：main hybrid inject 在 Graph 外每步重建全宽 causal mask；PR #10 根据既有 append-only/EOS 约束改为首步完整初始化、后续每行单槽更新，保留 dense 层所需形状和地址。19 CPU通过、12 CUDA跳过；无 GPU 提速结论。见 `agent_doc/results/incremental-mask-20261027/validation.md`。

2026-10-27 子空间维护复核：main17e56e8 compact ring 的未量化投影仍先 gather 再 copy；PR #10 增加可选 GPU 直写原投影缓存，覆盖 eager/Graph，不增 scratch。量化分支和 linear iter3 不变；CPU/解释器及离线编译通过，无 GPU 加速、显存或质量判决。见 `agent_doc/results/projection-write-20261027/validation.md`。

2026-10-27 linear gather 复核：hybrid linear 消费端仍把全宽 bool mask 转 int32/topk，暂不改选集协议；PR #10 复用现有 GPU pair-gather 合并 K/V 读取与无效槽清零，保持原顺序。CPU/解释器已验，无新增 GPU 加速/显存判决；逐行 scratch 驻留成本须计入。见 `agent_doc/results/linear-gather-20261027/validation.md`。

2026-10-27 main b0a378c 实质复核：hybrid index/attention 仍逐行；PR #10 补充可选 GPU KV 写入融合，保留逐请求冻结 gate/EOS。E177 脚本仅计算容量公式，67.2% 基于假设 25/36 层 ring，不是 CUDA allocator/真实 gate/质量实测。ring 几何为 4291 槽；旧 TPOT 数据不代表本 hybrid 实现已验收。当前写入回归和未完成候选见 `agent_doc/results/hybrid-write-20261027/validation.md`；无新增 GPU 提速或显存实测判决。


2026-10-26 路由复核（main 55c38d8）：长度缓存不更新、parallel 提前返回绕过 Graph、路由记录传模块而非层号，4 个 host 反例已复现；PR #10 保留正确生命周期并修复记录。上文 E176 数字保留为远端报告，当前树未找到 E176 原始脚本/回执，不能据此确认实际 parallel/Triton 命中或将零收益归因于 Python 循环。需以候选提交、完整 flags、路由和 profiler 重新配对验收；详见 agent_doc/results/parallel-route-20261026/validation.md。

2026-10-26 整合更新：main `2289900` 已包含旧版 compact 实现；整合分支把 PR #7 的融合路径与 PR #9 的逐请求×层混合缓存接通。多 batch/EOS/新请求重建已有 CPU 回归，混合 attention/index 仍逐行；不能声称全 batch 融合或新 GPU 提速/显存降幅。当前边界与远端验收见 `docs/unified_far_skip.md`，下列 E175/旧 PR 限制是历史版本记录。

| 过时结论（出现在 archived 报告/早期叙事中） | 推翻/修正判决 |
|---|---|
| 保护带（boundary bypass 前的 sink/swa 保护带几何）假设最优 | `e148_sinkswa_verdict.json`（否决） |
| prefill 稀疏可提速（FS/prefill sparse e2e） | `e149_sink4_verdict.json`（FS 质量 -5.18% → decode-only 叙事）；prefill 性能详见 `e139_prefill_profile_verdict.json` / `e140a_prefill_k1up_verdict.json` |
| K1 越大越好 | `e140b_k1_downscan_verdict.json` + `e140h_k64sweet_verdict.json`（倒 U 峰 K1=64）+ `e168_k64_e2e_verdict.json` / `e169_k64_llama_verdict.json`（K1=64 双模型族 e2e 成立，2026-10-25 生产口径切换 K1=64） |
| 32B@128K OOM 根因 = 显存碎片 | `e160b_128k_expandable_verdict.json`（半否定——真根因 = 进程内二次 prefill 与 graph 私有池残留共存；可行解 = 单 run per process） |
| layer_skip（D′ 静态门）/ dyn_dgate（E137 动态门）作为**延迟加速**主张 | 延迟视角已被 `e161_farskip_verdict.json` 否决（subspace 维度 decode 增益 ≈0，far-skip 臂 +2.1~+11.2ms/步）——**但 2026-10-25 用户指令叙事转向 KV cache 节省视角**：被跳层驻留退化为 sink4+near4096+swa128=4228 token 滚动缓冲（128K 下 3.2%），eps=0.05 实测层集 {L7,L8,L10,L11,L35}（5/36 层）→ KV 节省 ~13%；质量不可分实测见 `e137_dyn_dgate_verdict.json`（AVG5 Δ-0.03 CI 含 0）。接口**保留**，K1=64 口径复验在飞（E171）。**【E175 物理蒸发已合 perf/gpt-kv-eviction·未合 main】**codex far-skip-kv-eviction 实现物理 ring buffer 蒸发 + cyclic compact cache：几何区宽恒定（ring_capacity = sink+near+window+block−1 = 4551）与序列长度解耦；单层 evict KV 恒 ~8.62 MB（8K→128K 全长节省 46.6/86.6/93.3/96.8%）；全模型 13/36 层静态 sk4 mask 触发 → footprint 节省 32K −1.01 / 64K −2.24 / 128K −4.70 GiB（3.4/5.1/6.5%），verdict 见 `e175_m1_footprint_verdict.json` + `far_skip_decode_single_layer.json`。**显式约束**：(a) eager-only，CUDA Graph capturing 显式 raise（与 TLIStaticCache 路线正交）；(b) batch 必须 = 1，不支持 beam/offload/chunked prefill；(c) selector 非 bitwise 等价 4bit 生产 TLI（sink-boundary 排除 + tie-breaking 差异）；(d) **RULER 质量门 M2 未跑**，不得在质量未实测条件下声明"质量保持"；(e) 代码路径与 iter3 主树互斥（不合 main，留 `perf/gpt-kv-eviction` 作实验记录）。作为论文 supplement S8「implementation-level ablation」有价值，review + LaTeX 段见 `agent_doc/advice/e175_gpt_far_skip_eviction/`。补测 M3（同口径 TPOT — baseline 也跑 dyn_dgate 消除 skip 混淆）在进行 |
| kmeans/ccluster/sim_greedy 簇代表选择 | E4c negative result（严格预算下无一致优势，E4b 有整簇超选 bug）→ E110 系列保留作消融 → **已删除** |
| α/β/γ 比例化几何口径（E64 帕累托点叙事） | E155d 参数退役（用户 2026-10-19 拍板转向 nb/nt/nl 显式几何）；比例口径的全部历史数字仅作 E64-E155 谱系档案（`exp/trace_archives/probe_e5b_e98/`） |
| e152v2 dynamic-cache 性能口径（TLI 3.4×/2.5×/1.8× slower） | 已被 `e152v4_formal_verdict.json` 整步 graph 口径取代（论文图 E2 已 in-place 重写，commit e5071cb） |
| MLA 子空间增益可线性兑现到 e2e | `e158b_mla_rope_nope32_e2e_verdict.json` 饱和映射边界披露 |
| kernel fusion 已经消除 batch indexer 的所有 copy，融合越大越快 | `docs/tli_gpu_pipeline.md`：新增四类真实 Triton producer/consumer 融合，整行扫描曾占 255 registers/thread，分块后编译峰值 60；仍有 topk/softmax/full-mask/attention 流量。188 个 offline build 与解释器检查均不是实际 GPU 提速，远端与模型验收待执行。 |
| far mask 的13%账本已经等于GPU显存实省；当前ring每步搬迁全部历史 | `docs/tli_skip_noskip_fusion.md` 区分 main 默认全驻留与 PR compact：后者一次性压缩、每步单 token 写入，容量4291含页对齐。CPU底层存储/引用释放已检查；GPU allocator、严格截断质量及13%整模叙事仍待验。用户直接提供的 Kimi 意见已逐项审查。 |
| γ-off 自由竞争需 CLI `--tli_gamma off` | 显式几何等价物 `--tli_near_tokens off`（E155d；γ 接口已删除） |

## 6. 仓库结构速览（净化后）

- `sparse_attn/` — 主方法开源面：indexer（TLI/TIA/Quest）、patches
  （Qwen3/Llama3/DSV2 三族注册）、ops（eager decode、graph 链、
  sparse prefill）、info（treatment 身份）。
- `exp/trace/results/` — **判决 + 论文数据双归档，一个不删**。
- `exp/trace/` — 只保留被判决 script 字段引用的 14 个分析脚本。
- `exp/trace_archives/` — 全部历史归档（早期探针、旧图、tls_attn
  原型包、被淘汰 bench）；`agent_doc/advice/archive/` — 被推翻的
  advice 报告与 superseded research 文档。
- 对拍铁律：`exp/bench/test_e127_gates.py`（微逻辑回归门）+
  `exp/bench/test_tli_batch_gates.py` 每次改动后必须全绿。
