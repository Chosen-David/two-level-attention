# [TL-QWEN3-LBV2-VENDOR-20261112] Qwen3-8B LongBench v2 三 baseline 对齐与精度矩阵

Task-ID: TL-QWEN3-LBV2-VENDOR-20261112
Date: 2026-11-12

## Plan

### 目标与验收

按用户指令核查 Quest、JSSA、AsyncTLS 的 Qwen3-8B × LongBench v2 历史结果，统一最终 token 预算为 NFA K2=1024；AsyncTLS 粗筛固定 K1=128 blocks × 64 tokens。Quest/JSSA 必须执行官方代码或官方算子，AsyncTLS 按论文语义复现。统一 503 道四选一题、effective cap=31500、相同 prompt/chat、greedy、max-new=128 和 official-strict parser。

交付可复现入口、确定性互斥分片、runcap 身份、真实 Qwen3-8B smoke 和聚焦回归；推送 `origin/main` 后只在非本机空闲 GPU 运行远端精度矩阵。503/503 闭集、官方评分、哈希 manifest 和不同 owner 独立验收完成前，所有新质量数据保持 `pending-independent-verification`。

### 实现方法与历史边界

- 历史只找到 Quest 的旧 LongBench v2 汇总 30.22 和 FullKV 32.01，属于 legacy cap/parser 与跨机代码漂移口径，未找到 raw 503 行；JSSA/AsyncTLS 未找到 LongBench v2 结果。三者已有 LongBench v1 或 RULER 结果，不能冒充本任务。
- Quest 使用 `quest_vendor_kernel`：`benchmark/LongBench/pred.py` 加载 FP16 Qwen3，`exp/vendor_baselines/quest_vendor/qwen3_quest_kernel_patch.py` 直接调用官方 `quest.utils` 的 RoPE、append、prefill、estimate、top-k、sparse attention；官方 commit `01c1623bf9395009520874e989e29f683203b357`。固定 `token_budget=1024`、chunk/page=16、max_seq_len=40960；无 NFA K1 式额外粗筛。
- JSSA 使用独立的 `jssa_official`：`exp/vendor_baselines/jssa_vendor/qwen3_jssa_official.py` 直接加载 vendored 官方 Triton store/score/indexed-attend，官方 commit `ff65a6aec5c62534aef2a4d6f938c73546f74d92`。固定每 KV head budget=1024、rank=32、page=64，校准基为 `bases/qwen3_8b.pt`；无 NFA K1 式块粗筛。旧 `jssa_vendor` 仍是 Torch 语义移植，不与官方 runtime 混名。
- JSSA 完整 prompt 一次 dense prefill；短序列 eager 使用官方 partial 语义，稳定紧凑有效 Top-K 行后精确 attention，达到 budget 后进入官方 gather-free indexed-attend。每请求独立 cache，prefill 初始化物理槽，结束解绑所有 attention 层。
- AsyncTLS 使用 `asynctls_vendor` 论文语义复现：block=64、top_kb=128、token_budget=1024，每 KV group 32 个校准 channel、INT4 量化-反量化与组内平均 softmax；decode 使用上一步块集并同时计算当前块集。校准文件为 `dimension_channels/qwen3-8b/qasper_channels_per_group.json`。
- `benchmark/LongBench/run_longbench_v2_baselines.sh` 固定三臂参数并显式 `--nfa_enable_layer_skip false`，避免 vendor 不消费的 NFA 缺省掩码影响 treatment 身份。样本按 `data[rank::world_size]` 确定性切分，每片独立 `pred_postfix`；只有 complete、行数与 rank/world 身份均匹配才幂等跳过。
- JSSA basis 内容 SHA、Quest page/max_seq_len 和 budget/rank 均进入 treatment manifest/hash；runcap 记录源数据 SHA、选中 ID 集 SHA、实际输入 token 与截断标记。

## Progress

### 进度与证据

- 2026-11-12：完成 CLI、JSSA 完整生成与请求生命周期、短序列 official partial、调用计数、确定性分片、runcap 元数据、统一 runner 和 treatment 身份。
- 真实 Qwen3-8B rank0/world503 单样本三臂均完成，输出正确选项 `D`，实际送模长度均 31513，runcap 均 `status=complete`、`result_lines=1`、rank/world 与源 SHA 正确。日志中的 `284105 > 131072` 是中部截断前 tokenizer 通用警告，不是实际模型输入。
- Quest 产物：`/tmp/lbv2_smoke_20261112/quest_fixed/pred_r0of503/`；源码调用链证明生产入口执行官方 `quest.utils`/`quest._kernels`，不是 Torch reference。
- JSSA 产物：`/tmp/lbv2_smoke_20261112/jssa_fixed/pred_r0of503/`；receipt 为 36 层 prefill store、22 decode steps、792 次官方 indexed-attend（36×22）、partial=0、生成23 tokens并 EOS early stop。
- AsyncTLS 产物：`/tmp/lbv2_smoke_20261112/asynctls_fixed/pred_r0of503/`；输出和 runcap 完整。
- 校准身份：JSSA basis SHA256=`c0642d54a676d94f9e0f2227f364eda24c3655382f7e0c2c3b9199f19e315a22`；AsyncTLS per-group channel SHA256=`613a971cbc36d8e312a76af5ff2ee85bcd1bb0959c333ebd8e0605afd1524da4`。
- 回归：`tests/test_qwen3_jssa_official.py` CUDA 4 passed；`tests/test_longbench_v2_baselines.py`、`test_longbench_prompt_split.py`、`test_longbench_empty_handoff.py`、`test_qwen3_quest_graph.py` 合计17 passed + 4 subtests；runner `bash -n` 与 `git diff --check` 通过。
- 排障：tiny-Qwen 完整生成暴露 prefill 前未调用 `begin_prefill_request(0)` 导致未初始化槽，已修复并覆盖第二请求/EOS；初版 runner 未关闭 NFA layer-skip 导致缺省掩码门禁，已仅在 vendor runner 显式关闭，不改变既有全局 snapshot 语义。
- 全量 Quest/JSSA 原始分片已收回本地 `agent_doc/results/qwen3_lbv2_vendor_20261112/raw/`；`collect.py` 逐分片校验 runcap 字节SHA/行数、rank/world、源ID顺序、答案和 official-strict parser 后合并为503闭集。Quest=161/503（accuracy 0.3200795229，parsed 480），JSSA=165/503（0.3280318091，parsed 482）；两者仍 `pending-independent-verification`，不是最终独立验收结论。
- AsyncTLS 三分片仍在 pc31/pc40 运行，最近日志进度为106/168、109/168、96/167；未收回、未合并、未评分。Quest 初次 pc32 OOM 产物保留，正式闭集使用 `quest_retry1` rank0/1 与 pc02 `quest` rank2。

### 阻塞与下一步

- 继续监控 AsyncTLS 三分片；完成后收回原始 JSONL/runcap，运行同一 `collect.py` 校验合并并生成三臂 `summary.json` 与全文件 SHA256 manifest。
- 本机两张 H20 专用于 `TL-QWEN3-LBV1-VENDOR-PERF-20261112`，不与本精度矩阵混卡。
- 全部分数在不同 owner 独立验收前保持 pending；不得把生产侧校验通过表述为独立验收完成。
