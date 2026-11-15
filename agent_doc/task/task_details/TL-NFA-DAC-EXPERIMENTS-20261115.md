# TL-NFA-DAC-EXPERIMENTS-20261115

## 目标

恢复已有 NFA 投稿 DAC 实验数据，只补真实缺失的质量消融、参数敏感性和统一 baseline 性能，逐组形成论文可消费结果包；完成必要数据后正常推送 `origin/main`。不修改 Overleaf，不自动创建或合并 PR，不派生子 Agent。

主模型为 Qwen3-8B。质量主矩阵使用 HotpotQA 200、GovReport 200、RULER `niah_multikey_2` 32K/64K/128K 各 100，共 700 请求/臂；参数敏感性按任务文档和 inventory 中登记的实际子集执行。GLM-4.7-Flash NoPE 补充为低优先级，不阻塞 Qwen3-8B 主矩阵。

## 固定协议与复用纪律

- 共同 NFA 几何：page64、K1=128、near32/far96、K2=1024（含 protected）、sink4、SWA128、near_length=4096、dc=32、near Avg/far MinMax、boundary admission=true、obs256、完整 NFA 的 tau_near=tau_far=0.10。
- `nfa_near_tokens=off` 表示 L1 仍保留 near/far 页池配额，L2 不做 near/far token 配额分割而自由竞争；不得改写为固定 near-token 配额。
- 质量生成采用完整提示 dense prefill、greedy、正式 EOS/stop-string 和各任务正式 max-new；逐样本保存输入身份、预测、官方分数、生成长度和截断状态。
- 同长度配对必须使用完全相同的实际 `input_ids`。跨长度只能尽量保持种子和键值身份，不声称输入字节相同。
- 新性能采用 CUDA events 包围完整 32 个 decode 步，之前 3 步不计时；至少 2 次完整热身、5 次测量、轮换方法顺序，同时报告独立 host-wall。旧 `(T33−T1)/32` 数据不混入新曲线。
- 历史结果不可变；复用须核对真实模型、样本 ID、输入、运行参数、生成/计时协议和证据状态。历史 `tail32`、tau=0.05、question-suffix 等不得改标成新协议。
- 每批新数据在独立 owner 验收前保持 `pending-independent-verification`。CI 含 0 只能写“未检出差异”。
- 不重复已闭集的同配置实验；不把 smoke、partial、OOM 或失败 attempt 当正式结果。Quest/JSSA 无官方高性能路径时 fail-closed，不用串行 BS1 冒充 BS>1。
- 新 GPU 任务前实测空闲与 compute apps，性能卡独占、质量与性能分卡；点火 30 秒内完成进程＋显存＋log 三重验证，不用 `pkill -f`。

## 已恢复资产与边界

| 资产 | 当前判定 | 可消费范围 |
|---|---|---|
| `nfa_ablation_20261030/B_final_review` | 5 cells × 2 tasks × 200，共 2000/2000；独立 reviewer 官方重评分与 paired bootstrap 已验收 | 仅作为 tail32、epsilon=0.05、reclaim=true 的旧两任务区域评分组；新 C 的 HotpotQA/GovReport 优先复用，不重复生成 |
| 旧 A/C/E 历史报告 | 原 113/125 于 2026-11-15 实测 `No route to host`；本机 bundle 只有源码线索，无原始预测包 | 仅参考，不作为可消费数据，不凭摘要重建结果 |
| `nfa_obs_ablation_evidence_20261106` | 固定真实 Q/K 的 CPU dense-oracle/L1 机制证据 | 40 Q/K、1280 query cells、10240 KV-group cells；不替代端到端质量 |
| GLM subspace 完整 audits | native_full、random32 s0/s1、uniform32 有完整 200 条结果 | 可按真实 arm 名复用；uniform32 不得改标 first32/last32，其他缺口低优先级补测 |
| `qwen3_official_ruler_multikey2_perf_20261115` | 48 格旧 host-wall 测量组：26 measurable、2 OOM、4 JSSA blocked、16 unsupported | 保留全部 repeat 和异常；不能替代新 CUDA-event 矩阵 |
| `qwen3_nfa_fi_longbench16_20261115` | NFA/FlashInfer 均 16/16 任务、3750/3750 请求生产侧闭集，输入身份和官方评分 PASS | 仅 FullKV 的 HotpotQA/GovReport 两行可复用，共 400 请求；NFA 实际 prefill epsilon=0.05，不是新 A2 gate-off |

### Qwen3 LongBench 历史候选的真实运行身份

固定 runner：`/tmp/run_qwen3_nfa_fi_quality_eager_cbfd70da_20261114.sh`。NFA 质量入口 `benchmark/efficiency/benchmark_longbench_quality.py::_generate_nfa_quality()` 调用无参数 `nfa_args()`；固定 worktree 中真实导入和 `git diff --exit-code` 已确认：

- producer=`cbfd70da607eb5afed4343d6d055073f450e4162`。
- `nfa_subspace=tail`、`nfa_l2_subspace=same`、`tia_level2_cmp_ratio=4`。
- `nfa_near_tokens=off`、`nfa_enable_layer_skip=false` 只关闭已删除的静态 D′ 消费路径；`prefill_frozen_request` 仍用 `GateConfig(epsilon=0.05, rows=256)` 做动态区域 decisions，可能真实 skip。因此这批 NFA 只能作为历史 tail32/epsilon0.05 参考，**不能复用为新 A2 gate-off/epsilon0**，也不是 A4 tau0.10。
- FlashInfer producer=`7fac274fda9df13e82caadd95f3cca5c8374a1a3`，SDPA prefill＋FlashInfer page64 eager decode，可作为同输入、同生成协议的 FullKV 参考。

| arm | HotpotQA | GovReport |
|---|---:|---:|
| 历史 tail32/prefill-epsilon0.05（非新 A2） | 59.15 | 31.66 |
| FullKV FlashInfer | 59.42 | 33.48 |

两臂 HotpotQA 共同输入 ordered SHA256=`be1da2fa86c93e321479e167d10cba804da0076be419879cfedc35703db12d08`；GovReport 共同输入 ordered SHA256=`cdf8eb4e8e259acc4c6ec2c307457f7ff920ea47748d9043b96650d9519e013d`。prediction SHA、producer 和逐行结果位置已写入 `inventory.csv`，不能只凭目录名复用。

## Inventory 与真实缺口

可复现生成器：`agent_doc/results/nfa_dac_experiments_20261115/build_inventory.py`。输出：

- `inventory.csv`：303 行，record_id 全唯一。
- `inventory_summary.json`：分组人口、复用说明和 pending 状态。

新 A–F 质量及新性能记录均显式写出 `near_tokens=off`；FullKV 不继承 NFA 几何。新 A 共 25 行，其中恰好 2 行复用、23 行待生成：

```text
原矩阵                         5 arms × 700 = 3500 请求
复用 FullKV HotpotQA/GovReport                400
逻辑剩余（待复用行独立验收）                 3100
  A1 全维/gate-off 全 700
  A2 tail32/gate-off 全 700
  A3 全维/gate-on  全 700
  A4 tail32/gate-on 全 700
  FullKV MK2 三长度         300
```

B/D/E/F 的默认点只登记 cross-reference，须在运行时等价核验后引用，禁止重复生成。旧 B 与新 MK2 不拼成严格单因素组；旧 host-wall 与新 CUDA-event 性能分组展示。

## 后续执行顺序

1. 先完成 Qwen3-8B 新 A 的 3100 个逻辑剩余请求，只复用上述 FullKV 400 条；历史 NFA epsilon0.05 请求不进入新 gate-off 组。
2. 补 MK2 三长度坐标与区域选择对照，再完成 dc/tau/near-length 敏感性。
3. 定位旧 NFA 128K/BS2 三次 `15.807458/3.357201/21.380053ms` 波动；未定位前标不稳定，不删异常点。
4. 执行 FlashInfer、完整 NFA、NFA gate-off、Quest、JSSA、AsyncTLS 的新 CUDA-event 统一矩阵；unsupported/blocked/OOM 原样保留。
5. 分组交付原始预测、逐样本官方评分、原始计时、固定配置/命令、配对 CI、`quality_ablation.csv`、`sensitivity.csv`、`performance.csv`、可编辑单栏图及 README。
6. GLM NoPE first/last/random/fullNoPE 最后补，不阻塞主矩阵。
7. 新结果完成必要审计后尽快正常推送 `origin/main`；独立 owner 验收前不作最终论文结论。

## Progress

- 2026-11-15：建立任务；完成旧 B、旧 A/C/E、observation、GLM、旧性能和 Qwen3 NFA/FlashInfer 的只读检索与复用判定。113/125 不可达，未伪造恢复或重跑已闭集两任务。
- 2026-11-15：初版 inventory 曾把旧 NFA 的 `layer-skip=false` 误作新 A2 gate-off；真实调用链复核后纠正：prefill `GateConfig(epsilon=0.05)` 独立产生动态 decisions，旧 400 条 NFA 仅保留历史参考。修正后新 A 恰好 2 行 FullKV 复用、23 行待生成，复用 400 条、逻辑剩余 3100 条；全部复用结果保持 `pending-independent-verification`。
- 2026-11-15：质量入口提交 `cfd5a853` 暴露 full/tail × gate off/on；pc40 两个 gate-off smoke 完成单样本生成，但配置复核发现 gate-off 仍需显式 prefill epsilon=0。两条 gate-on smoke 均在模型加载阶段被不存在的默认静态 D′ mask 阻断，未产生 gate-on 数字。
- 2026-11-15：当前 GraphReady `NFAIndexer` 已无 `snapshot.skip_ids` 消费者；快照仅在显式声明旧静态路径时读取文件，不再展开无关默认 mask。gate-off 改为 `GateConfig(epsilon=0)`，请求回执新增逐层 decisions、四类 gate counts、epsilon/rows，并强制 off 全 36 层 keep。普通聚焦回归 57 passed，`python -O` 18 passed。下一步同步 pc40 后重跑四臂 smoke，再开始新 A 正式缺格。
- 2026-11-15：`dfb750d7` 修正质量臂配置与回执门禁；pc40 四臂修正版 smoke 均真实生成且输入 SHA 一致，off 两臂 36/36 keep，on 两臂 epsilon=0.10/obs256。随后双卡启动新 A LongBench 正式缺格：full/tail × gate off/on × HotpotQA/GovReport，每任务 200，共 1600 请求，输出 `~/ak_verify_results/nfa_dac_longbench_formal_dfb750d7_20261115`；当前仍在运行，未评分或归档。
- 2026-11-15：新增统一 RULER 质量入口并推送 `67f3b099`，复用真实质量引擎、正式 max-new128/min-new1、真实 EOS、`string_match_all`、输入身份和完整 runcap；普通及 `python -O` 聚焦回归均 20 passed。pc31 32K 五臂 sample0 smoke 已完成，生产侧审计确认五臂输入 SHA=`725ed411e6a7557a27e468a49b8a882856643fdf6a63cbbf139f299019a2e694`，off 全 36 层 keep，on 为 epsilon=0.10/obs256，预测、逐样本官方分数、receipts、complete runcap 与文件哈希全 PASS；证据=`~/ak_verify_results/nfa_dac_ruler_smoke_67f3b099_20261115`，仅为 smoke，不作质量结论。
- 2026-11-15：pc31/pc02/pc32 独立工作树均固定 `67f3b099` 且干净；三机长上下文输入各 100 行就位。32K/64K/128K 全部五臂 sample0 smoke 共 15/15 单元完成，生产侧统一审计确认逐样本官方评分、输入身份、真实 gate、complete runcap 和文件哈希全 PASS；`verify_runtime_rope` 对 64K/128K 的 YaRN factor 2/4、实际频率与旋转探针均 PASS。证据=`~/ak_verify_results/nfa_dac_ruler_smoke_67f3b099_20261115` 与 `~/ak_verify_results/nfa_dac_ruler_yarn_smoke_67f3b099_20261115/smoke_matrix_audit.json`，smoke 分数不作质量结论。
- 2026-11-15：新 A RULER 1500 条正式缺格已按 15 个互不重叠的臂×长度单元派发：pc31 双卡负责 32K full/tail off 与 64K full/tail on，pc02 四卡负责 128K 四 NFA 臂及 32K full/tail on、32K/64K FullKV，pc32 负责 128K FullKV 与 64K full/tail off。输出=`~/ak_verify_results/nfa_dac_ruler_formal_67f3b099_20261115`；八条链点火后均完成进程＋显存＋日志三重验证，pc31 两臂已复现 smoke sample0 的真实生成。当前仍在运行，未收割或评分；所有新结果保持 `pending-independent-verification`。
