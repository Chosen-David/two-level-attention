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
- 2026-11-15：用户提醒复核“第 128 个 token 恰为 EOS”的边界。真实 NFA/FlashInfer `observe()` 均先判 EOS、后判 max-new，但 RULER 汇总原先把 `len==128` 无条件等同于 truncated，确定会将合法的末 token EOS 误判并抛错；新增回归先在旧代码稳定复现，再改为仅当 `stop_reason=max_new_tokens` 时要求长度达到预算。新增真实 NFA 异步回执末 token EOS 优先级测试与 RULER 完整闭集测试，并补强 page64/K1/K2/near/dc/sink/SWA/scoring/boundary 配置断言；普通及 `python -O` 质量聚焦回归均 40 passed。正在运行的 `67f3b099` 进程不打断、不改其工作树；若实际命中旧汇总 bug，仅对受影响单元使用修复版本重跑，完整闭集不重复生成。
- 2026-11-15：EOS 修复与回归已作为 `cdb83688` 正常推送。pc31 的 32K full/tail gate-off 两个正式单元均完成 100/100；生产侧逐样本官方评分、输入身份、真实 36 层全 keep、K1/K2/page/near/dc/sink/SWA/scoring/boundary、complete runcap 和文件哈希审计全 PASS，官方分为 15.0/21.0，原始预测、receipts、scores、manifest 和审计已归档至 `nfa_dac_experiments_20261115/ruler_formal`，仍保持 pending-independent-verification。原派发脚本把 manifest 检查路径多拼一层 `L32768`，完成生成后以 rc=6 停止后续单元；未重跑已闭集的 200 请求。修正检查路径后，pc31 双卡在 `cdb83688` 上仅续跑 64K full/tail gate-on 两个缺格，点火已完成进程＋显存＋日志三重验证。其余在飞旧进程未热改，全部日志尚未出现末 token EOS 汇总错误。
- 2026-11-15：pc32 的 64K full-dim/gate-off 正式单元完成 100/100，旧 shell 同样只在生成后因 manifest 检查路径 rc=6；未重跑该闭集。生产侧审计确认逐样本官方评分、输入身份、36 层全 keep、完整 NFA 几何、真实 decode/eager 计数、complete runcap 与哈希全 PASS，官方分 12.0，已追加归档并使本地三格统一复审计通过。pc32 GPU0 的 128K FullKV 保持原工作树运行；GPU1 使用独立 `cdb83688` 工作树仅续跑缺失的 64K tail/gate-off，点火后进程、约 34.1GiB 显存与模型加载日志三重验证通过。新 A 已复用 400、新生产 300、逻辑剩余 2800 请求。
- 2026-11-15：长度上限按“任务输入 cap”与“实际输入＋生成的位置容量”分别收口。RULER 保持源提示不截断，已归档 32K 的最大实际输入为 32624、加 max-new128 后为 32752≤40960；64K 为 65481+128=65609≤65664，截断行均为 0。LongBench 保留官方 31500 输入 cap 和逐行截断标志；当前完整 tail/gate-off 臂 GovReport 最大实际输入 31512（含模板额外 token），加 512 为 32024≤40960，源截断 3/200，HotpotQA 为 0/200。审计器新增实际输入加 max-new、逐样本输入长度/截断绑定与位置容量门禁；32K 不额外启用 YaRN，64K/128K 的 factor2/4 和真实旋转探针继续强制验证。后续 256K/512K 性能只在真实输入、factor8/16 与显存均满足时测量，不足则记 OOM/unsupported，不缩短输入冒充完成。
- 2026-11-15：pc31 的 64K full/tail gate-on 两个正式单元均完成 100/100，真实 producer 为干净的 `cdb83688`，官方分 11.0/10.0；两臂 gate counts 均为 keep/near/far/both=2061/1539/0/0，epsilon=0.10、obs256 和完整 NFA 几何逐样本复核 PASS。pc40 的 tail32/gate-off LongBench 完整臂完成 HotpotQA/GovReport 各 200/200，官方分为 58.98/31.74，400×36=14400 层全 keep；官方命令与 `sample_score` 逐样本重算、输入身份、runcap、decode/eager 计数及文件哈希全 PASS。相对同输入复用 FullKV（59.42/33.48），差值为 −0.4373/−1.7443 points，10000 次配对 bootstrap 95% CI 分别为 [−1.7490,+0.7496] 与 [−2.2023,−1.2885]；HotpotQA 仅记“未检出差异”，不作等效结论。新增 `build_quality_ablation.py` 可重复生成结果表，审计脚本已入 `_dispatch`。新 A 当前复用 400、新生产 900、逻辑剩余 2200 请求；全部保持 pending-independent-verification。
- 2026-11-15：pc02 的 128K tail/gate-off、tail/gate-on 与 pc32 的 128K FullKV、64K tail/gate-off 四个闭集已拉取，连同既有五格通过九格统一生产侧审计，共 900/900 请求；真实 producer、逐样本官方评分、同长度输入身份、gate decisions、decode/eager 计数、complete runcap、文件哈希、输入加 max-new 位置容量和零截断全 PASS。64K tail/gate-off 官方分 13.0、36 层全 keep；128K FullKV/tail-off/tail-on 官方分分别为 86.0/8.0/5.0，tail-off 全 keep，tail-on 为 keep/near/far/both=1367/2233/0/0。128K 最大实际输入 131026，加 max-new128 为 131154≤131200，三格源截断均为 0。相对同输入 FullKV，tail-off 差值 −78 points、10000 次配对 bootstrap 95% CI [−86,−69]，tail-on 差值 −81 points、CI [−89,−72]；负结果原样保留，只限这些已完成单元，不作未测长度或总体等效结论。重建后 `quality_ablation.csv` 为 13 行，新 A 当前复用 400、新生产 1300、逻辑剩余 1800 请求；全部保持 pending-independent-verification。
- 2026-11-15：pc02 的 128K full/gate-off、full/gate-on 两个单元随后均完成 100/100 并形成 manifest，拉取后十一格统一审计共 1100/1100 请求全 PASS。两臂官方分分别为 10.0/6.0；full/gate-off 全 3600 层 keep，full/gate-on 为 keep/near/far/both=1367/2233/0/0。相对同输入 FullKV=86.0，差值分别为 −76/−80 points，10000 次配对 bootstrap 95% CI 为 [−85,−66]/[−88,−71]。至此 128K 的 FullKV 与四个 NFA 2×2 单元全部闭集，负结果完整保留；重建后 `quality_ablation.csv` 为 15 行，新 A 当前复用 400、新生产 1500、逻辑剩余 1600 请求，仍全部保持 pending-independent-verification。
- 2026-11-15：按用户新增要求将 D 组明确为 16/32/64/128 维。LongBench/RULER 质量入口新增 `--nfa-subspace-dim`，tail16/32/64 分别真实映射到 `tia_level2_cmp_ratio=8/4/2`，full128 保持 `enable_subspace=False`；不匹配维度与子空间 fail-closed，默认 A 组命令及目录名不变，仅显式 16/64 维使用 `tail16/tail64` 新臂名。运行时回执新增 `subspace_dim/subspace_cmp_ratio/enable_subspace`。新增 `NFAIndexer._subspace_indices()` 逐元素测试，验证两个 64 维半区实际选取的低频尾维，普通及 `python -O` 质量聚焦回归均 47 passed。D inventory 已显式登记 ratio 和 full 路径；32/128 维仍只在运行时等价核验后 cross-reference A2/A1，不重复生成。16/64 维 GPU smoke 与正式结果尚未产生。
- 2026-11-15：pc02 的 32K FullKV 正式单元完成 100/100，真实 producer=`cdb83688`，官方分 84.0；仅整理本地 tar 多嵌套的一层目录并删除本地运行时锁，未改远端现场或重跑闭集。整理后十二格 RULER 统一生产侧审计共 1200/1200 请求全 PASS。32K full/gate-off、tail/gate-off 相对同输入 FullKV 的差值分别为 −69/−63 points，10000 次配对 bootstrap 95% CI 为 [−78,−59]/[−73,−52]，负结果原样保留。pc40 的 full/gate-off LongBench 完整臂完成 HotpotQA/GovReport 各 200/200，官方分 59.20/32.16，14400 层全 keep；与 tail/gate-off 统一审计共 800/800 请求全 PASS。full/off 相对 FullKV 的 HotpotQA 差值 −0.212776、CI [−1.284030,+0.794738]，仅记未检出差异；GovReport 差值 −1.320235、CI [−1.788975,−0.861831]。
- 2026-11-15：`b70d8c08` 的 tail16/tail64 RULER 32K sample0 smoke 均真实生成 128 tokens，输入 SHA 与既有五臂 smoke 一致，36/36 层全 keep；扩展审计确认真实 `subspace_dim=16/64`、ratio=8/2、`enable_subspace=True`、完整 NFA 几何、官方逐样本评分、runcap 和文件哈希，2/2 单元 PASS，明确 `formal_result=false`，smoke 分数不作质量结论。`build_inventory.py` 与 `build_quality_ablation.py` 已从规范归档成功重建：inventory 303 行且 ID 全唯一，新 A 复用 400、新生产 2000、逻辑剩余 1100 请求；质量表 18 行，全部继续保持 pending-independent-verification。
- 2026-11-16：定位并修复两项 GraphReady 精度/正确性回归。① `_nfa_stats_dirty` 会改变 fused page-stats/Q kernel 的 `STATS_PROGRAMS` 与 Q program 起点，但旧 `_nfa_launch_geometry` 未覆盖该状态，禁用外层 call-plan 后稳态会复用首步 compiled kernel 并沿用旧 `q_full`；已把 dirty 状态纳入内层 launch geometry，同时让外层 call-plan 比较忽略这个仅限首建/稳态的特化位，避免第二步 layer0 重录。② `39b5d16b` 将持久页统计 key 从按层隔离改成 `(dim_numel,dtype,need_sum)`，36 层共享一份历史前缀；首步各层全量重建均正确，下一步增量只刷新最新块，使 layer0–34 的历史块沿用上一层、最终通常为 layer35 的统计。真实 decode1 的 layer0/1/34/35 min/max/sum 全部逐位正确，decode12 的 layer0/1/34 各约 12.8 万个元素分歧且 block0 与 layer35 真实统计相同，layer35 全对；已恢复 `(layer_offset,1,dim_numel,dtype,need_sum)` key，保持单 buffer 形状不变。
- 2026-11-16：修复提交 `2ddfd7416aedc03f0e43c6ea115c25c7212fa974` 已正常推送 `origin/main`，同时包含此前数值 `nfa_near_tokens` 的真实双区间 L2 实现与 off/Graph 回归。修复后 tail32/gate-off/near-pages8/near-tokens320/eager 的 32K MK2 sample0/1/2 max-new20 诊断分别输出正确答案 `6073160/2378217/6360170`，官方分均 100；最终 plan-on 与 plan-off JSONL SHA256 均为 `8346e16a0a999b4b01c31853634a81d6a5539bc88abeebd798c7d32cc88a3122`。完整 selector 回归 58 passed，paged-cache＋multi-request Graph 回归 33 passed；首次完整回归发现的 alternating-region plan 复用失配已修复并包含在上述最终测试中。三样本与 max-new20 仅为 bug 诊断，不覆盖旧异常闭集、不作为正式论文结果。
- 2026-11-16：按真实修复启动 32K MK2 tail32/gate-off 正式重测，使用共同默认 `near_pages=32`、`near_tokens=off`、max-new128/min-new1/真实 EOS，固定干净工作树 `2ddfd741`，pc02 四卡执行互不重叠的 `shard-index=0..3/num-shards=4`，每片 25 请求。输出=`~/ak_verify_results/nfa_dac_statsfix_2ddfd741_20261116/nfa_tail_gate_off`；首次 attached 启动为避免 10 分钟 SSH 上限已按精确 PID 停止并完整保留为 `_attached_attempt_20261116T0330`，随后 setsid 独立点火，四卡进程、各约 25GiB 显存和日志三重验证通过。sample0/1/2/3 正式 max-new128 均官方 100；完整 100/100、manifest、哈希和与 FullKV 的配对比较尚未完成，继续保持 pending-independent-verification，D16/D64 暂不恢复派发。
