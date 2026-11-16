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
- 2026-11-16：32K MK2 tail32/gate-off 正式重测四个分片均完成 25/25，合并后 source_index 恰好覆盖 0–99 且无重复。新审计器按 `source_order[rank::world_size]` 逐片核验并重排输入身份，真实 producer=`2ddfd741`、逐样本官方评分、36 层全 keep、K1/K2/page/near32/near-tokens-off/dc/sink/SWA/scoring/boundary、eager decode、max-new128/min-new1、EOS/truncation、位置容量、complete runcap 和文件哈希全 PASS；普通与 `python -O` 质量/分片回归均 10 passed，既有十五格旧 RULER 审计也再次 1500/1500 PASS。正式均分 83.0，相对同输入 FullKV=84.0 的差值为 −1.0 point，10000 次 paired bootstrap 95% CI [−10,+8]，仅记“未检出差异”，不作统计等效结论。原始预测、receipts、scores、manifest、日志、审计与固定脚本归档在 `nfa_dac_experiments_20261115/ruler_statsfix_2ddfd741` 和 `_dispatch`，旧异常 21.0 闭集原样保留，修正版作为 `A2_tail32_gate_off_statsfix` 新行加入质量表，仍保持 pending-independent-verification。首次 shell 仅在全部生成完成后把 manifest 检查路径多拼一层 `L32768`，因此没有 `.done`；实际四份 complete manifest 均在，未重跑这 100 条闭集，归档脚本已修正检查路径。
- 2026-11-16：32K 正式单元通过恢复门禁后，按同一干净 `2ddfd741` 启动其余 11 个受影响 NFA RULER 单元的双分片重测，每片 50 请求、每单元 100，输出=`~/ak_verify_results/nfa_dac_statsfix_matrix_2ddfd741_20261116`。pc02 两对双卡链依次负责 32K/64K/128K full-off 与 full-on，pc31 负责 32K/64K tail-on 和 128K tail-off，pc32 负责 64K tail-off 和 128K tail-on；四条链 arm×length×shard 互不重叠，固定 near32/near-tokens-off、max-new128/min-new1、真实 EOS 和 eager。pc02 四卡点火后已真实生成，pc31/pc32 初次误用无 Torch 的 `/usr/bin/python`，在模型加载前退出且未产生请求，失败日志已保留为 `_logs_no_torch_attempt_20261116`；改用两机既有正式 `/home/fluentllmenv/bin/python` 后重新点火，八条链均完成进程＋显存＋日志三重验证。当前仍在运行，尚未收割或评分；D16/D64 正式派发继续冻结，待其余修复单元逐格审计后再恢复。
- 2026-11-16：pc02 的 32K full/gate-off、full/gate-on 与 pc31 的 32K tail/gate-on 三个正式单元均完成两片 50/50，拉取后统一生产侧审计共 300/300 请求全 PASS。官方分分别为 88.0/89.0/82.0；full/off 全 3600 层 keep，两个 gate-on 臂均为 keep/near/far/both=3214/286/100/0。相对同输入 FullKV=84.0，配对差值依次为 +4/+5/−2 points，10000 次 paired bootstrap 95% CI 为 [−4,+12]/[−3,+14]/[−11,+7]；连同已完成的 tail/off=83.0（Δ−1，CI [−10,+8]），修正版 32K 四臂均仅记“未检出差异”，不作统计等效结论。`build_quality_ablation.py` 已扩展为消费两个独立 statsfix 归档、拒绝重复单元并按 source_index 合并分片评分，重建后 `quality_ablation.csv` 为 25 行；旧异常闭集保持不变，其余长度仍在运行，D16/D64 继续冻结。
- 2026-11-16：pc02 的 64K full/gate-off、full/gate-on 与 pc32 的 64K tail/gate-off 三个正式单元完成两片 50/50，连同 32K 三格的统一生产侧审计共 600/600 请求全 PASS。官方分分别为 80.0/75.0/80.0；full/off 与 tail/off 全 3600 层 keep，full/on 为 keep/near/far/both=2061/1539/0/0。相对同输入 FullKV=90.0，配对差值依次为 −10/−15/−10 points，10000 次 paired bootstrap 95% CI 为 [−17,−4]/[−23,−8]/[−16,−5]，均未通过逼近 FullKV 的恢复门禁，负结果原样保留且 D16/D64 继续冻结。tail/off 恰有 10 个样本从 FullKV=100 降为 0；full/off 也退化，不能把缺口仅归因于 tail32。已在空闲 pc40 双卡以同一干净 `2ddfd741` 对 full128 的 64K sample13 启动 K1=1024/K2=1024 与 K1=128/K2=8192 两个 max-new20 隔离诊断，点火后进程、各约 34GiB 显存和模型加载日志三重验证通过；诊断只区分 L1 粗筛漏页与 L2 预算截断，不改正式几何、不作正式结果。质量表重建为 28 行。
- 2026-11-16：其余修正版 RULER 单元完成两片 50/50，11 个 NFA 单元连同 FullKV 共 1100/1100 请求的生产侧审计全 PASS；64K full/off、full/on、tail/off、tail/on 为 80/75/80/79，相对 FullKV=90 的差值为 −10/−15/−10/−11，paired bootstrap 95% CI 分别为 [−17,−4]/[−23,−8]/[−16,−5]/[−19,−4]；128K 四臂为 69/70/64/54，相对 FullKV=86 的差值为 −17/−16/−22/−32，CI 为 [−27,−7]/[−26,−6]/[−32,−12]/[−43,−21]。32K 四臂为 88/89/83/82，相对 FullKV=84 均未检出差异。结果位于 `ruler_statsfix_matrix_2ddfd741`，质量表重建为 33 行；64K/128K 恢复门禁未通过，D16/D64 继续冻结，全部保持 pending-independent-verification。
- 2026-11-16：在 64K MK2 sample13 上以同一输入、max-new128/min-new1、真实 EOS、YaRN factor2 直接运行完整历史 TLI `08ce44d6`，tail32/full128、K1=128/K2=1024 两臂均为 0；当前 GraphReady 默认 current L2 与替换历史 L2 也均为 0。K1=1024/K2=1024 时 current L2=0、历史 L2=100，K1=1024/K2=8192 时 current L2=100，全保留物理 gather=100；说明默认预算失败不是 GraphReady 独有的旧引擎接入错误，且 L1 候选覆盖与 L2 排名联合影响结果，不能用扩大正式预算冒充默认修复。历史 runner 协议补丁、输入身份、脚本与日志均独立归档，旧失败诊断不覆盖。
- 2026-11-16：同一次历史 L2 apply 轨迹同时采集 L1/fine/target trace，108 层步的 far/near/candidate mismatch、Q mismatch、finite disagreement、fine/trace selection mismatch 全为 0。1197 个已进入 candidate 的目标 token 中 current/historical L2 实际选择 41/37 个；current missed raw rank 最小 922、historical missed strict-greater rank 最小 920，均正好超过实际 middle quota 892，未发现 Top-K 额外丢目标或 paged K gather/owner 去重错误。all-page 历史 L2 成功臂的 38 个目标头层步中，默认 candidate 缺 25 个，其默认同轨迹 far rank 最小 102、中位 339、最大 840，rank<96 为 0；该跨轨迹对照只作描述性定位，不冒充因果证明。
- 2026-11-16：同预算 decision-only L1 variants 显示简单 GQA mean/max/top2、Avg、MinMax+Avg 没有在目标覆盖、current/historical middle recall 与 all-page 成功相关覆盖上共同占优。真正 apply `mm_mean_plus_avg_mean` 与 `avg_mean`、保持 full128/K1=128/K2=1024/历史 L2/max-new128 后两臂官方分仍均为 0；完整 prediction、receipts、scores、runcap、manifest、L1 记录与日志已独立拉取，SHA256 分别为 `386fd21e40e097575a04f2ef59f8454c00ad5e143311b4b462a5c038352bcf21` 与 `cd521c4429808d9335ec59724b949ef114fda7d72294d8544247a8d54fb708d9`。恢复门禁仍未通过。
- 2026-11-16：为区分“压缩页统计过松”与“far 配额不足”，新增只做决策评估的 exact-page oracle：直接读取当前层真实 paged K，逐 token 使用与历史 fine-score 相同的 BF16 logit，计算页内 max/LSE 后再做 GQA mean/max，不改变实际轨迹或正式预算。首次两臂因 `kt_eff=1023` 与容量 `kt=1033` 形状不匹配，在首次选择前退出且未生成请求，失败目录保留；修正为 inactive 尾部显式 `-inf` 后在 pc40 双卡以 `current_exact_l1_mn128_v2`、`hist_exact_l1_mn128_v2` 重启，点火后进程、各约 34GiB 显存与持续推进日志三重验证通过，结果尚未收割。
- 2026-11-16：exact-page 两臂 decision-only 记录完成；进一步将 `exact_page_lse_mean`、`exact_page_max_mean` 真实 apply 到 L1，保持 full128/K1=128/K2=1024/历史 L2/max-new128，两臂官方分仍均为 0，日志 SHA256 分别为 `92720badcd961d5b58543c88f67c49d488cbcd80fec80e744fec67aa2ea998e7`、`02bd8090d45b5c419425b8f55e7883577d200b7c1da9fac4f41bd0932bcbe10e`，未通过样本恢复门禁。L2 variants 首两轮记录分别暴露实际 selection 重算/命中归约问题和目标坐标错误：旧记录误查页内 0–6，而真实 needle 在 15951–15957、47951–47957；v2/v3 无效记录原样保留。v4 改为实际 selection 只算一次、按 head 归约命中并直接使用真实 span，default-L1 与 exact-LSE-L1 两臂均通过 108 层步、216 trace 和 `applied historical_mean == trace` 门禁；v2/v3/v4 的 prediction/receipts/scores SHA 完全一致，证明修复只影响 decision-only 记录。默认 L1 下八个 L2 变体目标命中为 4–22、实际 historical_mean=21，exact-LSE L1 下为 32–52、实际为 47，但所有变体 layer7 均为 0；没有据总命中数直接 apply。
- 2026-11-16：新增 dense-probability L1 上界，使用真实 paged K、逐 query-head BF16 logit、末页 active mask、全序列逐头 softmax、GQA 概率聚合与页内 max/sum；与 all-page 历史 L2 对齐的 `exact_dense_meanprob_max` 在 current/historical L2 轨迹分别覆盖 433/434 个目标头层步，但 current/historical middle recall 没有同步提高，all-page 成功相关 38 目标仅覆盖 21/20。历史 L2 下 layer7 step0 head5、step2 head1 的 dense far rank 分别为 185/159，均超过固定 far quota96；quota96/128/160/256/640 的成功相关覆盖分别为 20/22/25/28/38，说明固定每头配额也是硬瓶颈，但该结果只是单样本跨轨迹描述，不能把 quota640 当成正式修复。dense 两臂实际轨迹未改，官方分仍均为 0，汇总=`diagnostics/nfa_selection_budget_diag_20261116/l1_variant_summary.json`。
- 2026-11-16：在同总 far 页预算 `8×96=768` 下记录跨头全局排名及 `global top768 ∩ 原每头 rank<256`，不改变实际选择。pc40 双卡 current/historical L2 两臂均完成 108 条 L1 记录、216 trace、完整 max-new128 manifest，官方分仍均为 0；prediction SHA 分别复现既有 current=`db9ff3dccef37ab75deb84c2617850fcbc35a15b24e85e86e174c1b1c4d42787`、historical=`61e3db881cf713d4c227cc28ca406351f81ca38dc0e12bf90efd72390e226f56` 轨迹，文件哈希和 production/candidate、固定/全局/cap 排名不变量全 PASS。历史 L2 下 production 固定/全局/cap256 对 38 个成功相关目标覆盖为 13/12/8，dense-meanprob 为 20/15/15；全局原始分数跨头比较有增益也有损失，且 layer7 step0 head5 的 dense global rank=1392、step2 head1=2886，均未进前768。所有已记录变体中也没有任一固定或全局方案同时保留这两个关键目标。按预定门禁停止该动态配额方向的真实 apply；cap256 字段只是交集上界，不冒充带 backfill 的完整配额模拟。证据=`global_quota_summary.json`、`analyze_global_quota.py` 及两个 `*_globalquota_l1_mn128` 独立目录，全部保持 pending-independent-verification；64K/128K 恢复门禁仍未通过，D16/D64 继续冻结。
- 2026-11-16：诊断目标真值改为从正式源行 `input` 与 `outputs[0]` 自动提取答案 token 的完整子序列，禁止手写坐标。64K sample13/23/32 的真实 spans 分别为 `(15951,15957)/(47939,47945)`、`(15607,15613)/(47574,47580)`、`(24489,24495)/(57003,57009)`；每次运行写 `target_truth.json`，分析器 fail-closed 校验输入身份和恰好两处目标。旧 sample13 第二处 `47951–47957` 等错误记录保留但不参与新判决；样本切换时分析器 SCORE 正则绑定 sample13 的问题也已通过环境变量后 reload 修复。
- 2026-11-16：固定 full128、K1=128/K2=1024、near_tokens=off、gate-off、current grouped-Q L2、max-new128 的三样本恢复矩阵完成：production MinMax 为 0/0/0；exact_dense_meanprob_sum 为 100/100/100（仅诊断上界）；avg_mean 为 100/0/100；avg_max 为 0/100/100；avg_top2_mean 为 100/100/100。`avg_top2_mean` 只使用现有页 Sum/Avg，对每个 query head 的页均值分取 GQA 最高两个分数的均值，是首个不扩大正式预算、保留 current L2 且共同恢复三样本的候选；三样本仍不等于正式闭集恢复。
- 2026-11-16：生产 `avg_top2` 经历两轮负结果后修复数值与执行语义。kernel-only 路径为 100/0/100，根因是诊断使用未回舍的 FP32 scaled-Q＋逐 request `einsum`，生产原 `_prepare_q` 会回舍 BF16 且 Triton 归约顺序不同；共享 unrounded-Q 又意外改变 near Avg，仍为 100/0。最终实现 split-Q：rounded BF16 Q 继续供 near/far mask 和 near Avg kernel，独立 unrounded FP32 scaled-Q 仅供 far Torch avg_top2；该 PyTorch 归约不能进入只记录 Triton launches 的 eager call-plan，故仅对 avg_top2 禁用 plan record/replay。Graph capture/replay 与实时 Q 测试覆盖 minmax/avg_top2 两条路径。
- 2026-11-16：新增非连续 request id=2 测试发现 Torch avg_top2 把 batch row 误作持久统计 request slot，MinMax 能选中新块而 avg_top2 读到全零统计；修复为 `k_sum.index_select(1, valid_req)` 后逐 row 归约，设备端 gather 无 host sync且兼容 Graph。修复后三样本 max-new128 官方分再次均为 100，每样本 216 trace、108 L1 记录、applied avg_top2 target mismatch=0；统一审计还核验输入/目标真值、K1/K2、full128、near_tokens=off、gate-off 全 keep、官方评分、runcap/receipts/scores/source manifest 与文件哈希，审计 SHA256=`c53645bb96266bed20bad01c21f0c73c648370d209bf4ce64ec3ed1e71385721`。诊断脚本及全部正负结果独立归档于 `diagnostics/nfa_selection_budget_diag_20261116`。
- 2026-11-16：正式实现从独立干净 `2ddfd741` 工作树提交为 `233d09afe535a697dc30038b3fe59f06b6304e00`，新增默认关闭的 `--nfa-far-method minmax|avg|avg_top2`、真实 CLI→NFAIndexer 配置链、split-Q/request-slot/eager多步/Graph测试；最终聚焦回归 92 passed，普通及 `python -O` L1 各 3/3，resolver PASS，`git diff --check` 通过。四机均通过 git bundle 部署同一干净 commit；64K MK2 四臂各两片 50 请求已在 pc40/pc02/pc31 八卡点火，固定 eager、max-new128/min-new1、真实 EOS、K1=128/K2=1024、near32/far96、near_tokens=off，输出=`~/ak_verify_results/nfa_dac_avg_top2_64k_233d09af_20261116`，已有 manifest 拒绝覆盖。点火后进程、各约34.2GiB显存和真实生成日志三重验证通过；当前尚未形成完整正式结果，D16/D64继续冻结，Torch avg_top2 也尚未完成性能验收。
- 2026-11-16：64K MK2 `avg_top2` 四臂八片全部完成 50/50，合并后每臂 source_index 恰好覆盖 0–99；统一生产侧审计共 400/400 请求 PASS，真实 producer=`233d09af`、同输入身份、逐样本官方评分、真实 gate、eager/max-new128/min-new1/真实 EOS、完整 NFA 几何、位置容量、complete runcap 和文件哈希均通过。full/off、full/on、tail/off、tail/on 官方分分别为 83/84/79/79；相对同输入 FullKV=90 的差值为 −7/−6/−11/−11，10000 次 paired bootstrap 95% CI 为 [−14,0]/[−13,+1]/[−19,−4]/[−19,−4]。full 两臂仅记“未检出差异”，tail 两臂仍显著退化，故整体恢复门禁未通过，D16/D64 继续冻结且未派发 128K。审计 SHA256=`237a97b3b5abab29de3cf930433cff1ef9a7a79c44702b8548fcd9c08345cbeb`。
- 2026-11-16：与 `2ddfd741` MinMax statsfix 同输入逐样本配对，avg_top2 相对旧臂差值为 +3/+9/−1/0 points，CI 为 [−6,+12]/[+1,+17]/[−9,+7]/[−9,+9]；只有 full/on 检出提升。full/off 有 12 个旧失败恢复但 9 个旧成功转为失败，full/on 为 13/4，tail/off 为 8/9，tail/on 为 10/10，说明三样本共同恢复不能泛化为无条件替代；配对转移记录=`ruler_avg_top2_64k_233d09af/_audits/paired_vs_statsfix.json`，SHA256=`e29c86c62efdfbf902c1231947dd8a82ac3b9eae10ae433433741462604891a9`。重建 `quality_ablation.csv` 为 37 行、inventory 为 307 行，独立 repair candidate 为 4 格/400 请求且不改变原 new_A 记账。主线合并时保留 `2ddfd741` 的 near-pages/near-tokens 接口并叠加 far-method，配置链 CPU 回归 23 passed；用户随后取消本机 RULER 多 BS/长度性能测量，本机未启动相关 GPU 任务，双卡保持空闲。
