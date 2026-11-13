# [TL-GLM-ASYNCTLS-LBV1-20261112] GLM-4.7-Flash AsyncTLS LongBench v1 全量精度

Task-ID: TL-GLM-ASYNCTLS-LBV1-20261112
Date: 2026-11-12

## Plan

### 目标与验收

按用户优先级，先完成 GLM-4.7-Flash × AsyncTLS × LongBench v1 英文16任务全量精度，再处理其他旧口径 baseline，最后重测 NFA。最终 token 预算固定1024，Stage-1 固定128个64-token块；逐 KV group 校准32个 channel，INT4 量化-反量化，decode 使用上一步块集，完整 prompt 一次 dense prefill。

完成真实47层校准与 smoke、16/16任务3750/3750请求、确定性互斥分片、runcap 输入身份、官方逐样本评分、原始证据和 SHA256 manifest 后，只提交本任务文件并推送 `origin/main`。生产侧校验不冒充不同 owner 独立验收，所有新分数保持 `pending-independent-verification`。

### 实现方法

- 复用 `glm4_moe_lite_asynctls_patch.py` 的 GLM MLA 投影和共享 AsyncTLS helper；GLM 每个 KV group 一个 query head，不额外做 GQA 平均。
- 校准使用 qasper 前8条、每条最多4096 token、`dc=32`，输出 `[layer][kv_group][channel]`，绑定模型、数据、代码和校准文件 SHA256。
- 使用 pc02 的 transformers 5.12.1 环境；真实运行只用点火前实测空闲 GPU，不与既有计算进程混卡。
- 聚焦回归覆盖 dense/full-budget 对照、prefill 块集初始化、decode 上一步块集、多步推进、第二请求复位、稀疏形状、非法 channel 和 EOS list。

## Progress

### 进度与证据

- 2026-11-12：pc02 `/home/fluentllmenv/bin/python` + `~/.local/pylibs_tf512` 已确认 transformers=5.12.1，可导入 `Glm4MoeLiteAttention`；本机 transformers 4.56.2 不支持该模型，不作为 GLM 测试环境。
- 原生 `Glm4MoeLiteAttention.forward` 源码已核对，adapter 的 Q/K/V 投影、RoPE、cache update 和 eager 输出接口与之一致。
- 新增 `tests/test_glm4_moe_lite_asynctls.py`；pc02 临时隔离树实测 `6 passed in 9.87s`，覆盖 full-budget native dense 对照、`M_{t-1}`、多步推进、第二请求状态复位、稀疏形状、非法 channel 与 EOS 提前停止。
- 临时测试树为 `/tmp/glm-atls-unit-20261112`，不是正式性能或精度产物；adapter SHA256=`ae308b22ddedb0b66334bb2eafa828f6bbe45f9dd47f3f30c13aee03357e6fad`，calibrator SHA256=`d796993f4c05c14e133a9611be40ea3ab62cd39b57832252606020b9d3665213`。
- 点火前盘点 pc02 GPU0/1 被 PID1476356/1476358 占用约66/70GB，GPU2/3 各1MiB；GLM 权重与 qasper 数据存在。GPU2 选作校准候选，正式 smoke/分片前重新检查。
- 真实47层校准已完成：qasper 前8条/max4096/dc32，输出47×20×32，channel SHA256=`87b28849a4ec034c6c52df4a0b1d35ef58c04d88a8557327d967140c1dc8058a`；本地与 pc02 字节一致。
- qasper/samsum 真实首样本 smoke 均完成并落 complete runcap；samsum 生成142.50s，输出重复文本只说明链路可运行，不作精度结论。GLM AsyncTLS 走完整 prompt dense prefill + eager 逐 token decode，samsum 不落入非 AsyncTLS 的 `model.generate` 特例。
- LongBench v1 已支持 `source_order[rank::world_size]`：先对完整有序源行计算 identity，再切分；输出 `_id` 使用完整源行号，runcap 同时绑定完整源人口、选中人口、source/selected SHA256。默认 rank0/world1 的源行 ID 与旧口径一致。
- pc02 transformers 5.12.1 环境聚焦回归 `19 passed, 12 subtests passed`，覆盖 LongBench v1/v2 分片、runcap、防重复源 ID 及既有6项 GLM AsyncTLS 语义。真实双卡四样本 smoke 每片2请求，complete ID 分别为0/2与1/3，完整源 `sample_count=4`、每片 `actual_input_ids.count=2`，证明多请求与分片身份闭包。
- 新增 `run_longbench_glm_asynctls.sh`，固定16任务人口、K2=1024/block64/K1=128、channel 文件、NFA layer-skip=false、互斥输出目录和 complete-only SKIP；4样本 smoke 用全量 runner 重跑时被 source=200 人口门禁拒绝，属于预期 fail-closed，未冒充幂等通过。
- 实现提交 `a0b086850ba7d290fda47c1c27833e7057e774bb` 已推送 `origin/main`；只包含本任务9个文件，未带入其他会话的 NFA 文档或 sqlite。pc02 内部 origin 无新提交且 GitHub fetch 阻塞，改用本地 `git bundle`（基元 `482911be`）离线部署到 `/tmp/glm-atls-release-a0b08685`，HEAD、pred/runcap/adapter/calibrator/channel SHA 均与本地一致。
- 固定提交部署后复跑 GLM/LongBench pytest `12 passed`，runcap unittest `7 tests OK`，runner 与双分片链脚本 `bash -n` 通过。新增生产侧 `agent_doc/results/glm_asynctls_lb_v1_20261112/collect.py`，将校验完整源身份、双分片 selected IDs、treatment/channel/commit、结果 SHA、官方逐样本评分与文件 manifest；合成4样本双分片闭集、源顺序与官方 scorer 测试 PASS。
- 2026-11-12 22:26 +08:00 正式点火：点火前 pc02 GPU2/3 各1MiB、MemAvailable=154250324kB，其他远端机器均有既有计算进程，未混卡。world=2，rank0/GPU2 runner=4105983、pred=4106005；rank1/GPU3 runner=4105984、pred=4106006；输出 `~/ak_verify_results/glm_asynctls_lb_v1_20261112`。点火后两卡显存约59GB，日志均进入 lcc `selected 250/500`，随后持续推进；完成前没有全量分数。

### 2026-11-13：保留1200条断点并按 NFA 输入口径跨机续跑

- 逐文件实读原输出，只有 `narrativeqa`（2×100）、`lcc`（2×250）、`repobench-p`（2×250）六个 JSONL，合计1200行；全部 runcap=`complete`、行数/SHA匹配、world=2、cap=31500、producer commit=`a0b086850ba7d290fda47c1c27833e7057e774bb`、pred SHA256=`cd155d8591591576c3723db65d1a31034ee9a3968dec5d73ed3a39a59e115f14`。没有 partial JSONL，不重跑或覆盖这1200条；其余13任务共2550条尚未生成。
- 用户要求配置对齐NFA并正确接入任务链。续跑仍使用同一模型/数据、LongBench中部截断、31500 cap、Qwen/GLM chat规则、官方dataset2maxlen、greedy与EOS停止；K2=1024、K1=128×64、dc32、完整prompt dense prefill、`M_{t-1}`及NFA layer-skip=false不变。config prompt/maxlen SHA256分别为`ac8056e8542bf4ad42b549d9ba450ef18819b6b2987f65973e20e7eeff6bbd1a`/`77e621d722647fc8d5293202cea5fbf0d06568848fac86de2b45cab122455391`，三机模型config/generation_config SHA一致。
- 按用户最新授权盘点 `agent_doc/guide/GPU_resources.md`：33.253.68.113与33.253.189.125均No route to host；pc40双卡均有计算进程；pc02/pc31/pc32各有空卡但可用主机内存只够再容纳一个GLM进程。采用剩余13任务world=3：rank0=pc02 GPU2、rank1=pc31 GPU1、rank2=pc32 GPU1；旧3任务仍为world=2，收集器必须按每任务真实runcap world验收，不能硬编码所有任务world=2。
- pc31/pc32通过增量git bundle部署独立worktree `/tmp/glm-atls-release-a0b08685`，HEAD和pred/runner/adapter/channel SHA均与pc02一致；bundle SHA256=`0e4b3def373497c7545b5d754c7f080b021c416441ce46a6126bf9d44be9e9c7`，runner链SHA256=`14df1aaa6a899ab89d705bb75ddb1f44f601c587c484f7e98533ed14555554b9`。三机Transformers=5.12.1，固定部署聚焦回归均`6 passed`。
- 18:50点火前pc02/pc31内存门槛通过；pc02 rank0 bash PID164076、pred PID164088，pc31 rank1 bash PID2815748、pred PID2815781。约1分钟后进程/显存/日志三重验证通过：两机均完成751/751权重加载并进入`gov_report selected 67/200`，输出目录分别为`pred_r0of3`/`pred_r1of3`。pc32当时MemAvailable=78040MiB<80000门槛，未降原rank0/rank1门槛、未点火，待资源自然释放后补rank2。
- 19:01起将rank2拆为三个互斥任务子集以利用实测空卡，worker脚本`/tmp/run_glm_atls_rank2_worker_20261113.sh` SHA256=`f299b04fa3a03ca5d90752e653d83b224afd5b98dda125d6bdb6860102704714`；新worker准入75000MiB，不改变模型、输入或算法，既有NFA等待链的100000MiB门槛不变。pc02 GPU1 bash PID166925负责`gov_report,qmsum,hotpotqa,samsum`，pc02 GPU3 bash PID167204负责`multi_news,musique,qasper,passage_count,trec`，pc32 GPU1 bash PID34318负责`2wikimqa,multifieldqa_en,triviaqa,passage_retrieval_en`；每任务仍只有一个rank2，rank0/rank1继续全13任务，合并目标为逐任务world=3闭集。
- 19:13五个父worker均存活且日志持续推进，pc32已完成`2wikimqa` rank2的66/200请求并于19:08进入`multifieldqa_en`；pc02 rank2两 worker 与rank0、pc31 rank1均在真实生成，未杀或抢占既有计算进程。
- 收集器已改为旧3任务world=2、其余13任务world=3，并用固定commit相同CLI确定性重建缺失的AsyncTLS treatment sidecar；canonical SHA256=`9547befd1fe3040e3d964cad7312e609e567988f31cfaecaaf2e2e777f532d4f`，与全部文件名`h9547befd1f`一致，JSONL/runcap字节不修改。隔离目录`/tmp/glm-atls-score-deps`安装jieba 0.42.1/rouge 1.0.1后，旧1200条完整源ID、runcap、输入身份、producer、treatment和官方逐样本评分全部PASS；生产侧重算narrativeqa/lcc/repobench-p分别为26.496911/67.078000/62.116000，仍保持`pending-independent-verification`，不作为全量或独立验收结论。
- pc32首个新分片`2wikimqa rank2/world3`已原样收回到本地`raw/2wikimqa/pred_r2of3`；66行、complete runcap、rank2闭集ID及字节SHA256=`f508cda011e168861783ce9354a27951410d8d419080f00fbb97ec83d53145f6`本地复核PASS，cap identity hash=`ec953f236b`。随后`multifieldqa_en rank2/world3`也已收回，50行、complete runcap、rank2闭集ID及SHA256=`22ce3a202881219df72d8c59d2d382357e90ca96e4576859da79675b4bc8ccf5`复核PASS，cap identity hash=`742fae2e8e`。两个分片均需与后续rank0/rank1合并后才构成完整任务。
- pc32`triviaqa rank2/world3`已完成并收回，66行、complete runcap、rank2闭集ID及SHA256=`dea378042376dc68c547ec340cdc990f482df3d0fcdf5f4a6e094a615d75c3b5`复核PASS，cap identity hash=`e1577047b9`。19:58原worker在启动最后`passage_retrieval_en`时因MemAvailable=73042MiB低于75000门槛而fail-closed退出，未生成partial；20:30复核GPU1空闲、MemAvailable=70584MiB后仅重启该剩余分片，准入70000MiB，新bash PID49553，751/751权重加载、66/200选样和compute PID4188243三重验证通过。
- pc32`passage_retrieval_en rank2/world3`已于20:38完成并收回，66行、complete runcap、rank2闭集ID及SHA256=`870a14085df1874b2709116256eda330a024b107c5b0aa15f21a70643d89f066`复核PASS，cap identity hash=`0283267e6f`。至此pc32原四个rank2短任务全部完成，旧1200条之外已有248条新complete分片落盘验收。
- 按用户“并行铺满空闲卡、逐卡balance”最新指令，20:49重新盘点：pc40双卡仍有其他计算进程，未抢占；保留四个在跑的`gov_report:r0/r1/r2`与`multi_news:r2`，将剩余31个未开始task:rank单元重新互斥排队。通用worker=`/tmp/run_glm_atls_balanced_worker_20261113.sh`，SHA256=`801448311ba3c27cf29eca79b1afd478b824992b8f66ca029ed59ce83b93d579`，本地集合校验`31 queued+4 active+4 complete=39`且无重复。A/pc02GPU2 queue PID184820等待pred164088后跑`qmsum:r2`；B/pc31GPU1 PID2835546等待pred2815781后跑`multi_news:r0,samsum:r0,qasper:r0`；C/pc02GPU1 PID184821等待pred166938后跑`multi_news:r1,samsum:r1,qasper:r1`；D/pc02GPU3 PID184822等待pred167216后跑`qmsum:r1,samsum:r2,qasper:r2`；E/pc32GPU1 PID53598立即跑`qmsum:r0`及其余20个短分片。只精确停止旧顶层调度父shell，当前runner/pred继续自然完成；五个新队列、E的751/751权重加载/67选样及全部在跑pred已三重复核，旧父链不会与新队列重复接力。
- 每小时`:07`轮询定时器已同步切换到上述平衡队列，旧cron `98ae4cbc`删除，新cron ID=`e63325ae`；继续负责complete分片收割、异常补跑、完整world=3任务验收和最终3750条汇总，不启动其他实验，session-only且3天自动过期。
- 21:14定时验收：A/B/C/D四个平衡queue均存活并正确等待各自pred，E/qmsum:r0正在真实生成26/67；在跑进度为gov_report:r0=18/67、r1=19/67、r2=19/66，multi_news:r2=57/66。三机无新增complete runcap，本地已收割的四个rank2分片仍为全部新完成证据，无world=3任务三分片齐套，未重复收割或启动补跑。

- 22:24定时验收与异常恢复：`multi_news rank2/world3`已原样收回，66行、complete runcap、rank2闭集ID及SHA256=`fd302e3f6430cb6068b90261dc2a8fd7d8c29f7c80522d4a3344ffdfd8002d5b`复核PASS，cap identity hash=`2ddfdb83bd`；`qmsum rank0/world3`已原样收回，67行、complete runcap、rank0闭集ID及SHA256=`740f2f0139a6dbb8ebbafaa8e4e266af0e45187cad8a9d7bf6993e23c744c437`复核PASS，cap identity hash=`c097ef884b`。本地已收割验收1581/3750条，两任务其余rank仍在跑或排队，均未冒充完整任务。
- 原E在`qmsum:r0`完成后启动`2wikimqa:r0`，因MemAvailable=64628MiB<65000门槛fail-closed退出，未生成partial。22:24复核pc32 GPU1空闲、无E残留进程且`2wikimqa:r0`目录为空，使用同一balanced worker、相同20个互斥短分片恢复E2；准入60000MiB仅作启动前资源保护，不改变模型、输入或算法，qmsum:r0未重复入队。E2 bash PID68745、runner PID68749、pred PID68757，751/751权重加载、`2wikimqa selected 67/200`、GPU1约68GB及真实生成2/67三重验证通过；pc32 GPU0既有计算进程未触碰。
- 22:19其余队列状态：A/B/C仍分别等待`gov_report:r0/r1/r2`自然完成；D已接力`qmsum:r1`至45/67。新增complete均已收割，无world=3任务三分片齐套，未启动其他实验。

- 23:18定时验收：新收回`qmsum rank1/world3`67行，complete runcap、rank1闭集ID及SHA256=`717f90fee1549c642883b8a454fddd9f1d5065548d037ae33e6555c7a024fcd0`复核PASS，cap identity hash=`c097ef884b`；另收回`2wikimqa rank0/1`与`hotpotqa rank0/1/2`五个complete分片。本地累计收割验收1982/3750条。
- `2wikimqa`与`hotpotqa`已分别通过world=3三分片完整源闭集、runcap、treatment、答案metadata和官方逐样本评分：2wikimqa=45.986905，merged SHA256=`68877e9e7b6da38e48c85f3165b9adf5dcfc495582013b9b048c1ea66489694a`；hotpotqa=61.031855，merged SHA256=`aee7d6e83919f15a719e16fd8a3dbe83cb1444e5ccb0eceae2039bf24f6759e7`。两分片组cap identity hash分别为`ec953f236b`与`a96446b897`；仅为生产侧重算，分数仍`pending-independent-verification`。
- D在`qmsum:r1`完成后因60996MiB<65000门槛于`samsum:r2`启动前fail-closed，E2在完成2wikimqa/hotpotqa五分片后因59956MiB<60000门槛于`musique:r0`启动前fail-closed；两目标目录均无partial、对应GPU空闲且无残留队列。23:17以相同worker、55000MiB启动前内存准入恢复D2=`samsum:r2,qasper:r2`与E3剩余15个短分片，不重复任何complete。D2 bash/runner/pred PID=207828/207832/207841，751/751权重、samsum 66/200选样和GPU3约75GB验证通过；E3 PID=77746/77750/77758，751/751权重、musique 67/200选样、GPU1约72GB及真实生成1/67验证通过。A/B/C仍等待gov_report自然完成，未触碰其他计算进程。

- 00:12定时验收：新收回`musique rank0/1/2`三分片与`multifieldqa_en rank0`50行，本地累计收割验收2232/3750条。musique完整world=3闭集、runcap、treatment、答案metadata和官方逐样本评分PASS，生产侧分数=36.106431，merged SHA256=`df15f01691899472387e89f3ef23d7f010001936249cdb1c8124a235a002cc99`，cap identity hash=`8e01aea586`；rank0/1/2结果SHA256分别为`04d7d357ee8ea95a819986e34de339ea43e08810e5c11f494a4866ac2e90b49b`、`c7372166025367c62731a9bbd562d52ab86c147438945f3fa86265fff6ea3156`、`3e31180f58089953a986c0d4c8f1383b17831f12cf7b607ae28c5f7c4f58137e`。multifieldqa_en:r0结果SHA256=`f910896d039b6d4057381fd2dc0c49a4d2d7ccab7b9b54835b4b52e20524fd32`，rank0闭集及cap identity hash=`742fae2e8e`复核PASS，r1正在28/50，尚未冒充完整任务。
- 本轮五队列均无新增异常：A/B/C继续等待gov_report自然完成；D2的samsum:r2推进至23/66；E3已完成musique并推进multifieldqa_en:r1。未启动其他实验、未覆盖complete或触碰其他计算进程，所有分数保持`pending-independent-verification`。

- 01:14定时验收：新收回`multifieldqa_en rank1/world3`50行，结果SHA256=`2b2a8283202e15aa83de81c2d43c090d08cbb37459cd993fdf5b6b72b1b9084f`；与rank0/2合并后完整150条闭集、runcap、treatment、答案metadata及官方逐样本评分PASS，生产侧分数=57.088600，merged SHA256=`0ef7209d801fb1dba35b09274e54a556b8b2e3d41d053af74c1804b13bad1198`，cap identity hash=`742fae2e8e`。本地累计收割验收2282/3750条，完整collector PASS共7/16任务，分数仍`pending-independent-verification`。
- E3完成multifieldqa_en:r1后因54860MiB<55000门槛于`triviaqa:r0`启动前fail-closed，无partial、无残留进程且GPU1空闲。01:13以相同worker、45000MiB启动前内存准入恢复E4剩余10个短分片：`triviaqa:0/1,trec:0/1/2,passage_count:0/1/2,passage_retrieval_en:0/1`，未重复任何complete。E4 bash/runner/pred PID=96484/96488/96496，751/751权重、triviaqa 67/200选样、GPU1约66GB及生成循环验证通过。A/B/C仍等待gov_report；D2的samsum:r2推进至53/66，无其他异常。

- 02:21定时验收：新收回`samsum rank2`66行、`qasper rank2`66行、`triviaqa rank0`67行，本地累计收割验收2481/3750条。三者complete runcap、rank/world及源ID闭集均PASS；结果SHA256分别为`b25a11839a5f6c3309395871f149c38b0d74e74cad384e1f4bdf47a1b99c1b07`、`1574d9d83b5b89a6b1c6bcd3faa0308ec3bbb46abae76c201c6e0d7694a37d43`、`9c4c35d85d2785a98f94b427af9625b547cfb56d5ffc369b2989c81ca696a38d`，cap identity hash分别为`2283b721ab`、`1aa50339d6`、`e1577047b9`。三任务其余rank仍在跑或等待接力，未冒充完整任务。
- D2已自然完成`samsum:r2,qasper:r2`并退出。按用户并行铺满与逐卡balance指令，02:19复核pc02 GPU0/GPU3和pc32 GPU0空闲后，将7个尚未开始的单元重排到即刻空卡与gov_report后续接力卡；只精确停止A/B/C旧等待调度shell及其tail，三个gov_report pred全程未触碰。新互斥分配为：D3/pc02GPU3=qmsum:r2；F/pc02GPU0=multi_news:r0；G/pc32GPU0=multi_news:r1；A2/pc02GPU2等待gov:r0后samsum:r0；B2/pc31GPU1等待gov:r1后samsum:r1；C2/pc02GPU1等待gov:r2后qasper:r0/r1；E4继续trivia:r1及后续8个短分片。无task:rank重复或遗漏。
- 新队列三重验证通过：D3 bash/runner/pred=231203/231220/231236，751/751权重与qmsum 66选样；F=231204/231219/231235，751/751权重与multi_news 67选样；G=104257/104261/104269，751/751权重与multi_news 67选样；A2/B2/C2 bash PID=231201/2885838/231202，日志分别等待正确gov pred=164088/2815781/166938。E4 trivia:r1推进35/67。未启动其他实验，所有分数保持pending。

- 03:14定时验收：新收回`gov_report rank0/1/2`、`qasper rank0`与`triviaqa rank1`五个complete分片，本地累计收割验收2815/3750条。gov_report三分片SHA256分别为`a494949e61b117b107a608f629e3ba82902418953a04e93df4c16a40d3c93080`、`8a80a97002ee7d5bfa481548d01896ae4c1f6d5e0d9a6de14fdf6afefbd82b08`、`48fec0e4c93ddad9772447cf5b781278ef42e8827904624f356d8e6632be9c10`，cap identity hash=`ddf3ef194c`；完整world=3闭集、runcap、treatment、答案metadata与官方逐样本评分PASS，生产侧分数=32.7948786306，merged SHA256=`68de53b8e3fb57379832ac86e16c64aa238118ff3b8baa27796410a3ca3dfb01`。triviaqa rank1结果SHA256=`2911f96404e09cea1d0db5676ba3f76c5be70ab6e06d1d2e68b175ec491bedcd`，三分片cap identity hash=`e1577047b9`；完整任务collector PASS，生产侧分数=91.575000，merged SHA256=`1f843219dd0e88c4889d746e881c035dc1df326709c091fa2d6867194ef525a8`。qasper rank0共67行，结果SHA256=`e17effc86517705a76d6b33d2d397440f9ea2313f9a75e06518704b06eda6667`，cap identity hash=`1aa50339d6`，rank0闭集复核PASS；完整collector PASS累计9/16任务，分数仍`pending-independent-verification`。
- C2在02:50完成qasper:r0后启动r1时CUDA OOM；只读定位pc02 GPU1已有另一会话NFA pred PID235298（`/tmp/glm_nfa_eps010_graph_frozen_20261112.py --rank 7 --world-size 8`），同卡两个约70GB进程导致失败。qasper:r1没有JSONL或complete runcap，仅有`.runcap_gate.lock`，不计入产量；未杀NFA、未覆盖complete。新增H/pc02GPU3互斥队列PID239756，仅包含`qasper:1`，等待qmsum:r2真实pred PID231236自然结束后接力，日志已确认`GLM_ATLS_BALANCED_WAIT queue=H_pc02g3 wait_pid=231236`。
- 03:09–03:14在跑状态：F/pc02GPU0=`multi_news:r0`19/67，A2/pc02GPU2=`samsum:r0`18/67，D3/pc02GPU3=`qmsum:r2`49/66，B2/pc31GPU1=`samsum:r1`12/67，G/pc32GPU0=`multi_news:r1`22/67，E4/pc32GPU1=`trec:r0`46/67，后续仍为`trec:1/2,passage_count:0/1/2,passage_retrieval_en:0/1`；H等待qasper:r1。未启动其他实验，所有分数保持pending。

- 04:19定时验收：新收回`qmsum rank2`66行、`qasper rank1`67行与`trec rank0`67行，本地累计收割验收3015/3750条。qmsum:r2结果SHA256=`bf515942ed5a194d2ee6d76d5da75b09e3ab359ee099b871f1f4e10e1d53a9e8`，qasper:r1结果SHA256=`514a0f1bdce88770604c4d5b41a17410393329ae2ba23fd453f53385d339e52d`，trec:r0结果SHA256=`3dac6e39eae9e3ef83ec2ff74c02f28abfb1254ac5841379163dbbc8ca8e2140`；三者complete runcap、rank/world、源ID闭集、答案metadata和输入身份复核PASS，cap identity hash分别为`c097ef884b`、`1aa50339d6`、`51f58327ab`。
- `qmsum`与`qasper`完整world=3 collector PASS：生产侧官方逐样本重算分别为23.3977817861与36.9821739358，merged SHA256分别为`b8457e00e521009984f3edd0c6e5be27ea1ad7899803f9bf489db8cc85ce82a2`、`539f7a28a9a200010992b637c0d1a4c482db66c4ccbbe27a346cb18396c8d203`；完整collector PASS累计11/16任务，分数继续`pending-independent-verification`。H/pc02GPU3已在qmsum:r2自然完成后接力qasper:r1并于03:41自然完成，无重复运行。
- E4在03:31完成trec:r0后启动trec:r1时CUDA OOM；pc32 GPU1已有另一会话NFA PID112280（容器显存PID3072425，约75.88GiB），与E4约63.55GiB混卡。trec:r1无JSONL或complete runcap，仅有`.runcap_gate.lock`，不计产量；未杀NFA、未覆盖complete。04:19将E4未完成且未在跑的7个短分片`trec:1/2,passage_count:0/1/2,passage_retrieval_en:0/1`互斥转移至I/pc02GPU3，queue/runner/pred PID=250600/250604/250613，751/751权重、trec:r1 67/200选样、GPU3约67.8GB及真实生成1/67三重验证通过。F=`multi_news:r0`、A2=`samsum:r0`、B2=`samsum:r1`、G=`multi_news:r1`继续运行，未启动其他实验。

- 05:24定时验收：新收回`multi_news rank0/1`、`samsum rank0/1`各67行及`trec rank1`67行，本地累计收割验收3350/3750条。multi_news:r0/r1结果SHA256=`40783dccc38b0537b90a62b7984f5e87704e31c99c1f505f65c552211a5b9cfb`、`351aa7ac4fb8b1c0cee7f0cbf1e1bdc1fb2a9295ef5f6bb76843e402261fc541`；samsum:r0/r1结果SHA256=`0517d7aa32a7e9eba7c9050ceecac02390ef64b556813ea1f5421c481f2fe8bb`、`170fcf2eed5cde838ac66690452b4bdd2d7804cd19ddc51c7642bd396fac0353`；trec:r1结果SHA256=`9d59465587affbbe8b5779d4e746f16a4602cd8c90a012eca602509bcc27e1dd`。五分片complete runcap、rank/world、完整源ID闭集、treatment与答案metadata复核PASS，cap identity hash分别为`2ddfdb83bd`、`2283b721ab`、`51f58327ab`。
- `multi_news`与`samsum`完整world=3 collector PASS，生产侧官方逐样本重算分别为26.7522797368与41.0184843996，merged SHA256=`2e094bf6c7f0b0efedcd1c2ee13894d0a3703e42b112b16cb75dd3e5bf39dab3`、`50e1513da9b69bb0acaa5aa7ccfe7ede17f354fbf346ad432aebe4f1fadb9cf5`；完整collector PASS累计13/16任务，分数仍`pending-independent-verification`。
- 按并行铺满与逐卡balance要求，只精确停止I旧调度父shell PID250600，trec:r1 runner/pred PID250604/250613全程未触碰并自然完成。剩余6个未开始单元互斥重排为：J/pc02GPU3 queue261667等待trec:r1后接力trec:r2；K/pc02GPU0 queue261875跑`passage_count:r0`，L/pc02GPU2 queue262166跑`passage_count:r1`，M/pc32GPU0 queue131265跑`passage_count:r2`，N/pc31GPU0 queue2909517依次跑`passage_retrieval_en:r0/r1`。点火进程、显存、751/751权重和选样日志均通过；J已于05:23自然接力trec:r2，其余队列真实生成，无task:rank重复，未触碰任何NFA进程。

- 05:32续收：`passage_count rank0/1/2`共200条全部原样收回，结果SHA256分别为`4ae6c01a05ae0aea71423bb6e3283e2376b2d210c8f0a4b48a6833908f46eb7d`、`c51f07ef22cefa72d094c034433ae3ef90e26e26425a241fc985217e7dc4093e`、`680e4d13b4535f0e4e28a48e62669d20b9df685686118e66f8517ebf994b9bab`，cap identity hash=`078b040102`；完整world=3 collector PASS，生产侧官方逐样本重算=4.500000，merged SHA256=`0c16020d4bada305c110dd21bc85f0957ca5f3e1e257cbe0fa68d21b5ceabb61`。该低分如实保留，不按预期修改配置或样本。完整collector PASS累计14/16任务。
- `passage_retrieval_en rank0`67条已原样收回，结果SHA256=`9033e5567fc5195f35c212f81f0ab20eb50baa351193a997d0bd54f6ef2df66a`，complete runcap、rank0闭集、输入身份、treatment及答案metadata复核PASS，cap identity hash=`0283267e6f`；N已自然接力rank1。当前本地累计3617/3750条，仅剩retrieval:r1与trec:r2共133条在跑，分数继续`pending-independent-verification`。

- 06:09定时验收：`passage_retrieval_en rank1`67条已于05:36自然完成并原样收回，结果SHA256=`9e58aa244db6cd1300b7d5b2a54f13b5568f3bf986e2d9501dc01c38f0b803a4`；三分片complete runcap、rank/world、完整源ID闭集、treatment、答案metadata及官方逐样本评分PASS，cap identity hash=`0283267e6f`。完整任务生产侧分数=95.500000，merged SHA256=`ed4d045cf4a1bdc98f29db247124f3c6756043fbfc32662df932baeb57eea71a`。本地累计收割验收3684/3750条，完整collector PASS共15/16任务，仅剩`trec:r2`66条；J/pc02GPU3进程、显存与日志正常，06:08推进至42/66。pc31/pc32无遗留AsyncTLS任务，未启动其他实验或触碰NFA，分数保持`pending-independent-verification`。

- 06:30最终收割：`trec rank2/world3`自然完成并原样收回66行，结果SHA256=`1422640a9842c198a557c1447236998e5ec7f6ad25fb4d150f10499472b262ce`，runcap SHA256=`62f394451ec9f34177881fdd220867aab50680a8234a022036d5eb52644325eb`，cap identity hash=`51f58327ab`。trec完整world=3闭集、runcap、treatment、答案metadata与官方逐样本评分PASS，生产侧分数=77.500000，merged SHA256=`43b880273cea6d31ff706e4f374413e58b4f05a6c5b6f6fb6f8ef89921863c5c`。J/runner/pred均自然退出，pc31/pc32无遗留AsyncTLS任务，未杀或抢占其他计算进程。
- 全量collector完成16/16任务、3750/3750请求；生产侧官方逐样本重算macro AVG16=`49.12033124404847`，micro AVG3750=`51.49035465622877`。结果=`agent_doc/results/glm_asynctls_lb_v1_20261112/summary.json`，summary SHA256=`8c554e9ab9aac7852ad04ee43829c04eb882a3fb49d5da96370d6ac1ef17138b`；153项manifest全部SHA256复核PASS，manifest SHA256=`70d01264d06bd7e4b9907fe35d8aeadd49987dc316d5a138c4294058c1484520`。
- 另一新进程未调用collector主流程，重新遍历原始JSONL/runcap/source/merged，复核完整源ID与顺序、答案metadata、官方逐样本分数、macro/micro和153项文件哈希全部PASS。这是生产侧第二遍只读复算，不冒充不同owner独立验收；全部结果继续保持`pending-independent-verification`。

### 阻塞与下一步

- 生产运行、原始证据回收、merged/summary/SHA256 manifest与生产侧复算均已完成；只提交本任务文档和结果证据并推送`origin/main`，不带入其他会话修改。
- 剩余工作仅为Task #21的不同owner独立验收；通过前不作`usable-with-scope`结论，不启动其他baseline或NFA跑批。
