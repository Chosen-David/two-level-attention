# TL-GLM-SUBSPACE-SKIP-RETEST-20261109

## 目标

按用户 2026-11-09 指示重测 GLM-4.7-Flash 子空间：旧 `first32/last32` 与新候选存在
producer、样本量和 near/far gate 口径差异，不再跨口径排名。统一比较纯子空间与实际
skip 开启两组，最终结论以本轮独立审计为准。

## 固定协议

- 任务：RULER 64K `niah_multikey_2`，n=100，seed=42。
- producer：`benchmark/RULER/pred_ruler.py` SHA256
  `802d3ed94d0ee59f5cd4cdd8370ff9ce95cbaef8cb3eda9127f7d12729644e07`。
- 快照：`/tmp/tli-glm-jssa-prod-20261106`，GIT_HEAD 登记
  `6760cc495f554ffa85e8eb6d6e552ce02911f457`。
- 候选：`first32`、`last32`、`nope_uniform32`、`nope_random32(s0/s1)`、`full256`。
- 每个候选分别运行：
  - `full`：47 层全 keep，隔离纯子空间影响；
  - `dual`：显式开启 prefill-frozen near/far skip，epsilon=0.05、rows=256、
    statistic=mean。所有 dual 臂必须出现实际非 keep 决策才能 PASS。
- 每臂独立输出目录，审计 100 唯一 ID、源答案 SHA、producer SHA、receipt SHA、
  47 层 decision/KV 一致性、执行模式和生成 token IDs。

## 执行与产物

- 机器：pc32，双 worker（GPU0/worker0、GPU1/worker1），每 worker 六个臂串行。
- 入口：`/tmp/run_glm_subspace_skip_matrix_20261109.sh <GPU> <WORKER_ID>`。
- 审计：`/tmp/audit_glm_subspace_skip_20261109.py`；已用既有 full-region 100 行
  产物自测 PASS（score=15、keep=4700、非 keep=0）。
- 产物：`~/ak_verify_results/glm_subspace_skip_retest_20261109/`。
- 日志：`/tmp/glm_subspace_skip_retest_gpu{0,1}.log`。
- 安全准入：单臂启动前要求 GPU 空闲显存 ≥110000MiB 且主机可用内存 ≥100000MiB；
  不擅自杀现有任务、不绕过资源门槛。2026-11-09 09:44 两 worker 均在安全等待。
- pc32 的 GLM 权重与统一快照已从 pc33 部署；config 与 safetensors index SHA 一致。

## 当前状态

running。旧 `SELECTION.md` 的 random32 结论降级为阶段性候选，最终排名待本轮 12 臂完成。
native fullKV 参考为同 producer 已验收的 90.0；native 本身无 NFA gate，不伪称开启 skip。

## Progress

- 2026-11-09 10:23：按用户“立刻释放吧”指示，以精确 PID 停止 pc32 旧 SGLang DSA
  服务（主 PID 876854）及 pc33/pc02/pc40/pc31/pc32 已确认的陈旧队列、监控和僵尸父链；
  未触碰 GLM/Qwen 正式实验、16en watcher 或本任务等待 worker。pc32 双卡均释放至 1MiB。
- 10:23:20/10:23:21：worker0/worker1 分别通过资源准入并启动 `full_first32`、
  `full_last32`；Python PID 3031160/3031184。点火后 35 秒三重核验通过：进程存在、
  双卡各加载 57454MiB、日志权重加载正常。
- 10:30 监督：producer SHA256 复验为固定值 `802d3ed9…44e07`；两 worker 与 Python
  均存活，首两臂进入 100 请求生成阶段。audits 下两个预检 FAIL 仅因当时尚无
  `.nfa_gen` 指针，不是已运行臂审计失败，完成前不计入 12 臂 PASS 数。
- 13:16 两臂完整收官并自然切换：`full_first32` 审计 PASS（100 行、score=16、
  keep=4700、非 keep=0），`full_last32` 审计 PASS（100 行、score=30、keep=4700、
  非 keep=0）。原 Python PID 3031160/3031184 退出是单臂完成而非 worker 异常；
  两卡随后分别进入 `dual_first32`、`dual_last32`，当前各约 121GiB 显存。两个 dual
  预检 FAIL 仍是尚无 generation pointer，不能计 PASS；12 臂排名继续 pending。
- 13:50 监督：`dual_first32`/`dual_last32` 分别推进至约 45/100、47/100，双 worker
  持续生成；两个 audits JSON 仍为无 generation pointer 的预检 FAIL，不计入完成数。
  当前只有 full_first32/full_last32 两臂正式 PASS，其余十臂和最终 full/dual 排名 pending。
- 14:10 监督：`dual_first32`/`dual_last32` 分别推进至 95/100、96/100，双 worker
  PID 3025350/3025351 存活，双卡各约 121GiB 占用，仍属正常生成。完成 pointer 尚未出现，
  不把接近 100 行当作完成；待 pointer 落盘后立即运行独立审计，重点核验 dual 实际非 keep>0。
- 14:12 `dual_last32` 完成并经独立复验 PASS：100 行、score=27，gate_total=
  keep 2747 / near 1953 / far 0 / both 0，实际非 keep=1953，满足 dual 非零 skip 门禁；
  审计写入 `audits/dual_last32_independent_20261109T1412.json`。worker1 随后自然切换
  `dual_nope_uniform32`，进程、显存与权重加载/首请求日志核验正常。当前正式 PASS 增至 3/12。
- 14:14 `dual_first32` 完成并经独立复验 PASS：100 行、score=17，gate_total 与
  dual_last32 相同（keep 2747 / near 1953 / far 0 / both 0），实际非 keep=1953；审计写入
  `audits/dual_first32_independent_20261109T1414.json`。worker0 已进入
  `full_nope_uniform32`，与 worker1 的 `dual_nope_uniform32` 形成同候选配对。当前 4/12
  臂 PASS；first32 full/dual=16/17、last32 full/dual=30/27 仅为已完成臂记录，不提前汇总最终排名。
- 14:26 监督：`full_nope_uniform32`/`dual_nope_uniform32` 分别推进至 27/100、28/100，
  Python PID 3063134/3062857 与父 worker 均存活；两个锁文件经非阻塞 `flock` 实测均为 HELD，
  不存在重复 worker。两个 uniform32 审计的无 pointer FAIL 仍是启动前预检，不计入结果；
  当前正式完成数保持 4/12，未提前发布 full/dual 排名。

