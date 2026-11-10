# [TL-RULER-DYNAMIC-GATE-SCAN-20261110] RULER 三长度 dynamic-gate skip 扫描

Task-ID: TL-RULER-DYNAMIC-GATE-SCAN-20261110
Date: 2026-11-10

## Plan

### 目标与验收

用户要求：“你现在先用空闲卡继续扫描完这些各个的skip情况完成后推送到远端，我希望32K,64K,128K的都扫描完”。

冻结范围：Qwen3-8B，RULER 11 个任务，32768/65536/131072 三档，每任务只取原 quick-scan 协议的 `sample_index=0`；dense prefill 后读取首个 dynamic-gate 决策，每层记录 near/far/both skip、完整 profile 和 decision。共 33 个 task×length cells、每格 36 层，共 1188 层记录。不是全样本质量评测、生成评测或性能测量。

完成条件：33/33 cells 完整；输入、源码、配置、模型、环境与原始结果身份齐全；独立脚本重新计算逐层 decisions 与汇总并核验哈希；只提交本任务产物并推送远端 main。

### 实现方法

- 源码固定为 `7798ddb6569071d1238ca21e7ddf925a18d8dcd5`，pc02 `/tmp/ruler-gate-scan-7798ddb6-20261110` detached worktree；生产时不修改仓库源码。
- gate：`epsilon=0.05, rows=256, statistic=mean, mode=dual`；NFA 几何与选择器由固定提交的 `nfa_args()` 显式参数产生。
- pc02 四张实测空闲 H20-3e，任务按排序后 `index % 4` 固定分片；每卡一个串行驱动，顺序为 128K→64K→32K，每任务只属于一卡。
- 每片原子写入、断点续跑；记录 receipt/cache 一致性和由 profile 重算的 decision 一致性。初次按真实 `cache.freeze()` 生产时，32K 的 far-skip 请求触发固定提交 `paged_region_cache.py:685` 的 `resident page accounting is inconsistent`。单请求独立进程仍复现，故不是跨请求污染。最终统一为 `decision-only-no-reclaim`：仍调用真实 `prefill_frozen_request`，只在其计算完 profiles/decisions 后跳过物理页回收并写入 cache skip state；不做 decode，不宣称 freeze/decode 正确性。所有 33 格按该模式重采，避免混合协议。
- 既有 `/tmp/ruler{32k,64k}_dynamic_gate_scan_current.json` 是 11/11 候选，但缺运行时 source-diff 身份，决定为 `rerun`：本轮统一 clean 源码重采，旧文件只作交叉核对，不作为主数据。

## Progress

### 进度与证据

- 2026-11-10：检索确认远端没有三长度全任务 quick-scan 汇总；本机有 32K/64K sample0 候选，128K 未找到。pc40/pc31/pc32 均有其他 compute PID，未抢占；pc02 四卡各 1 MiB、无 compute apps。
- pc02 固定提交对象、Qwen3-8B、三档 RULER 数据和 Python `torch=2.8.0+cu128, transformers=4.56.2` 导入检查通过；隔离 worktree HEAD 正确且 status 为空。
- 生产脚本 `/tmp/ruler_dynamic_gate_scan_20261110.py`，最终分片驱动 `/tmp/run_ruler_dynamic_gate_decision_only_20261110.sh`，最终远端输出 `/tmp/ruler_dynamic_gate_decision_scan_20261110/`。早期真实-freeze 输出保留在 `/tmp/ruler_dynamic_gate_scan_20261110/`，不进入主数据。
- 最终四卡父 bash PID 为 3198531/3198532/3198533/3198534；每卡每请求一个独立 Python 进程，任务仍为排序后 `index % 4`。最终点火后首批 128K cwe/fwe/multikey_1/multikey_2 均完成，skip 列表与早期真实-freeze 对应格一致；四卡进程链活跃。
- 2026-11-10 收割时 33/33 格全部完成，无 traceback，生产进程均退出，pc02 四卡显存均释放。33 个原始 JSON 与对应日志已分别归档到结果目录 `raw/` 和 `validation/logs/`，五个生产/审计/打包脚本归档到 `scripts/`。
- 独立只读审计 `scripts/audit_and_summarize.py` 不导入 producer helper，重新哈希 33 个输入文件、首行和 input 字符串，逐层重算 1188 个 near/far decision、skip 列表、交集、极值与汇总；结果为 `usable-with-scope`，`errors=[]`。单 Agent 约束下这是独立实现/独立进程检查，不冒充第二位人类 reviewer。
- 真实-freeze 与 decision-only 的 22 个完整 64K/128K 同格交叉核对：`skip_mismatches=[]`，`max_abs_profile_mean_delta=0.0`。旧 32K/64K 候选与新数据各 11 格的输入身份和 skip 决策也一致，profile mean 最大绝对差均为 `1.1920928955078125e-7`；旧文件仍只作辅助交叉核对。

### 最终观测与边界

| 长度 | far-skip 任务数 / 层决策数 | near-skip 任务数 / 层决策数 | both-skip 任务数 / 层决策数 |
|---|---:|---:|---:|
| 32768 | 5 / 9 | 0 / 0 | 0 / 0 |
| 65536 | 0 / 0 | 8 / 21 | 0 / 0 |
| 131072 | 0 / 0 | 11 / 121 | 0 / 0 |

- 32K 的 far-skip 只落在第 11、14 层；64K 的 near-skip 涉及第 7、25–33 层中的若干层；128K 每任务均有 near-skip，第 7 层为 11/11 任务，第 30 层也为 11/11 任务。逐任务层列表见 `derived/summary.tsv`，完整 36 层 profiles/decisions 见 `raw/`。
- 结果目录：`agent_doc/results/ruler_dynamic_gate_scan_20261110/`。`record.json` 与 `manifest.json` 均为 `usable-with-scope`；manifest SHA256=`03f821c979a05fd0ccd9eccfbf2843c94ec2bb486ec56e4d578305c86486f02c`，46 项 artifact 哈希复核通过。
- 范围限定：每任务仅 `sample_index=0`、dense prefill 后首个 gate、decision-only/no-reclaim，无 decode 或生成。不能据此宣称全数据集 skip 均值、生成质量、物理 freeze 正确性、显存收益或性能收益。

### 收官状态

测量、独立审计、汇总和哈希打包均完成；只发布本任务文档、叙事索引与结果目录，不带入 AsyncTLS、NFA 性能优化或其他 dirty 改动。
