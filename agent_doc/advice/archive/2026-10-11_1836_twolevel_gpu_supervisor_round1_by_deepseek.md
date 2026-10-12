# two-level GPU 监督器 Round 1（by deepseek）

- 时间：2026-10-11 18:36
- 触发：cron 244242d1（每 2h）
- 结论：**未发现新 bug**；在飞链正常；**存在大量空闲 GPU** → 登记派单（S-T023 已入 TASK.md）

---

## 1. GPU 占用表

| 池 | 卡 | 占用 | 在跑任务 |
|---|---|---|---|
| 本地 | 0,1 | **全空**（1MiB/0%） | 仅 CPU 态 `test_sparse_prefill_impl.py`（sparse prefill 功能双跑测试） |
| 10.238.9.187（wys_8xh20/.187） | 1/4/5 | 22/25.6/22GB，60/72/100% | **S-T022/E125** 3 臂 repobench-p：mminmax(.875,.875,.125) / aavg(.125,.125,.375) / cclustersim(.125,.25,.125,sim0.9) |
| 同上 | 0/2/3/6/7 | **5 卡全空** | — |
| 33.32.33.24（wys2-8xh20/.24） | 1/3/4/5/7 | 63-73GB，100% | **S-T019/E116b** RULER 128K：aavg(0,0)×{niah_multivalue,niah_multikey_3} + mavg(.25,.125,.625)×同 2 任务 |
| 同上 | 0/2/6 | **3 卡全空** | — |
| 10.238.139.251（2xh20-8100/.251） | 0,1 | **2 卡全空** | — |

## 2. 在飞任务与脚本（ps 核验）

- **E125（.187）**：`python -u -m benchmark.LongBench.pred --task repobench-p --method tli
  --tia_level1_topk 128 --tia_level2_topk 1024 --tia_level2_cmp_ratio 4
  --tli_enable_layer_skip false --output-dir /home/wangyuanshuo02/tmp_e125_screen`
  参数与 S-T022 派单协议（E123 审计修复版：086 前缀归一 + 008 failmask）一致。3 臂在跑，
  第 4 臂 cavgsim 未见于 ps（可能已完成或待起）——**下轮核验 E125 四臂是否齐**。
- **E116b（.24）**：`pred_ruler --context_length 131072 --yarn --max-num 100
  --pred_postfix _1024` 落 `exp/results_ruler/e116b_full_Qwen3-8B/{aavg,mavg}`，
  与 S-T019 三件套口径一致。
- **本地**：`/tmp/e125_tail_wt/test_sparse_prefill_impl.py`（+ `-O` 双跑）＝TLI_SPARSE_PREFILL
  门控功能测试，非生产链，不动。
- 锁文件：`e119_ruler128k_formal_*.lock` 残留于 17:21；在飞进程落袋是 `e116b_full_Qwen3-8B`，
  e119 锁为已收口任务的旧文件（E119 64K 已收口、128K 数据旧），**非活性锁**，不需处理。

## 3. bug 审查结论

- 在飞进程参数与任务链/TASK.md 描述逐项一致，未发现参数偏差。
- 代码侧 Round 1 已登记（2026-10-11_1818 报告）：GPT Static cluster tail coverage（cluster
  0,0）、F6、near kmeans guard 缺失。**本轮无新增 bug**。
- 注意：E125 若含 cluster 臂（cclustersim 在跑），其 producer/consumer 边界是 sim_greedy
  路径（cluster(0,0) 用 kmeans），当前在跑臂走 `_update_far_greedy`/`_update_near_cluster`
  sim_greedy 分支——tail coverage bug 属 kmeans (0,0) 分支，本轮 3 臂不直接触发，但
  **冠军若落 cclustersim 系仍需审**（sim_greedy 分支有独立边界处理，见 1818 报告③）。

## 4. 派单（用户 18:2X 指令 + 空闲资源）

**新任务已登记**：S-T023「DeepSeek 专属开源 baseline 复现」入 TASK.md，
思路/矩阵/执行阶段见 task_details/S-T023.md。

- 空闲可派：本地 2 + .187 5 + .24 3 + .251 2 = **12 卡**。
- **立即可做（不占已调度链）**：
  - 阶段 0 收尾：quest raft / ClusterKV 3rdparty 子模块补齐、两独立 venv `uv sync`、
    Qwen3-32B / DeepSeek-V2-Lite-Chat 下载 → CPU/下载任务，可派本地。
  - 阶段 1 C 轨 sanity（Quest 官方复现 LB 单任务）→ 1 卡，可派 .251 或本地。
- **等空窗**：阶段 2 B 轨移植、阶段 3-5 全量三件套 → E116b/E125 收割后复用派单模板。

## 5. 下一步

1. 下轮（~20:37）核验 E125 四臂齐没齐 + E116b 进度；
2. 启动 S-T023 阶段 0（子模块/venv/模型下载，派本地）；
3. 阶段 1 找 1 张空闲卡跑 Quest C 轨 sanity；
4. 用户 18:2X 附加要求：完成后把复现所需代码/数据/权重同步到其他 GPU 池（rsync + md5 核验）。

— by deepseek
