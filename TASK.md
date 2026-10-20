# E155d 任务链（TASK.md）——主 AI 维护，每轮监督更新

> 用户指令（2026-10-19 19:1X-19:3X）：网格跑完 → 立刻写 ablation 论文部分 →
> 敲定论文主方法 → 全量任务测试 + baseline 对比 → 修复性能浪费点。
> 完成后及时推送远端 main。用 TASK.md 维护任务链并推进直到全部完成。

## 当前阶段：E155d 三维网格在飞（12 卡 / 4 机）

**监督快照 13:02**：零 FAIL 恒定（19:54 起）。**36/87 格完整 + 在飞格全部
推进到 800 行档**（12 格 4/5 任务、nl1024 对向消费线尾段）。速率符合
~1 格/h/线节奏，ETA 维持**明晨 ~02:00**。T2 管线已预建冒烟全通
（ac482cd）：收官后 `bash exp/trace/e155d_grid_collect.sh &&
.venv/bin/python exp/trace/analyze_e155d_grid.py` 即出终版判决。
**监督快照 12:01（历史）**：36/87 完整 + 21 在飞；速率 ~3.7 格/h。
**监督快照 09:08（历史）**：25/87 格完整 + 12 格在飞 repobench 尾杆。
**🔴 09:06 .187 加盟（用户「全部空闲 GPU 用上」）**：8 卡全空闲（vendor relay
已退场、E153b 已收官、0 进程核实）→ worktree ~/e155d_tree@ecfbd4f +
系统 python 4.56.2 + 数据/模型已有 → 8 线起跑，**对向消费剩余 62 格**
（nl 升序 1024→8192，与 12 线正序对向；SKIP 幂等兜底，先完成方收割）。
**共居双进程试点判决（09:2X，用户问「提高利用率跑更快」）**：GPU0 双进程
实测各慢 2.07×（4.31→8.9s/it），聚合吞吐 0.97× 零收益——bs=1 decode 瓶颈
在 HBM 带宽+launch 不在 SM，利用率指标在此负载下误导性；20 线已是当前协议
最优并行，剩余加速空间在 T7 kernel 侧（E133c）。试点格 nb64nt256nl4096
hotpotqa 跑完即止，L7 后到 SKIP 兜底。

- 设计（用户定稿）：nl∈{1024,2048,4096,8192} × nb∈{4,8,16,32,64} ×
  nt∈{128,256,320,448,640,off}；约束构造期强制（nb·64≤nl、nt≤nb·64、far≥0）。
  86 geo 格 + 1 CTRL 单池对照 = 87 格；载体 LongBench AVG5（n=200/500，
  与 E125/E147h 同口径，历史锚点直接可比）。
- 树 ecfbd4f（bee 已推，origin 推迟待 paper 会话提交边界）。
- 12 线分工：本机 L00/L01（CTRL 在 L00）、8100 L02/L03、pc02 L04-L07、
  pc31 L08/L09、pc32 L10/L11。输出各机 ~/tmp_e155_grid/pred_<tag>。
- **口径铁律：transformers 4.56.2 全局统一**（本机 pylibs_tf512=5.12.1 已弃用，
  pc31 pylibs_tf512=5.12.1 同陷阱；pc31/pc32 走 /home/fluentllmenv）。
- 起跑时间：19:35（本机/pc02）、19:21（8100 重平衡后）。
- 预计 ~7-8 h/线（E153b 经验 ~1 卡时/格）。

## 任务链（顺序推进）

- [ ] **T1 网格监督**（cron ~50min）：四机进度/FAIL 诊断/SKIP 幂等；
      全部 87 格 × 5 任务行数门通过后进 T2。
- [ ] **T2 网格打分 + 判决**：LB eval 打分 → AVG5 排序 →
      exp/trace/results/e155d_grid_verdict.json。纪律：前 3 名进配对 CI，
      不按点估计拍板；预登记故事分支（nt 甜区赢/ntoff 赢/原位赢/nl 单调）。
- [ ] **T3 nl=全mid 退化端点**：top-2 各 1 格（网格判决后才知道是谁）。
- [ ] **T4 RULER canary**：仅 AVG5 前 10 + 边界格（不再逐格）。
- [ ] **T5 ablation 论文部分**：网格结果写入论文 ablation 节
      （TLI_paper_en.tex / TLI_paper.tex 双语镜像）。
- [ ] **T6 主方法敲定 + 全量测试**：冠军几何 → 全量任务（LB 16+5、
      RULER 三档、32B/Llama 迁移）→ vs 四 baseline 对比表更新。
- [ ] **T7 性能浪费点修复**：E133c A1-A4/A6 **实现已全部在 main**（逐位
      对拍硬门已过；A5 被 l2_quant=false 默认淘汰）+ E152 launch-bound
      归因落地；**剩余未闭合项 = .187 独占窗口稳态 bench 同机对照**
      （tail 臂 ≤ full 锚点 1.6884ms 判据）+ 判决 JSON——等网格收官
      释放卡位。问题③ 判定已闭环（契约 084f6b5 与 T7 零冲突，见
      agent_doc/advice/2026-10-20_t7_vs_prefill_mask_contract_adjudication.md）。
      **10-20 双仓审计轮（用户指令）**：e113b kill-switch 绕过已修
      （d27009b：SGLANG_TLI_GREEDY_KERNEL=0 提到 v2 分支前；生产零扰动，
      e113b GPU 复跑等网格收官）+ T7 stub 块粒度修正（06efa4c，并行会话
      代审）。**审计收官（sglang 侧三役全落）**：①几何语义审计 →
      boundary_bypass 移植（39847a7fe，E155b 新口径三消费点 + host 冒烟
      全 PASS）；②backend/kernels 只读深审（20 项发现）→ **15 修 5 缓行**
      （5ba097eb3：B1 兜底分支双重累加 🔴 正确性 / B2 行回收过回收 🔴
      性能塌方 / B4-B7 防线 + P1-P4/P6/P7/P9/P10 冗余消除；P5/P8/P11-P13
      缓行随 E133c bench 验证）——双提交已推 sglang origin main。
      **三真问题
      裁定：全部无需重跑**（网格 Geo-only 走 reference 默认路径，
      TLI_PREFILL_BATCH_SELECT / TLI_PREFILL_RUNTIME=candidate 均未设，
      20 线零 FAIL + E155c 逐位门 + sha 去重三重兜底；sglang 仓与网格
      仓互不相扰）。
- [ ] **T8 推送远端 main**：全部判决/论文/代码提交后 origin push
      （含历史挂账：ecfbd4f 等）。

## 挂账 / 注意

- 8100 旧收割目录已改名 pred_legacy_ratio_l1024_dontuse（旧比例口径，隔离）。
- pc02 untracked 备份：~/pc02_untracked_backup_20261019/（vendor 轨已在 027566b 入库）。
- pc14（.187）8 卡 vendor 在飞勿动。
- wys E153b 旧口径消融数据待网格判决后统一处置。
- kill 顺序坑：bash 循环链须 bash+python 一轮同杀，单杀 python 会推进循环重生孤儿。
