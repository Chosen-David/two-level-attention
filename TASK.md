# E155d 任务链（TASK.md）——主 AI 维护，每轮监督更新

> 用户指令（2026-10-19 19:1X-19:3X）：网格跑完 → 立刻写 ablation 论文部分 →
> 敲定论文主方法 → 全量任务测试 + baseline 对比 → 修复性能浪费点。
> 完成后及时推送远端 main。用 TASK.md 维护任务链并推进直到全部完成。

## 当前阶段：E155d 三维网格在飞（12 卡 / 4 机）

**监督快照 09:08**：四机 12 线零 FAIL（19:54 起恒定）。**25/87 格完整** +
12 格在飞 repobench 尾杆（本机 nb64nt448nl8192 已收）。
**🔴 09:06 .187 加盟（用户「全部空闲 GPU 用上」）**：8 卡全空闲（vendor relay
已退场、E153b 已收官、0 进程核实）→ worktree ~/e155d_tree@ecfbd4f +
系统 python 4.56.2 + 数据/模型已有 → 8 线起跑，**对向消费剩余 62 格**
（nl 升序 1024→8192，与 12 线正序对向；SKIP 幂等兜底，先完成方收割）。
12+8=20 线 → 剩余 62 格 ETA ≈ 62×4.5h/20 ≈ **14h → 今晚 ~23:00 收官**
（原 12 线 ETA 10-21 凌晨，提前 ~5h）。
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
- [ ] **T7 性能浪费点修复**：E133c 计划（A1-A4/A6，A5 已被 l2_quant=false
      默认淘汰）+ E152 launch-bound 归因落地；kernel microbench 与 e2e 双口径。
- [ ] **T8 推送远端 main**：全部判决/论文/代码提交后 origin push
      （含历史挂账：ecfbd4f 等）。

## 挂账 / 注意

- 8100 旧收割目录已改名 pred_legacy_ratio_l1024_dontuse（旧比例口径，隔离）。
- pc02 untracked 备份：~/pc02_untracked_backup_20261019/（vendor 轨已在 027566b 入库）。
- pc14（.187）8 卡 vendor 在飞勿动。
- wys E153b 旧口径消融数据待网格判决后统一处置。
- kill 顺序坑：bash 循环链须 bash+python 一轮同杀，单杀 python 会推进循环重生孤儿。
