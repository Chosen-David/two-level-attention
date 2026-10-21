# E155d 任务链（TASK.md）——主 AI 维护，每轮监督更新

> 用户指令（2026-10-19 19:1X-19:3X）：网格跑完 → 立刻写 ablation 论文部分 →
> 敲定论文主方法 → 全量任务测试 + baseline 对比 → 修复性能浪费点。
> 完成后及时推送远端 main。用 TASK.md 维护任务链并推进直到全部完成。

## 当前阶段：T2/TDS 收官 → **T6 暂缓等 TG 升级**（用户 10-21 06:4X：
## 先做库 batch 化 + CUDA Graph，性能提速确证后再跑论文主方法全量——
## bs=1 单卡利用率不高，全量测试等升级后代码库跑）

**【T2 终版判决 06:19（87/87 格，零 eval 失败，零 SHA 冲突）】**
- **冠军 nb32nt448nl4096 AVG5=48.668**（vs CTRL 单池 47.462，**+1.21**）；
  top-3 = nb32nt448/nt320/nt128nl4096（48.668/48.512/48.468），配对 CI
  全含 0 → 前三名噪声带内等价，但**全部同族（nb32 nl4096）**，几何结论稳健。
- **故事分支判定**：nt 甜区赢（best_per_nb 的 nb16/32/64 最优格全 nt448；
  nt-vs-off 70 对中 42 对正，最大 +0.5）；nl 单调**否决**（档均值 1024:
  47.96 / 2048: 47.82 / 4096: 47.82 / 8192: 47.25——nl8192 塌 -0.7）；
  原位锚 (8,320,8192) **否决**（不在榜首）。
- **几何规律**：best_per_nb 全落在 **nb·64 = nl/2**（near 区半填充）对角
  线：nb16→nl2048、nb32→nl4096、nb64→nl8192，最优 nt 恒 448。
- 预算恒 ~1021.5（K2=1024 同预算口径成立）。判决文件：
  exp/trace/results/e155d_grid_verdict.json（06:19 终版）。
- 收官清理：五台远端 + 本机网格进程全清零（gpu=0 × 6 机核实）。

**【TDS DeepSeek T7 fix 审查通过（06:1X，三重独立复跑）】**
①E127 门禁独立复跑 **15 PASS / 0 FAIL**（含 fine_gather 候选 vs 全宽
mask 逐位一致 80/80 步、cand_ulp=0）；②batch 化门禁独立复跑 **26 PASS
/ 0 FAIL**（G1-G4 全过）；③E133c mavg 开关对照——TLI_FINE_GATHER 0/1
两跑 full-vs-tail 差异轮廓**逐步逐位全等**（8804/8236/… 24 步全同）→
候选化对存量路径零扰动。代码审查：E156c 重建修复（cand_idx 存在即
无条件 scatter，修 R_max==S_width 形状守卫漏网）+ fail-closed 门控
（σ/ccluster/投影/moba/per_q_head 臂全回退全宽）+ _select_l1_blocks
单源防口径漂移——设计正确。**fix 采纳，WIP 已提交入库（DeepSeek 轨
提交），T6 消费含候选化代码库。**诚实注记：B=1 无 wall 收益（全宽路径
在 H20 已 flat ~0.9ms），候选化真实价值 = batch 化地基 + 消全宽物化。

**监督快照 00:39 + 🔴 主树 WIP 污染事故处置**：唯一 tag 并集 **86/87 过门**
（85 geo + CTRL；nt160 两目录轴外残留不计）。**事故**：DeepSeek 会话在
主树改 T7 代码，18:42 窗口 tli_indexer.py 短暂语法破损 → 本机链（从
主树逐任务起新进程）吸坏代码，nb8ntoffnl2048/nb16nt128nl2048/
nb4nt128nl1024/nb16nt448nl1024 本机臂 FAIL——**全部被 .187 等对向
双臂自愈**（完成集核验）；后续本机在飞 nb16nt640nl1024 用 WIP 代码
冗余重跑（.187 已有干净副本）→ **已杀本机两条链（bash+python 一轮
同杀）+ 清污染目录**。**缺口格 nb32ntoffnl2048 = 主树 WIP 事故唯一
未自愈格**：钉版 worktree ~/e155d_final@ecfbd4f（与 .187 同树）GPU0
补跑起跑 00:41（hotpotqa 4.4s/it），ETA ~02:10-02:30 收官；.187 L1
队列仍兜底（SKIP 幂等）。**教训（坑）**：监督 FAIL 扫描须含链日志
（/tmp/e155d_grid_l*.log），只扫产物目录会漏 rc=1；主树生产代码在
网格收官前被并行会话改动的风险已实证——收官前一律钉版树补跑。
**监督快照 23:39（历史）**：83/87 格过门（唯一 tag 并集口径；
.187 24/pc02 22/本机 10/pc31 12/pc32 12/8100 11 完整，含跨机重复臂）。
**剩 4 格真在飞**（全 nl2048，多带对向双臂处于 repobench 尾段）：
nb16nt320nl2048 / nb16ntoffnl2048 / nb32nt128nl2048 / nb32ntoffnl2048
——**ETA ~00:30-01:30 收官**。收官即跑终版 T2（收割+打分+配对 CI）。
**监督快照 20:39 + T2 冒烟（历史）**：74 格打分
+12 在飞中段）。**T2 收割管线全链冒烟通过**：六源收割 OK、74 格打分
零 eval 失败、**SHA 冲突零**（跨机重复格逐文件 bit-identical）、CTRL
单池锚 47.462。部分快照 top-1 = nb32nt448nl4096 AVG5=48.668（+1.2 vs
CTRL），nl4096 档霸榜前 10——初步信号指向 nt 甜区 + nl4096；终版以
87 格判决为准（top-3 配对 CI）。末波格起点晚（18:30-19:30 释线），
**ETA 修正为今晚 ~23:30-01:00**。
**监督快照 19:39（历史）**：76/87 格完整（本机 10 /
pc02 20 / 8100 10 / pc31 10 / pc32 10 / .187 16），在飞 21 线（含对向
消费重复格），速率 +9 格/h。**ETA 收敛今晚 ~21:00-21:40**。收官后序列
（用户 17:5X 指令）：T2 打分判决 → **TDS 审 DeepSeek T7 fix**（L2
候选化逐位对拍门禁）→ 对则 **T6 全量（Qwen3-8B + Qwen3-14B，后者已
就位）** + 四 baseline 对比。
**监督快照 18:39（历史）**：67/87 格完整（本机 10 /
pc02 16 / 8100 / pc31 / pc32 / .187 各 8），实际在飞 21 格（含对向消费
重复格）；速率自 13:02 起 ~11.4 格/h（大量格处 4/5 尾段集中收割），
**ETA 提前到今晚 ~21-23 点**。本机 nb32nt160/nb4nt160nl8192 两目录核实
为 10-19 旧口径残留（派单文件零命中），分析器轴外排除正确。T2 收割
入口不变：`bash exp/trace/e155d_grid_collect.sh && .venv/bin/python
exp/trace/analyze_e155d_grid.py`。**TG（用户 10-20 指令）**：CUDA Graph +
大 bs 升级由并行会话实现，本链转协调位（968cb59：网格在用树保护 +
逐位/锚点门禁后才切 + T3/T4/T6 消费升级后代码库）。
**监督快照 13:02（历史）**：36/87 格完整 + 在飞格全部
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

- [x] **T1 网格监督**（已完成 06:0X）：87/87 唯一 tag 并集过门（缺口格
      nb32ntoffnl2048 本机钉版补跑 05:19 过门收官）；六机进程清零核实。
- [x] **T2 网格打分 + 判决**（终版 06:19）：87 格零 eval 失败零冲突；
      判决摘要见顶部；纪律执行：top-3 配对 CI 全含 0（不按点估计拍板）
      → 冠军几何 = **nb32nt448nl4096**（点估计第一 + 同族稳健：nt448
      主导 best_per_nb、nl4096 档最强、+1.21 vs CTRL）。
- [ ] **T3 nl=全mid 退化端点**：top-2 = nb32nt448nl4096 / nb32nt320nl4096
      （CI 等价，几何叙事用），各 1 格 nl=全 mid 退化端点补跑。
- [ ] **TG harness CUDA Graph 化（用户 10-20 14:3X 指令；10-21 06:4X 提级
      为 T6 前置硬门）**：eval harness decode 步 Graph 化（StaticCache 静态
      槽位 + 长度分桶 capture + host 算术 GPU 化，E134-pre StaticCache +
      E139 graph 门禁先例移植）+ 大 bs 化，**实现由另一 AI 会话承担**，
      地基已就位：DeepSeek 阶段 0（E156 候选化 = batch 省算前提，B×R vs
      B×T）+ 阶段 1（HF batch>1 decode 逐行核心 + pred_ruler_batch 入口
      + BatchCtx）已入库（09735a2，门禁 15/15+26/26）；阶段 2（graph
      capture + 批量张量化）见 research/docs/TLI_batch化 阶段1交付 §衔接。
      本链只做协调：①网格已收官，六机全空闲可自由升级（主树 a28d12d
      即升级基线）；**10-21 主 AI 审计轮**：TG WIP（阶段 2a 固定宽度
      plumbing，base.py+tli_indexer.py）审计通过——valid_length kwarg
      全链贯通、inc 硬门 S_now>S 恒假固定宽度恒全量（无 stale）、
      S_real 单源 min(_vl,S_width)、compute_mask 与 _select_l1_blocks
      双副本对称改写、E127 15/15 复跑零回归；**修一处硬 bug**：
      base.prepare_mask 无条件传 valid_length kwarg → quest/tia/twi
      首调即 TypeError（已修：None 时省略 kwarg）。sglang 侧 WIP N1
      哨兵修复审计通过 + 补齐 i_g/skip_far 两处整行 topk 同族饥饿
      泄漏守卫（哨兵 S 协议与 B02/#58 统一）。两处修复均在 TG 会话
      WIP 文件内未提交，由其会话一并集成；**同轮第二批 WIP（E155e
      缺省几何翻转：三 --tli_near_* 缺省隐式注入冠军 nb32/nl4096/
      ntoff，SUPPRESS+hasattr 三元判定）审计**：注入构造逻辑自洽
      （隐式 ≡ 显式 nb32ntoffnl4096 逐位等价、显式 None=legacy 意图、
      fail-closed 互斥保留）；**又修一处身份碰撞级 bug**：
      resolve_treatment_snapshot 读 args 原始态看不见 __init__ 里的
      隐式注入 → 新默认与旧默认（无参数=单池老逻辑）manifest 完全
      同值 → treatment hash + _h<hash10> 文件名尾段碰撞（E155c 同类）。
      已修（info.py manifest 镜像，gating method==tli）：五场景对拍
      全过——隐式默认 ≡ 显式冠军臂 manifest_json 全等（同一有效
      treatment 同一身份）、③α0β0 ④无method ⑤pages=None 三路全
      分离、显式几何臂 manifest 逐位不变（网格/历史锚点身份稳定）。
      注意：nt 默认 off 是用户 10-21 在升级会话的拍板（nt448 点估计
      略优但取 off，长上下文去轴判决见 T4 预登记）——T4 RULER 64K
      结果将终审此默认值；②逐位
      门禁（graph vs eager 逐例对拍 + 大 bs vs bs1 锚点对拍）过了才切；
      ③**收益落点 = T3/T4/T6 派单用升级后代码库跑**（T6 三模型全量 +
      四 baseline 对比为最大消耗方；大 bs 口径与 E125/E147h 锚点可比
      性须升级会话出对拍证据）。
- [ ] **T4 nt 轴可分性 RULER 实验（10-21 用户定稿方案，预登记；TG 升级
      完成后跑——用户「等全面改动完后跑」）**：固定 nb32/nl4096，
      nt∈{128,320,448,640,off} 五臂 + 交互对照 nb64/nl4096/ntoff +
      单池 CTRL = 7 臂，RULER 64K 全 11 任务（n=100）先跑；前 2-3 名
      再上 128K 代表任务（single_2/mk1/vt，E119 先例）。**判定规则
      （预登记）**：主比较 = nt=off vs nt=448 在 64K 的配对 CI；CI 下界
      > -0.3 → 去 nt 轴（主线免调参自由竞争，少一个超参）；CI 下界
      < -0.3 → 保留 nt=448（写宽甜区 + nb 交互）。nt∈{128,320,640}
      只作甜区宽度证据不参与冠军竞争。**LongBench 先验（10-21 补算，
      网格数据零成本）**：nt448 vs ntoff 配对 CI **排除 0**
      （+0.405 [+0.099,+0.786]）；且 **ntoff vs CTRL 含 0**
      （+0.803 [-0.194,+1.839]）而 nt448 vs CTRL 排除 0
      （+1.209 [+0.177,+2.265]）→ **短上下文上 nt 轴承载全部统计
      显著性**，去轴判决只能由长上下文 RULER 出——64K 若 ntoff vs
      nt448 含 0 且 ntoff vs CTRL 分离，则「分区价值在长上下文才成立
      + 免 nt 调参」双故事成立，论文叙事更简。
- [ ] **T5 ablation 论文部分**：网格结果写入论文 ablation 节
      （TLI_paper_en.tex / TLI_paper.tex 双语镜像）。
- [x] **TDS DeepSeek fix 审查（用户 10-20 17:5X 指令，已通过 06:1X）**：
      E156 精筛候选化（阶段 0）+ TLI batch decode（阶段 1）三重独立复跑
      全过（E127 15/15 + batch 26/26 + E133c 开关对照轮廓全等），代码审查
      E156c 重建修复正确。WIP 已按 DeepSeek 轨提交入库；详见顶部 TDS 段
      与 research/docs/{E156,TLI_batch化,论文同模型对齐验证报告}_*by_deepseek.md。
- [ ] **T6 主方法敲定 + 全量测试（用户 10-21 06:4X 改序：暂缓，等 TG
      升级落地提速确证后再跑）**：冠军几何
      **nb32nt448nl4096（+1.21 vs 单池）**
      → **主文 Qwen3-8B + Qwen3-14B 全量**（Qwen3-8B 全量主战场：LB
      16+5 + RULER 三档；Qwen3-14B 组件轴：本地 ~/Qwen3-14B 已就位
      ——40 层/40 头/8 KV 头，GQA kv=8 与 8B 同构 TLI patch 直接兼容；
      旧 32B/Llama 落袋锚保留可用，主表阵容以今晚指令为准）
      → **vs 四 baseline（JSSA/FASA/Quest/AsyncTLS）对比表更新**
      （既有战况：JSSA 58.7 四家最强 / AsyncTLS 50.09+64K 33.88+128K
      37.47 / Quest 三档全崩 / FASA LB-EN 33.40——四家 RULER/LB 逐档
      补齐到同口径，冠军几何同预算行）。**派单已预置未启动（06:4X 用户
      叫停于起跑线前，零进程零清理）**：/tmp/t6_lb_{local,8100,pc31,pc32,
      pc02}.sh（LB 21 任务 12 GPU 分配 + SKIP 幂等）+ .187 RULER 三档
      8 线方案；六机 21 任务数据/RULER 数据/模型普查全过（14B 在
      pc02/8100/pc32+本机）。升级后重跑即用（注意按新 harness 入口
      改命令行）。
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
