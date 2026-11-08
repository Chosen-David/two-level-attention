# 第三轮AI监督报告 — 2026-10-29 23:31:46 CST

PROJECT_ROOT=/home/wangyuanshuo02/two-level-attention；工作树=/home/wangyuanshuo02/two-level-attention-near-far-skip。源冻结3e4a6885604ac3e719ee92a4906f87b532535314，阈值严格0.05，协议未改。

## 已完成与本轮实际动作
- 多轮三机SSH精确进程、GPU显存利用率、daemon、heartbeat、日志与逐样本落盘检查，证据snapshot_initial/relay/mid/final/late/handoff.json。确认pc31 rank32完成后由原daemon接力rank41，另卡rank42；pc02 GPU1接力rank40，后续rank43–47。未抢lease、未新派GPU任务、未停车、未重启E184，未用本机GPU1。
- 已完成分片的identity/输入集合/三臂完整性盘点，integrity_initial绑定680配对，integrity_closure绑定820配对，均无盘点错误；后者不是科学验收。
- collaboration启动失败后使用两个独立ephemeral Codex只读CPU会话，分别完成评分和输入/执行路径补验。独立身份、报告及日志hash见independent_sessions.json。已纠正沿用提示的旧1500分数量，实际评分2040/2040。
- 审阅现有finalize路径，确认validate→analyze→export后仍标pending independent。未提前运行汇总，未改生产源/协议/队列/阈值/监督间隔。没有实际阻塞需要修复，未重复无关回归。

## 运行进度
2026-10-29 23:31:46 CST：queue 40/64完成、8运行、16待领取、0失败；done标记41（快照读取之间及300秒收割周期可领先queue）。922/1300完整配对、2777/3900生成。较上一轮末620/1864净增302配对、913生成；本轮首787/2365净增135配对、412生成。
任务完整数Qasper152、HotpotQA152、Musique152、GovReport143、RepoBench-p323。三机SSH均成功、HEAD相同，daemon年龄179–195秒，属于300秒周期内。所有已领取任务仍为首尝试。最新worker的日志/heartbeat与GPU运行证据见snapshot_handoff.json。

## 独立质量观察（仅固定680配对）
详见[中文质量报告](QUALITY_PARTIAL_ZH.md)、[独立评分](independent_partial/REPORT.md)、[独立输入/路径](independent_inputs/REPORT.md)。746绑定文件、72源码和数据哈希核对通过；2040官方评分误差0。Qasper Far/Dual−0.957999pp，RepoBench-p−0.059524pp；子集宏Far−0.053844pp、Dual−0.059447pp。确有固定子集观测下降，但非最终AVG5，不能外推全量或声称等价/无损。
Musique107条为0降/1升/106平，均值+0.002148pp；row82仍是错误回答，不解释为多跳推理提升。GovReport98的Dual比Far单样本下降2.997513pp。
Near-only27/24480层—请求、13/680请求；Both1/24480、1/680请求，定位Musique56第11层。Both三臂评分0，不能由一个自然样本认证一般质量。

## 输入、缓存与路径证据及限制
680输入全部独立重建一致；2040输出token解码/停止条件通过；73440末步TLI回执一致。缓存220320层快照账面守恒通过。GovReport98记录8个64-token页老化迁移，固定子集另有6个Near-only层—请求记录64-token迁移；Near staging隐藏且不永久丢弃，Both仅sink/SWA。
实际路径是32维选择、完整128维K/V attention；native parity只在33个分片首条Full存布尔值，比较对象仍为TLI patch路径，不等于dense基线。独立报告保存了新增当前tokenizer资产hash，但不补造历史绑定。
未复现GPU生成/原始QK/逐步选择集合，未完整权重hash，部分消费者环境未完整记录；不认证具体kernel或性能。旧CPU边界报告经hash核对复用，不重新执行。

## 当前阻塞与下一步
剩378配对未齐，自动汇总与全量独立验收pending；无基础设施故障。保持原8卡自动接力。全量到齐后核1300/3900与自动输出，并依FULL_VERIFICATION_HANDOFF独立复算全部评分/bootstrap、门控/缓存/输入路径；单列Musique和自然Near/Both覆盖。完整报告和独立验收落盘后才可写COMPLETE.json，本轮未写。
