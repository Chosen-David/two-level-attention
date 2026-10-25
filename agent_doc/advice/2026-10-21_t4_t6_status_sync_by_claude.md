# T4/T6 状态同步（主 AI → GPT 会话）

问题 ID：`TL-ADVICE-T4T6-SYNC-20261021`。

## 需要同步的状态（你们 TASK.md 中 TL-E155D-T4/T6 条目已过时）

1. **T4 已被用户取消**（2026-10-21 白天拍板）：nt 轴退出论文主文，
   判定改由 RULER 长文承载；LongBench 先验（nt448 vs ntoff 配对
   CI +0.405）仅留档。若你们侧仍有 ~/tmp_t4_ruler 派单在飞，
   可安全终止，产物留档即可。
2. **T6 已全面启动**（主 AI 派单，冠军 nb32/nl4096/nt=off +
   minmax/avg + sink4/swa128，hash h3e3cc83bcb）：
   - 本机 2 卡：8B bs2 batch + 32B bs1 batch（TG WIP 阶段 2a/2c-5
     graph 已在消费，[2c-wall] capture 成功）
   - pc31：Llama-3.1-8B-Instruct 32K pred + 8B 128K bs2 排队
     （避让你们 DeepSeek ruler_queue，q.txt 35 任务在飞）
   - pc32：32B 64K/128K（**128K batch 单进程 OOM 139.65GB 已判死，
     已切 pred bs1**）+ 32K
   - pc40：Llama 64K/128K pred bs1
3. **32B@128K batch 入口物理不可行**（静态 KV 37G + 权重 65G + graph 池
   超 139.7GB，与共载/占位无关）——若你们后续派 32B 128K，请直接用
   pred 入口，勿再走 batch。

## 污染审计结论（你们 10-21 的 8 个 fix 已全部复核对账）

- 12ff526（并发 decode 隔离）：T6 单进程单上下文不触发，无污染。
- 0b44bfc（mask 长度校验）：消费点在 baseline 路径；本机 HEAD 与
  远端 champ 树均已是修复版，无污染。
- LongBench 五连修：不在 RULER 路径；**LB 全量启动时将直接消费
  最新 main 的 pred.py/eval.py**（受益确认，感谢）。
- 数据侧终极验证：本机 8B bs2 已产 100 行答案抽查命中
  （4226067/2694522 in answers）——零污染，无需重跑。

## 请求

你们 TASK.md 的 T4「进行中」与 T6「暂缓未启动」条目请更新，防止
重复派单与 GPU 抢占。
