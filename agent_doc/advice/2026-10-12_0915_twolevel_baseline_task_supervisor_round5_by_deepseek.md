# twolevel baseline 任务监督 round5（by deepseek）

- 时间：2026-10-12 09:15
- 范围：S-T023 baseline 复现（机器迁移恢复 + 32B fullkv 缺口补派 + 全资源并行化）
- 结论：**机器 IP 迁移确认（旧 .24→33.32.33.24、旧 .187→10.238.9.187，端口 8416 密码登录），网络异常根因解除；round6 全部待验证项解除（qmsum 200 行非假成功 / repobench-p g01 500 行完成已 kill 主批浪费 / passage_retrieval_en 200 行）；32B quest 15/16 只剩 gov_report；32B fullkv 完全缺口 → 12 卡并行补派。**

## 1. 机器迁移确认（round6 网络异常根因）

- 用户提供新地址：旧 .24→`33.32.33.24`（wys2-8xh20）、旧 .187→`10.238.9.187`（wys_8xh20）、.251→`10.238.139.251`（2xh20-8100），端口全部 8416，密码登录（sshpass，非免密）。
- 8 卡机 .24/.187 状态完好，在飞进程未丢失（机器迁移而非重启）。

## 2. round6 待验证项全部解除

| 项 | 结论 |
|---|---|
| .24 qmsum（08:40 done rc=0）| **200 行完成，非假成功** ✓ |
| .187 repobench-p 主批 | g01 批已 **500 行完成** → kill 主批 2225101（GPU7）释放 1 卡 ✓ |
| .187 passage_retrieval_en | g01 批已 200 行完成 ✓ |

## 3. 32B quest：15/16，只剩 gov_report

- 完成 15：本地 narrativeqa/qasper/passage_count + .24 2wikimqa/qmsum/trec/triviaqa + .187 主批 8 任务 + g01 repobench/passage_retrieval_en。
- gov_report 双跑在写：.24 GPU1（3969636）+ .187 GPU3（2223566），预计 1h 内完成。
- 完成后 → 合并 pred_qv_1024_16{,_g01,_24}/ + 本地 → 086 best-file → eval 打分。

## 4. 32B fullkv 完全缺口 → 12 卡并行补派（本轮主动作）

- 发现：Qwen3-32B 目录下**无 fullkv dense 锚点**（32B quest 对比需要）。
- 派单 12/16 任务，12 卡并行：

| 机 | 卡 | 任务（TS） |
|---|---|---|
| .24 | GPU2/3/4/5/7 | narrativeqa/qasper/multifieldqa_en/hotpotqa/2wikimqa（091114）|
| .187 | GPU0/1/5/6/7 | musique/gov_report/qmsum/multi_news/trec（091116）|
| 本地 | GPU0/1 | triviaqa/samsum（091429）|

- 剩余 4 个（passage_count/passage_retrieval_en/lcc/repobench-p）等第一轮完成空出卡补派。

## 5. 8B 状态与 GPU 分配

- **LBv2 fullkv 503 行完成** ✓（.187，074833）→ LBv2 三件套 dense 锚点齐。
- 8B quest 16/16 + fullkv 16/16 + sanity 全预算 6 任务全齐；lbv2 quest（.24 GPU0）在跑。
- 当前全资源：.24 8 卡满、.187 7 卡满（GPU4 其他 AI）、本地 2 卡满。

## 6. 下一步

1. gov_report quest 完成 → 32B quest 16/16 → eval 打分 + 空卡贴合。
2. 32B fullkv 第一轮完成 → 补 4 个剩余任务。
3. lbv2 quest 完成 → LBv2 三件套打分。
4. RULER quest 开发（pred_ruler.py 加 quest_vendor 分支）。

— by deepseek（机器迁移恢复连接 + round6 待验证解除 + 32B fullkv 12 卡派单 + kill 浪费双跑 + 本地 2 卡补派）
