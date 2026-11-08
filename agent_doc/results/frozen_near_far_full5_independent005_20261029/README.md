# AVG5 全量双 Gate 自动接力实验

范围：Qasper、HotpotQA、Musique、GovReport 各 200 条，RepoBench-p 500 条，共 1300 条配对样本、3900 次生成。三个配置为 TLI gates-off、Far-only 0.05、独立 Near/Far 0.05。prefill 最后 256 query 标定并冻结；decode 使用分区缓存；算子优化关闭；上下文 cap=31500。

三台远端各运行独立常驻调度器和 shell watchdog：pc02 GPU0–3，pc40 GPU0–1，pc31 GPU0–1。每 300 秒检查空闲卡，每小时记录审计；64 个逻辑分片按空卡动态领取，每分片最多尝试 4 次，已保存的请求/配置结果可断点续跑。不会使用本机 GPU1。

结果直接写入三机与本机共用的共享盘。本目录是固定入口，日志、预测和结果实时可见，不需要等待助手在线回传。共享盘与远端主机须保持可用；不会自动抢占失联主机尚未确认退出的任务。

- `dispatch_plan.json`：冻结代码、参数和授权 GPU 范围。
- `queue.json`：分片待跑/运行/完成/失败状态。
- `daemon_pc02.json`、`daemon_pc40.json`、`daemon_pc31.json`：各机调度器最新状态。
- `worker_rank*_attempt*.log`：每次尝试日志。
- `manifest_rank*.json`、`worker_done_rank*.json`：输入身份与完成证据。
- 完成后自动产生 `REPORT.md`、`summary.json`、`validation.json`、`scores.tsv`、`per_request_scores.tsv`、`predictions.jsonl` 与 `raw_index.json`。
- `FINISHED.json` 表示生产和机械核验完成；精度结论仍须独立结果审阅。
- `NEEDS_ATTENTION.json` 表示有分片耗尽重试次数，不能视为全部完成。

创建本目录中的 `PAUSE` 文件可暂停领取新任务，已运行任务继续；删除该文件恢复。调度器不依赖 SSH 会话、Codex 或本机监督进程。主机重启后的操作系统级自启动不在本轮部署范围。

## 30分钟AI监督（用户最新设置）
`ai_supervision/status.json` 保存当前AI轮次及下次时间；每1800秒实际调用 Codex CLI 检查、决策并推进，报告为 ai_supervision/时间戳.md，下一步为 NEXT_STEPS.md。远端自动派单仍每300秒，远端原始审计日志仍每小时；此处新增的AI推进监督每30分钟。不会修改运行中源码或提高门限。独立核验和最终报告全部完成后停止；用 ai_supervision/STOP 可手动停止。本机或Codex不可用时AI监督不能执行，但远端任务仍独立接力。
