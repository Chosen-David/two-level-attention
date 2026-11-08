# LongBench V1 16任务完整确认集

3750配对样本，三臂共11250次生成。multifieldqa_en150条，lcc/repobench-p各500，其余13任务各200。所有本地数据行，不是抽样；不含V2或中文任务。

三臂：TLI gate-off、Far-only0.05、Near/Far独立0.05。Qwen3-8B，prefill末256query平均mass逐层冻结，单请求eager，operator off；cache和scorer不做本轮优化。上下文cap31500，固定各任务生成上限。V1 no-chat5任务，SAMSum EOS+newline及min1停止规则。one-shot full-prompt协议与历史split-prompt结果不能直接混表。

AVG16等权任务均值，单任务和AVG16下降3个百分点内为用户工程容忍度，不宣称无损。保持逐任务/配对差异及门控触发率报告。旧AVG5没有并入新测量，因为main源码和协议身份不同；原结果保留。

三机8卡按dispatch_plan授权范围，每300秒自动空卡接力，128个逻辑分片，失败最多4次尝试。AI监督每1800秒启动实际Codex轮次，负责健康诊断、推进和最终独立验收。共享盘结果在本机本目录实时可见，不依赖助手手工回传。

进度看queue.json和daemon_*.json；AI记录看ai_supervision/status.json及NEXT_STEPS.md。FINISHED.json仅表示生产/机械核验完成；独立验收及最终报告完成才写ai_supervision/COMPLETE.json。创建PAUSE暂停新GPU派单；ai_supervision/STOP停止AI监督。当前准备中，点火结果以deployment_receipt.json及launch_live_check.json为准。
