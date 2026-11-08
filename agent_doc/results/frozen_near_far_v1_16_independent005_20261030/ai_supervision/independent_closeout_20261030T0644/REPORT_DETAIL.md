# 独立verify_experiment_result收尾记录

V1完整科学验收pending。本轮没有发布部分均值、CI或3pp结论，没有运行部分质量汇总；最终实时库存见LIVE_COVERAGE.json。3750/11250/128done未齐前不写V1 FINAL_REPORT_ZH.md。

## 身份与权限

独立Codex exec会话01a18b8c-fcc0-7603-9166-bd2360bc7539，实际启动头SESSION_HEADER.txt，IDENTITY.json保留环境身份。用户明确指定独立验收者；collaboration不可用为用户交接信息，本轮未调用或创建其他agent。PROJECT_ROOT=/home/wangyuanshuo02/two-level-attention，WORKFLOW_ROOT=/home/wangyuanshuo02/agent-research-workflows。exec login=false、CUDA_VISIBLE_DEVICES空、Python -B；未派GPU、kill、创建监督器，未写guide/任务/生产源码/原始数据/NEXT_STEPS/COMPLETE。

## 实际完成

已读gpu-experiment-dispatch、项目文档/复用/验收契约、执行规范及作战手册相关节、项目GUIDE/任务、现有0615与0313工具/证据、0355 Qasper成套入口、0546 MK2历史验收、0415封存入口及最新YARN_SCOPE_NOTICE。检索限用户指定run，不假称全项目搜索。adopt现有独验工具；adapt prepare_review.py增加全量前置门禁并固定生产导出（diff已保存）；defer V1全量科学结论至完整验收。

V1 recheck/cache/closeout/extra_checks/reference_dependency及六类artifact封存工具成套准备，PREPARED_TOOL_BINDINGS.json记录字节；FULL_V1_EXECUTION.md记录精确步骤。没有重新造通用验收工具。主控负责原生产validate/analyze/export，本轮未代写或启动生产分析。

专项目录独立运行CPU_TESTS 8项通过，CPU_SOURCE 53项通过（真实冻结源码四态缓存/环绕/短输入/profile参考/tiny Qwen与TLI参考/stop/request guard）。SOURCE_REVIEW.md记录本轮真实代码审读。仅工具及CPU参考验证，不是V1全3750原始输入/评分/缓存或B5000/AVG16已验。V1完整合同六项均inconclusive，无N/A绕过。

Qasper优先等待200/600并单独使用B10000；实际最新结果见专项independent_closeout_20261030T0644/REPORT_DETAIL.md。不得将Qasper独验替代V1。旧MK2 original40960/factor2仅历史范围，旧报告SHA与0546封存相符后备份，当前专项报告补上范围更正；不能冒充original32768新主表，本轮未重新生成MK2。

## 限制与故障记录

无独立GPU重生成，无QK不能重算真实mass，启动后磁盘hash不能证明历史内存权重。缓存回执不能证明逐步GPU张量。Full是TLI gates-off而非dense FullKV。无性能、无损或等价结论。

一次inventory已完成库存但写专项PROGRESS时错误使用V1共享盘兄弟路径，退出1；已改为显式PROJECT_ROOT并成功重跑。一次functions JavaScript括号语法错误在执行前拒绝。几次只读查找猜测文件名不存在，未据此推断科学失败。早期PROGRESS手写06:48时间标注已更正，真实时间以command.json/工具回执为准。

最终EVIDENCE_SHA256.json仅绑定本目录稳定证据（包含本报告），排除活session.log、CLI_FINAL.md和清单自身，写出后逐文件重新读取校验。独立身份来自用户授权及实际会话，不以自写JSON自证平台可信provider已安装。

最终收尾观察：2026-10-30T07:06:50.721686+00:00，Qasper 198/200，缺失 [{'row': 41, 'rank': 127}, {'row': 134, 'rank': 127}]；V1最近库存 3650/3750、10957臂，队列{'done': 119, 'running': 8, 'pending': 1}（库存时间2026-10-30T07:06:17.629828+00:00）。完整科学验收仍pending。

本轮末尾另核两处实际V1源码各81文件与08ce44d git对象一致，含4个CPU测试源及修订报告共167外部绑定重读一致，errors=[]；见专项SOURCE_AND_REPORT_BINDINGS.json。此范围没有原始生成全量验收或模型权重重hash，不冒充六类结果artifact闭环。
