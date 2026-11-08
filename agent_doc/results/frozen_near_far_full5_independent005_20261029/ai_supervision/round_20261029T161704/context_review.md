# 第五轮依据与动作边界
PROJECT_ROOT=/home/wangyuanshuo02/two-level-attention；工作树=/home/wangyuanshuo02/two-level-attention-near-far-skip。读取用户AGENTS、GUIDE、当前任务详情/索引、README/dispatch_plan、上一轮报告/交接以及gpu-experiment-dispatch技能和项目文档/结果复用/独立验证规范。两个项目根无磁盘AGENTS.md。沿用既定冻结源及300秒派单；不修改指南、阈值、算子或运行源，不重启E184、不用本机GPU1，不添加调度器。
复用：旧screen不并入全量；旧CPU边界及源码路径检查仅限其绑定范围，第四轮1241评分/1227输入子集不替代本轮完整1300集合。未规划新增实验，不以外部知识替代实测。advice目录最新仍是10月29日17:48 RULER条目，未见上轮后新增；无关扩展defer，需用户另行授权。主项目并发修改（包括guide与其它代码）全部保留，不纳入本任务修改。
独立collaboration后端报thread-store缺rollout，失败原文保存；转两个实际codex exec --ephemeral会话，行为只读CPU，外层保存日志与报告。不声称OS只读沙箱。现有verification_plan为简版，保留历史合同限制不倒填。
