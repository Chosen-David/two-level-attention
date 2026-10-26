# TL-SKIP-NOSKIP-FUSION-20261026

用户要求：skip 与 non-skip 都享有 GPU 融合；同时核查直接贴在会话里的 Kimi
驻留意见（不是远端 advice 文件）。实现和 adopt/adapt/reject 逐项记录见
`docs/tli_skip_noskip_fusion.md`。

新增区域打分/GQA/几何融合，skip 没有 far score launch/top-k；固定容量 B=1
可用公共路由，冻结后的 B=1 动态 gate 准入；compact 模式非 skip 层仍可用
full-KV 融合，skip 层用已有 ring 路由。中途 gate 改变需 clear/recapture。
真实两层 Qwen 混合路由测试暴露 B=1 valid_length list/int 不匹配，已修复。

当前 ring 已一次性转换、单 token 原位更新，本轮没有重写它。新增测试验证
两圈 wrap、恒定写入量、旧存储引用释放、逐 prompt gate 漂移，以及独立
storage/allocated/reserved/peak 远端探针。4291 的容量包括页对齐，不能直接
缩成4228；严格截断与历史 boundary bypass 有语义差，质量结果不得继承。

本地仅 CPU/解释器/离线编译。任务保留未完成直至真实 GPU、完整模型和质量
验收；无独立 reviewer 结果。原始候选 spill 证据、最终结果及 hash 绑定存于
`agent_doc/results/tli_skip_noskip_fusion_20261026/`。

整合 main 262517b（保留其论文/数据和 CLI 缺省变更）；恢复构造函数被误删的
增量/聚类状态初始化，修复全量统计融合对已抽取子空间再次用全局维度索引。
已补已知 channel 值回归；旧 AST harness 同步当前图上下文、元数据签名和
四返回值接口。最终本地已下载测试子集 254 passed / 182 skipped / 217 subtests；
合并后解释器子集 99 passed / 38 skipped；未改 pipeline 的离线编译 284 项。
