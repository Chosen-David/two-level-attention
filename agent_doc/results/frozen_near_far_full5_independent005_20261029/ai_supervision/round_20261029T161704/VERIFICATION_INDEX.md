# 全量独立验收证据索引

此索引由监督主控整理，不是主控自生产自验收。两个独立codex exec会话均实际运行、exit 0，真实session ID、启动头与报告/log/prompt哈希见independent_sessions.json。

- [评分/统计/导出独立报告](independent_scores/REPORT.md)：1300配对3900官方评分、bootstrap、最终summary/五TSV/predictions/raw_index及1533资产绑定，usable-with-scope。
- [输入/缓存执行路径独立报告](independent_paths/REPORT.md)：1300输入3900输出、140400回执、421200缓存快照及源码/资产，usable-with-scope。该报告写出时尚未有汇总的pending现由评分会话闭合；不改写原报告。
- [独立会话证据](independent_sessions.json)、[评分结构化证据](score_check_evidence_extracted.json)、[路径结构化证据](path_check_evidence_extracted.json)。

两会话绑定同一最终1300请求摘要fd595e95527362034fe2bf859061d93060c378cf3b179ce38ed21b98de8fa431。1533资产摘要901e031bbf6110667ebcdabacad7c03447e3750cea84910deddb84206b5471f5，主控再次按实际字节核对未变，见binding_closure.json。数值/准确率和记录一致性已在上述范围独立通过；不存在要求中的剩余全量缺口。

未独立GPU重复生成、原始张量/逐步选择轨迹、完整权重历史hash及部分kernel环境仍属限制，不伪称通过。无事前等价界，不称无损；不支持速度/整卡显存收益。现有简版验收合同不倒填schema。质量下降不影响测量的限定可用性。
