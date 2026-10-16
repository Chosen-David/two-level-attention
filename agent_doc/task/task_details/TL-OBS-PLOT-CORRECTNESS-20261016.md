# TL-OBS-PLOT-CORRECTNESS-20261016

## Plan
用户授权：截至00:55UTC完成有界绘图正确性修复，否则checkpoint。基线main47e4ab2，治理远端06ed701。只修改绘图/标签/出处及测试，保留raw与Git可恢复历史资产；不改manuscript/public-paper、GUIDE、SGLang、Claude，不GPU/重采/扩大矩阵/动态算法。

采用可行整箱贪心质量排序诊断，遇不合预算的箱跳过，报告实际token数；不冒充token TopK或上界。绝对token边界由q-d变换并验证region_bounds，拒绝跨区箱，零分母未定义；配对有效分母明确。E6面板阻断gate安全性推论，不复用skip阈值。

验收计划与DAG见`agent_doc/results/20261016-observation-plot-correctness/validation_plan.json`；独审者/root/independent_review。历史结果只复用身份及限制，新增合成CPU边界检查不属于科学数据验收。无实际supervisor后端，不声称已启动监控；由当前执行器检查截止时间。

## Progress
- Fresh fetch并fast-forward main47e4ab2；原dev起点fd562101；无并发本地修改。只读既有O1数据/guide observation及相关advice；未读取人类GUIDE。
- 已实现真实绝对端点与区域质量核验，bin边界不支持拆分时拒绝；无箱内插值。整箱greedy遇不fit跳过，actual tokens与finite paired N入JSON/TSV。
- 现存数据一致性：128条记录完整覆盖且region质量和一致；当前diagnostic far mean=0.6759161457、near mean=0.9314942679（各n128），near>far=100/128。这些是改正定义后的整箱诊断值，不是tokenTopK/e2e或gate结论。
- 初次图件检查发现colorbar覆盖(a)说明，已移动并重画；(b)统计框移出点区。PNG/SVG/PDF均带diagnostic警示；E6 panel blocked，O3比例标签不变。
- 证据runner首次regression命名与已有日志冲突，拒绝覆写；保留failure记录。修为phase前缀并在执行前检查存在性。最终每个subprocess.returncode单独保存；不能用最后tail状态替代。
- 最终独审通过：统计7项/资产7项普通与-O均exit0；受影响collector15项普通与-O均exit0。独审确认JSON/TSV128行一致、预算<=2048、128有限配对，当前资产/代码/raw/旧Git快照哈希对应；仅诊断算术与呈现可用，不科学采信。
- 发布前fresh main前进至9e03100；同图2f1ba4d新增quota曲线继续使用已确认错误的区域/预算oracle逻辑，已可恢复归档于manifest。保留已审诊断图，合入E152文件且与main字节一致，未执行/改动E152。merge9b582aa后资产normal/-O各7项exit0，整合独审无阻塞。`exp/paper/observation_bins.py`已显式纳入Git。
- 初次及修正版render日志、真实进程exit codes、独审与整合回执均在本任务results目录。SVG的Matplotlib多行path有尾随空白，非语义警告；其余文件diff-check通过。不因格式问题重做已验收结果。
- 无GPU、新采集、权重下载、main推送、稿件目录或人类GUIDE修改。未发现托管CI配置；本轮不声明CI通过。下一owner为用户审核dev；科学采信/当前gate安全及文档独立性仍未验收，不自动新增实验。
