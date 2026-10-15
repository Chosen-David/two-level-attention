# O4 部分末页修复交接

唯一任务：[TL-E140-O4-PARTIAL-PAGE-20261015](../task/task_details/TL-E140-O4-PARTIAL-PAGE-20261015.md)。

代码已修复有效末页统计/候选裁界，配额不跨区补齐。CPU测试与独审通过不代表新实验。
旧 O4 v2 raw 和面板(d)必须继续标历史诊断，不可采信；O1不受本缺陷影响。
无重采、无GPU。将来仅在用户另行授权且身份齐备后执行v3采集；默认dry-run、fresh输出防覆写。
不要把现有O3相对reference-mid-TopK mass比例写成full-attention mass。
提交仅原dev，用户审核后合main。
