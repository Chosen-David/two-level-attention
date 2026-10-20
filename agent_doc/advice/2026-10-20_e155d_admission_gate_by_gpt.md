# TL-E155D-ADMISSION-20261020：固定配额网格汇总准入修复

日期：2026-10-20。基线main `1cc1e8ceedabd798c196b5dc48c6dd74805c35f4`。仅修复独立TwoLevel仓库，未修改其他仓库/人类指南/根TASK。

## 实际问题与根因

`exp/trace/analyze_e155d_grid.py`：

1. `task_files`（修前73–87行）检测同任务多JSONL后仍留下按文件名排序的最后一个；`collect_cells`只写excluded记录，却照常把该主机作为完整副本。
2. `collect_cells`（修前124–138行）发现完整副本SHA不同时仍返回complete，用首个主机打分；与模块明确写出的“逐文件SHA全等才放行”冲突。

真实消费链为收集快照 → task_files/collect_cells → stage_and_score → AVG5 ranking → top-N per_example_scores/paired_bootstrap。因此仅写warning不能保护固定配额nb×nt×nl选优。最新main的5项最小测试3失败，覆盖歧义单机、歧义优先机+有效后备机、冲突完整副本；两个一致/partial控制通过。

## 修复及取舍

- 从文件映射移除歧义任务，不挑文件；该主机变为不完整。保留excluded的歧义说明和缺任务诊断。其他合法完整主机可继续作为来源。
- 多个有效完整副本发生冲突，整个格不进入complete；incomplete和incomplete_before_closure保留reason=sha_conflict、基准主机、冲突主机、差异任务的完整SHA256。
- 不作多数票，不跨主机拼partial，不放松200/500行门；相同完整副本保留原host优先序与dup_hosts。
- CLI及JSON notes明确不完整列表也包含冲突，不能把冲突解释成“等待跑完会自动消失”。

当前技能沿用 research-implement-optimize 的失败复现→有界修复→回归→独立复核流程；采用标准库集合移除和既有SHA比较，无新依赖。与重写快照系统相比，这个补丁直接恢复已登记准入契约；完整快照原子性、sidecar全身份对比不在本次验证范围。扫描/哈希复杂度不变，不声称更快。

## 实际验证

证据位于 [e155d-admission86](../results/e155d-admission86/plan.md)：before.log保存5项3失败；after.log和after-O.log均13项通过。覆盖空集、499行RepoBench、三份相同任务文件仍歧义、单任务冲突完整hash、2相同+1冲突、partial不合并、完整后备来源、legacy轴排除/CTRL保留、main不把冲突传到评分和CI。主入口测试替身只替换外部打分器，验证派发过滤，不代表真实LongBench评分通过。

相邻host回归：固定配额17项（explicit.log），E152结果/计时16项（adjacent.log），vendor冻结身份7项（vendor.log）通过。[独立验证](../results/e155d-admission86/review.md)结论usable-with-scope：87个合法格与基线全部收集返回值一致，2相同+1冲突全部6种主机顺序拒绝；两个合法格实际进入main的CI控制流、冲突格不进入。评分与CI数值使用替身，不能替代真实评测。

环境：Linux x86_64，Python3.12.14，标准库临时文件；无Torch、nvidia-smi，未使用GPU/驱动/编译器/kernel配置。合成输入由测试生成，fixture文件SHA256 `0b23d028dcf6720318594cdbc83c90b3113ccec86fba123cafa0ae0c60be60b1`；候选分析器SHA256 `7f55f7305f17088781fccd9d3530e917474620e6370937a838d24a0667f8e541`。

## 历史影响与未完成项

本次没有拿到远端完整快照，不声称已证实某格或历史冠军污染。没有更改原始预测、旧verdict或论文数字；没有启动/停止任何远程作业。生产推理及固定配额公式未改，无理由仅因汇总bug重跑全部GPU预测。

T2所有旧排名在使用前应由执行者用修后收集器复核：发生歧义/冲突的格先核对原文件及provenance；不能仅删除不利副本以通过门。解决后重新评分、排名与CI，才消费为T3–T6论文/全量测试依据。若需重新推理，必须先确认损坏来源和准确范围；不能以本host测试代替真实数据验收。

GPU未完成项保持：near-width候选待真实选集验证，T7同机独占稳态对照待可用资源；外层chunk2048、算法选择粒度、预算、子空间、D′ gate、sink/SWA与kernel均未改。本轮仅恢复结果准入正确性，不报告质量/速度/显存提升。
