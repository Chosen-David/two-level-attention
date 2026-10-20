# E155d admission86 独立复核

结论：**usable-with-scope**。在本次明确的 admission 契约与合成 CPU 文件夹测试范围内，未发现阻塞问题。歧义任务不再按文件名取赢家；有效完整主机副本存在任意预测文件 SHA 冲突时，不再按主机优先级或多数票放行。此结论不是远端实验、真实评测或性能验收。

## 对象与可追溯性

- 复核工作树：`twolevel-audit86`；基线：`1cc1e8ceedabd798c196b5dc48c6dd74805c35f4`。
- 已读 `plan.md`、完整 analyzer、新增测试和 analyzer 对基线的 diff；查看既有 `before.log`，并独立重现其失败数。不读取含凭据的收集 shell，不访问远端资源。
- 当前 analyzer SHA256：`7f55f7305f17088781fccd9d3530e917474620e6370937a838d24a0667f8e541`。
- 当前测试 SHA256：`0b23d028dcf6720318594cdbc83c90b3113ccec86fba123cafa0ae0c60be60b1`。
- `git show` 获取的基线 analyzer 文本 SHA256：`621c6610b7a4f28dfff700af19c0657b2236ce369aeac75b35db42dc0168915a`。
- 本复核只新增本文件；未改源码、测试、TASK、guide，未提交或推送。复核期间共享工作树另出现 TASK 和 task_details 变动，它们不是本复核所写，也不包含在代码结论中。

## 六项检查及实测结果

| 检查 | 独立执行的证据 | 结论与范围 |
| --- | --- | --- |
| 1. 失败重现与新测试 | 将当前测试模块的 `grid` 替换为内存加载的基线 analyzer，仅运行最初五项：5 项、3 failures、0 errors；失败为歧义入围、冲突入围、歧义主机阻挡有效 fallback。当前普通及 `python3 -O` 各 13/13 通过。 | 原因与修复方向一致；不依赖可被 `-O` 删除的断言。 |
| 2. 正常网格对基线等价 | 独立按轴及几何约束枚举 86 个合法 geo 格，加 1 个 control；每格在两主机各写一完整同字节副本。新旧 `collect_cells()` 全部四个返回值逐项完全相等，87 格全部入围。 | 保留合法同 SHA 副本的源主机选择、去重及返回数据；未调用真实评分。 |
| 3. 歧义副本及有效 fallback | 单主机同任务两个同字节文件仍判歧义；`task_files()` 不含该任务；另一完整主机可以入围，且 excluded 中仍保留歧义主机/目录/任务诊断。对比基线源主机 a，新实现选择 b。新增测试另覆盖三个同字节任务文件。 | 文件内容相同不消除“每任务唯一文件”约束；歧义主机不会否决其他有效完整主机。 |
| 4. 冲突及诊断 | 两个相同完整副本加第三个不同副本，枚举三主机的全部 6 种优先顺序：基线均放行、新实现均拒绝。独立逐项重算诊断中的主机、任务与两个完整 SHA。新增测试另覆盖仅一个任务冲突。 | 无多数票例外；reason/source_host/conflicting host/full hash 足以定位本次冲突。`per_host_missing` 为空列表并不代表放行，须连同 `reason=sha_conflict` 解释。 |
| 5. 其他拒绝路径及主消费者 | 互补 partial 两主机和轴外目录的新旧结果完全相等。独立调用真实 `main()` 控制流，构造 1 个冲突格及 2 个有效格；替代评分器只收到两有效格，逐例分数替身仅收到两有效格，CI 替身实际调用 1 次；落盘 verdict 排名无冲突格且拒绝诊断存在。 | 比仅有一有效格的新增 main 测试进一步覆盖了实际进入 CI 分支时的过滤；评分与 bootstrap 数值本身未验。 |
| 6. 改动边界与成本 | 对基线 diff 仅涉及歧义任务移除、冲突拒绝/诊断和 CLI/verdict 文案；`git diff --check` 成功。未改变 scorer、排序公式、bootstrap、kernel 或预算。 | 仍是目录/行计数扫描与完整副本字节哈希；没有性能测量，也没有速度、真实模型质量或远端受污染格数量结论。 |

独立正常与优化模式命令：

```text
python3 -m unittest discover -s tests -p 'test_e155d_grid_collection.py' -v
python3 -O -m unittest discover -s tests -p 'test_e155d_grid_collection.py' -v
```

补充动态探针使用临时目录和内存加载的基线/当前模块，无源码替换；测试通过后临时合成数据删除。其 stdout 摘要：

```text
Independent baseline reproduction: tests=5 failures=3 errors=0
CHECK1 valid full-grid baseline equivalence: PASS 87 cells (including ctrl), two replicas each
CHECK2 ambiguous equal-byte files + valid fallback: PASS; baseline selects a, current b; ambiguity diagnostic retained
CHECK3 majority conflict: PASS all six host permutations rejected; baseline admits all six; exact hashes verified
CHECK4 partial-host no merge + invalid axis: PASS, identical baseline/current
CHECK5 main consumer: PASS; scorer only two admitted cells; actual CI control flow invoked once; conflict diagnostic persisted
```

## 限制与非阻塞事项

- 本契约中的“有效完整”在收集层指合法轴、单主机五任务各唯一文件、达到行数门槛；收集层没有验证每行 JSON/ID/内容语义。真实 eval 的 strict/runcap 消费链未在本环境执行。
- SHA 比较对象是五个预测 JSONL，不是 sidecar/runcap/manifest 的全 provenance 一致性。这是现有实现边界，不把此次 admission 修复解释为完整 provenance 证明。
- 文件计数、哈希和后续复制不是原子快照；并发写入/重收可能造成读取时序问题，本次没有模拟或解决。应以稳定采集快照使用本结论。
- 部分副本不参与完整副本 SHA 冲突判定；无歧义的有效完整副本可覆盖 partial 或歧义副本。这与此次固定契约一致，不能延伸为对所有残留文件内容一致性的担保。
- 只有 `complete` 进入正常主入口的评分、排序与 CI；独立探针证明该传递关系，不证明第三方自行绕过收集门调用 `stage_and_score()` 时仍有防护。
- 无 Torch/GPU、无真实模型推理或远端快照验证，未改变原始实验数据/既有分数。后续真实评测需单独验收。
