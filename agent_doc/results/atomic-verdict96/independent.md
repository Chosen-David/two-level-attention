# TL-E155D-ATOMIC-VERDICT-20261021 独立复核

## 修前复核

- 基线 HEAD：`77f3c0d16d6d35da32da69b9241723e1e0f25c5c`。
- 已独立阅读计划、实际 `analyze_e155d_grid.py` 的 `main()` 发布路径、7 项发布测试、`before-v2.log` 与 `before-probe.json`。
- 独立运行 `python -m unittest discover -s tests -p test_e155d_publication.py -v`：7 项中 6 失败，正常控制通过。异常序列化和子进程 `os._exit(23)` 均将原输出截断至 `b'{"ranking":'`；首发异常会留半成品；原实现未调用 fsync/replace，且会跟随目标符号链接。与修前原始记录一致。
- 计划的同目录独占临时文件、完整序列化及 flush/fsync 后 replace，足以解决单次发布时公开 JSON 被半写覆盖的问题。mode 保留及拒绝目标符号链接属于明确发布约束。

## 必要边界

- 不承诺断电持久化：文件 fsync 不等于父目录 fsync。
- 硬退出可能留下本次唯一临时文件；可消费的公开路径应仍为完整旧文件或完整新文件。
- 不解决共享 staging 并发、目标路径检查与 replace 间的并发 ABA，亦不让旧判决自动代表当前数据。
- CPU 故障注入不等于真实 T2 数据重评或 GPU/内核验证；未读取 collect 脚本、guide、凭据，未使用网络、GPU或其他代理。

## 修后复核

结论：当前候选的公开 JSON 原子发布修复可验收；未发现阻塞问题。复核对象为当前工作树，生产源 SHA256：`0365f22e42ac257814a43af73df654d4cc6c5add1596b3d9c46489fdd43c388d`。

- 源码独审：只有 stat/uuid 导入、`_publish_verdict` 和 main 最后发布调用变化。独立 AST 比较在逆转这三处后与 `77f3c0d` 完全相同。helper 同目录 exclusive `x`，成功获取后才标 owned；fchmod、dump、flush、fsync、关闭均在 replace 前。finally 仅处理自有临时路径；碰撞不会误删已有文件。
- 最终发布测试 10/10 通过，`python -O` 亦 10/10：序列化中断、首发中断、fsync/replace/fchmod 故障、进程硬退出、正常控制、符号链接、临时名碰撞，以及新增读者可见性。独审新增测试：fork 子进程将临时文件写至中途、flush 后用 Event 暂停；主进程仍读到完整旧公开 JSON，释放子进程后读到完整新 JSON。它验证单个发布者和读者，不是多发布者/staging 并发测试。
- 内存加载 `git show 77f3c0d:exp/trace/analyze_e155d_grid.py` 后执行同一最终 10 项测试，独立重现 9 失败、正常控制通过、0 error；未覆写工作源码。最终测试文件 SHA256：`7bf2126ef52314a8d1fb137ab180aac04a98f85c3fed5c42e08a53d6f55dc55c`。根代理修前最终原件 `before-final10.log` SHA256：`6c0356d84c40d01e66d4a4f156d09a8c99d5f72553baa3f35be56dfe4c7532a6`，已独立阅读并与重跑结果核对。
- 全部 E155d 最终重跑：`python -m unittest discover -s tests -p 'test_e155d*.py'`，67/67 通过；全组 -O 由根代理执行，独审者单独执行最终发布 10 项 -O。
- 相邻 CPU：`test_explicit*.py` 17/17、`test_vendor_frozen_manifest.py` 7/7 通过。额外尝试 `test_e151c*.py`，4 个模块均因缺失 torch 导入失败，未进入测试；不记为通过，不安装依赖或据此外推内核行为。
- 原有检查：`check_e155d_geometry.py` 验证 87 个真实 producer 几何 manifest，51 个错标拒绝，输入 SHA256 `70b6b0f0f99cf7f3e213cd98169540a648ca7e5b28db6e3a6eae8f9575dbd241`；`check_e155d_snapshot.py` 的 24 个完整入口差分通过，输入 SHA256 `3aaf366af9095b401cb7517f1767b986d99093e03091adf1b3be7d9f3125f8f9`；`check_e155d_score_units.py` 的 24 个历史差分及真实聚合 scorer 分制通过。
- 追加直接前版差分：当前真实 main 与 `77f3c0d` 在同样 24 个合成评分输入上执行，仅移除 `meta.generated` 后整个 JSON 相等，未剔除路径、预测/治疗 SHA 或任何指标字段；输入 SHA 与上项 24 入口相同。
- 追加发布 probe：保留旧打开文件描述符读到完整旧字节、成功后公开文件为新 inode；原有 0640 保留，首发在 umask 0027 下为 0640；Unicode UTF-8/indent=2 序列化字节与原格式一致。`git diff --check` 通过。

以上主流程测试覆盖真实收集/快照/几何判定与 main，评分处使用固定合成值；发布故障测试替换收集为空 fixture、继续走真实 main。它们支持发布协议及合法输入行为不变，不构成真实模型评估或科学结论。独立执行的详细原始输出在本次工具记录；修前原件继续保留于 `.agent-runs/atomic-verdict96/before-v2.log`、`before-v3.log`、`before-final10.log` 和 `before-probe.json`。未检查远端推送或发布后主分支；由根代理执行并读回。
