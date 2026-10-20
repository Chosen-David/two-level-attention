# staging97 独立复核

结论：可以按“E155d 同机分析的 staging 运行隔离与证据保留”这一范围发布。没有发现此补丁制造评分或置信区间收益，也不能据此宣称真实模型 T2 达标、性能提升、全局并发发布顺序或持久存储保证。

## 范围与实际差异

基线 `143ff2cf39fe5012d929afe81d71229f054c3f4d`；复核目录 `/workspace/scratch/5d3e4236a5b4/twolevel-work97`。实际查看 `git diff BASE -- exp/trace/analyze_e155d_grid.py tests/test_e155d_snapshot.py`、两文件及 publication 测试相关实现。生产 diff 仅增加 tempfile、空 complete 直接返回、以 `makedirs(exist_ok=True)` 与 `mkdtemp(prefix='run-', dir=STAGING)` 替代删除公共 staging，并把 cell_root 放入独占 run 目录。新增五个回归测试。评分公式、入选条件、AVG5、bootstrap、判决发布逻辑均未改。

未读取 agent_doc/guide；未使用网络、凭据、GPU 或未知作业，未提交或推送。除本报告外无主动仓库文件编辑。测试及额外实验使用临时目录、合成 JSONL 与评分 stub，且设置 `PYTHONDONTWRITEBYTECODE=1`；未调用模型评分子进程。

## 独立执行结果

| 检查 | 实际结果 |
| --- | --- |
| `python -m unittest discover -s tests -p 'test_e155d_*.py'` | 72 tests，全部通过 |
| `python -O -m unittest discover -s tests -p 'test_e155d_*.py'` | 72 tests，全部通过 |
| `git diff --check` | 通过，无输出 |
| `python tests/check_e155d_staging.py` 与 `python -O tests/check_e155d_staging.py` | 独立读查并复跑；各 24 组 direct-parent main 判决差分全部相等 |
| 基线内存加载，单独运行五个新增 SnapshotTests 方法 | 5 tests：1 failure、4 errors，符合故障预期 |
| 真实两个 OS 进程，A 暂停于第一格评分后，B 完整评分退出，再恢复 A | 两个进程均 exit 0；独占不同 run 目录；两份结果均可继续 CI |
| 两进程退出后删除所有 source JSONL 和 manifest，再验证 staged 文件并 CI | 20 个 staged prediction 与 20 个 staged manifest 均通过逐文件 hash 核验；两份 CI 均成功 |
| 空输入，STAGING 原本不存在 | 返回 `{}`，不创建目录 |
| 空输入，已有两进程快照 | 全目录相对路径清单保持一致；已有快照仍可读取 |
| 新运行抛出评分启动异常 / 产生损坏 result.json | 异常分别传播为 RuntimeError / JSONDecodeError；旧 10 个 prediction 与 2 个 result.json 字节完全保留 |

基线复现方法：`git show BASE:exp/trace/analyze_e155d_grid.py` 读入内存，以同一脚本路径设置 `__file__` 后编译执行成独立模块，将 fixture 的 grid 引用换为该模块；只构造五个指定方法的 unittest suite，未把导入模块再次交给自动 discovery。重复评分测试因两次 files 完全相同而 FAIL；其余分别出现他人 sentinel 消失、交错评分 FileExistsError、旧 result.json 消失、空输入删除旧快照导致 FileNotFoundError。基线 traceback 的源码行显示可能对应当前磁盘内容，判断依据为内存加载的基线代码、异常类型和路径，不依赖该显示行。

根侧随后新增的 `tests/check_e155d_staging.py` 已独立读查并普通/-O 各执行一次。24 组不同 seed 的合成输入，每任务 2–6 条且乱序，以真实 main 从 collect、stub score、CI 至发布完整运行。差分只剔除 meta.generated 与 files 的运行目录前缀，保留 tag/pred/filename，并在归一化前实际复核发布快照 hash；其余完整 JSON 相等。两次均输出输入摘要 `3aaf366af9095b401cb7517f1767b986d99093e03091adf1b3be7d9f3125f8f9`。这增加了评分/排序/故事等原有行为不变的证据，仍是合成 stub 差分，不是模型实测。

## 两进程验证细节

独立通过 importlib 加载 SnapshotTests，只手动 setUp 一个 fixture，未调用 unittest.main 或重复 unittest suite。fixture 重新构建两个合法格 `pred_nb8nt320nl8192`、`pred_nb4nt128nl1024`，同一 local host，每格五任务、每任务两条有显式 `_id` 的合成记录；manifest 内容由格参数生成，以 canonical JSON SHA-256 前十位绑定文件名。MIN_ROWS 仅在 fixture 中设为 2；这不是生产完整数据量。

用 POSIX `multiprocessing.get_context('fork')` 启动两个独立 worker（本次 PID 6 与 7）。A 的 after_eval 在第一格写出 result.json 后发 ready 事件，并等待 B 完成事件；B 收到 ready 才调用 stage_and_score，评分全部格后通知 A；A 才继续其余格。父进程通过队列取回两份结果，检查两个不同 PID 与 exitcode 0。评分 subprocess.run 在各自进程里 patch 为 fixture.eval_stub；没有运行真实 eval/model，进程之间共享真实临时 staging 文件系统而不共享 Python 内存写入。

父进程先独立计算 source prediction 与原始 manifest SHA-256，确认实验期间源文件未变；两个进程均重新 collect，source_host 必须为 local。进程结束后删去源 prediction 与 manifest，再核验两份结果的 staged prediction hash 等于 admission hash、staged manifest hash 等于原始 manifest hash，同时重新 cell_treatment_hashes 与返回 treatment_sha256 一致。共核验 2 runs × 2 cells × 5 tasks = 20 份 prediction 和 20 份 manifest。两次输出文件所属 run 根路径不同。

两进程评分均为合成 AVG5 `[75, 50]`；进程退出及源文件删除后，父进程从各自 staged files 重新 per_example_scores，再以 B=40 计算 paired_bootstrap，均为 point_diff=25、ci95=[25,25]、每任务 aligned_n=2。这些数字只证明字节绑定和进程隔离后逻辑一致，不能当真实 T2 或统计性能证据。

## 语义判断与保留限制

- 重复、交错和后续失败运行均不再删除公共 STAGING。run 路径由系统原子创建，且返回的 files 始终指向该次私有目录；单次正常执行结束、CI 完成乃至评分进程退出都没有自动清理。旧预测及结果因此不会被本分析器的下一次调用覆盖，适合后续复核。
- 原有 source admission hash、评分前后 prediction hash、treatment manifest 验证、CI 重读 prediction hash 均保留；改变 staging 位置不会改变排序、score units 或 CI 算法。已有专项覆盖源文件替换/删除、staging 变更、manifest 身份、配对约束、发布异常等。没有新增真实数据或测量，不能推导任何模型效果或运行速度收益。
- 这是同机、同一可访问文件系统、通常同 UID 的隔离；mkdtemp 目录默认私有权限。判决 files 不是跨机器可移植文件，也非只读或防篡改存储；外部清理、人工删除、磁盘丢失、权限变化仍会使旧引用失效。CI 对 prediction hash 的复核不等于对所有 sidecar 的永久持续完整性保证。
- 正常/异常 run 目录都累计，含启动异常、损坏 result.json 与评价失败。补丁有意不做 GC；需后续独立设计保留策略，清理时必须知道哪些判决仍引用这些目录。磁盘容量耗尽也不由本修复解决。
- OUT 仍是一个最终判决文件，既有原子发布保障完整 JSON，但无按数据时间的单调新鲜性或多写者排序。两个 main 同时运行仍可能后完成者覆盖先完成者，即使后完成者输入更旧；失败前留存的旧判决也不能冒充本次成功产物。应结合运行成功状态和判决 meta.generated/collect_root、输入 hash 判断。空 complete 的 stage 无副作用，但 main 仍可发布空 ranking，这属于原有行为。
- 此次独立额外双进程实验只覆盖 stage_and_score 交错与进程退出后的读取；不声称覆盖任意恶意目录操控、远端同步、不同 UID、真实完整模型评估或所有 OS/文件系统。现有子进程异常和 malformed result 的传播行为保持不变，不扩大修复。

综上，实际差异与目标一致，修前故障独立可复现，修后专项与额外双进程证据通过。可按本次唯一 staging 修复范围发布，报告中的生命周期、新鲜性及非真实 T2 限制应一并保留。
