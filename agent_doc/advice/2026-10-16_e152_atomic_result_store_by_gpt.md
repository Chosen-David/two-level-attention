# TL-E152-STORE-20261016：结果追加串行化与原子发布

## 触发、根因及修复

基线为重新读取的 main `be2817946e678ef1464eab1d03d1142218e82602`。
CLI fetch 返回 CONNECT 403 后，通过已连接 GitHub 读取新提交的三个变更文件，
逐个校验 blob SHA，重建 tree 和 commit 并匹配远端 SHA，再 fast-forward、
重放本次补丁。保留该提交的 StaticCache 适配和原始实验判决，不将其 GPU
收益视为本轮实测；未修改 human-only guide。

旧 `exp/bench/e152_e2e_latency.py:284–293` 在无锁状态读取整个数组、追加、
用 `w` 截断原文件。两个进程交错时最后写者会覆盖另一条记录；序列化异常
或进程退出可留下不完整 JSON。编排器交错启动进程并不能保证写入互斥。

修复为同文件 `173–213` 的 `_append_record`，main 在 `328` 行调用：

- 对 realpath 对应的持久 `.lock` sidecar 持有排他 flock，覆盖整个读取、
  追加、发布过程。锁文件不删除，避免等待者与新写者锁住不同 inode。
- 在同目录临时文件中完成 JSON 序列化、flush/fsync，再 `os.replace`。
  普通发布前异常保留原文件并清理临时文件；损坏或非数组输入拒绝覆盖。
- 保留已有文件权限位，新文件使用 mkstemp 的 0600。读者打开目标路径时
  看到完整旧版或新版；已打开的旧文件描述符仍可继续读取旧版。

只加锁仍会在中断时截断原文件；只原子替换仍会丢并发更新，因此两者同时
采用。不引入 SQLite、后台服务或第三方依赖。仍重写整个数组，时间与临时
空间均随历史 JSON 大小增长；没有声称解决无限历史扩展或获得 I/O 提速。
写入发生在请求计时之后，未修改模型调用、StaticCache、schema、预算、
chunk2048、selector、kernel、同步策略或请求时间戳定义。

## 实际验证

版本/hash、环境、命令及原始短日志见
[validation.json](../results/e152-store-20261016/validation.json)。
固定历史输入字节为 `[{"schema":"e152-e2e-latency-v1","runs":[]}]` 加换行，
SHA256 `5425eeb7a85365318f354bf6a52c0226137f80538c5853e84cad5ad1160d930b`。
测试 AST 执行实际源码的函数；基线执行实际 main 写入尾段，排除模型加载。
真实 fork 子进程与事件协调交错，不使用虚构 GPU 或模拟计时结论。

```text
最新基线：Ran 8 tests; FAILED (failures=5, errors=1)
并发记录：[历史,A,B] 预期，旧版实际 [历史,B]
中断写入：子进程 exit=23，旧目标变为 [{"incomplete":
修后普通：Ran 16 tests; OK
修后 -O：Ran 16 tests; OK
```

8 项新增覆盖：串行历史/Unicode、双进程丢更新、进程中断后保留原件与重试、
序列化/replace/fsync 异常、坏 JSON/非数组拒绝、新文件及已有 0640 权限。
其余 8 项为既有 E152 协议、计时、实际 main 写出回归。修前故障数包含新
fsync/replace 防护尚不存在所致的故障注入失败，不代表六个独立生产 bug。
编译与 whitespace 检查通过。普通/`-O` 的 11 项 LongBench 捕获回归本次
均在导入阶段因 `ModuleNotFoundError: torch` 未运行；保留原始失败，未伪造
pass、未用 stub 补测。此前旧基线环境的通过记录不作为本次重放后的验收。

环境为本次 JSON 中 Python/Linux/x86_64；新增测试仅标准库，无 Torch、
Transformers、GPU、CUDA/Triton 编译器或驱动参与。没有新增质量、速度、
显存、kernel 数、host 同步或端到端实模收益结论。

## 适用边界与历史影响

仅覆盖同一解析后路径、遵循 sidecar 协议的本地 POSIX 写者。非合作写者、
硬链接别名、锁文件被外部删除、分布式文件系统/NFS 锁语义不在本次验收内。
未同步父目录，不提供断电持久性保证；替换后是否成功未知时重试不具幂等性，
调用者需自行核对记录避免重复。文件替换不承诺保留 inode/owner/ACL/xattr。
被杀进程可能留下未发布的隐藏 `.tmp`，不会自动被当成有效结果。不要在有
写者时清理 `.lock`；离线清理 orphan 前应确认不属于在飞写者。

不改写既有实验数据，不能恢复已丢记录。过去并发写同一路径的 E152 文件
应与任务清单逐臂核对条数和身份，缺失或损坏的运行重采；外部作业状态未知，
不能声称所有历史结果均完整。旧 timing advice 中“要求编排器串行写、未验
并发”的历史限制由本项在上述局部范围内补齐，原文保留。真实 GPU 路由、
低扰动 E2、prefill 性能与质量验收继续开放。
