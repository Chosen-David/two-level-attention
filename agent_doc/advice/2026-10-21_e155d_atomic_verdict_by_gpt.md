# TL-E155D-ATOMIC-VERDICT-20261021：T2判决原子发布

基线`77f3c0d16d6d35da32da69b9241723e1e0f25c5c`。
旧`exp/trace/analyze_e155d_grid.py:439–441`先用`open(OUT,'w')`截断旧判决，
随后才序列化。实际main故障注入和子进程硬退出均留下`{"ranking":`，
旧有效JSON丢失；写到一半时，并发读者也会读到半成品。

修复在当前`:379–402,467`：同目录唯一临时文件独占创建，完整dump、flush、
fsync、关闭后才replace；异常只清理自己已创建的临时文件，旧公开文件保持。
保留既有文件mode，新文件遵循普通open的umask；拒绝目标符号链接。
正常序列化、分值、预算、排名和CI算法未改。源码SHA256
`0365f22e42ac257814a43af73df654d4cc6c5add1596b3d9c46489fdd43c388d`。

## 验证与范围

最终10项实际main发布回归，修前9失败/1正常；修后全部E155d普通及`-O`
各67项通过。覆盖写中断、首次写失败、fsync/replace/fchmod失败、硬退出、
读者中途观察、mode、symlink及临时名碰撞归属。
[短日志](../results/atomic-verdict96/verification.txt)、
[输入/环境/日志hash](../results/atomic-verdict96/observed.json)、
[独立复核](../results/atomic-verdict96/independent.md)保留失败及范围。

独立直接前版24组真实入口对照，仅剔除生成时间后整个JSON逐字段相同，
包括路径/预测与治疗hash；87个实际元数据producer几何和51错标检查、
既有24入口/24分制差分通过。评分使用合成值，不是模型精度重评。

Python3.12.14、Clang22.1.3构建、Linux6.18.55/x86_64/glibc2.39；本机无
Torch/Transformers/nvidia-smi，没有GPU、驱动或kernel测量。固定旧判决输入
SHA256为`4505a6364dab5658db36fae7c40f1d7759c3b1e8fb73a660a5e4011ac16455ad`。
没有质量、时延或显存收益主张；文件fsync不代表断电后目录持久化。

这只修公开JSON发布。共享staging并发、路径ABA和旧判决新鲜性仍未解决；
硬退出可留本次临时文件，不能将其当正式结果。未取得真实T2收集快照，
未证明历史冠军/论文已受影响，不改历史数据、推理预算、selector或kernel。
正文/token身份迁移及GPU验收继续按原任务保持未完成。
