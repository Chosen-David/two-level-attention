# TL-E155D-STAGING-20261021：T2暂存快照按调用隔离

基线`143ff2cf39fe5012d929afe81d71229f054c3f4d`。
旧`exp/trace/analyze_e155d_grid.py:204–209`在每次打分前删除整个STAGING，
随后复用相同预测路径。第二次或交错运行会删除/覆盖前次评分及CI仍需的
快照；即使前次已发布，其verdict.files也可能变成后一次字节或悬空引用。
这是上一份原子发布简报明确保留的缺口，不是GPU选择器问题。

修复在当前`:203–213`：非空打分调用用mkdtemp独占创建`STAGING/run-*`，
所有cell和返回files使用该目录；不删除任何既有快照。空集合不动目录。
保留预测/manifest哈希复核、排名/配对CI算法及最终JSON原子发布。
源码SHA256`5bb6cf15651a32e226dd0c464efeaa55b4daf4ddd6c5fa83405abe9423d4a236`。

## 验证与边界

新增5项回归修前1失败/4错误，覆盖重复打分、交错打分、其它目录所有权、
失败及空输入保留旧证据。修后E155d普通与`-O`各72项通过。
直接前版24组真实main对照在普通/`-O`各通过：仅归一化生成时间和预期改变的
暂存父目录，保留文件名、预测/治疗hash及全部其它JSON字段；排名和CI相同。
另通过既有24快照/24分制对照及87合法producer几何、51错标拒绝检查。
评分为合成stub，不是LongBench真实质量或T2重评。
独立复核另以两个真实OS进程交错打分：A暂停、B完成退出、A恢复，两份
快照均保留；删源后20份prediction与20份manifest哈希仍相符，两份CI可读。
启动异常及损坏result不破坏旧文件。此额外实验仍使用stub，不扩大质量结论。
[实际短日志](../results/staging97/verification.txt)、
[输入/环境/文件hash](../results/staging97/observed.json)、
[独立复核](../results/staging97/independent.md)。

Python3.12.14/Clang22.1.3/Linux6.18.55/x86_64/glibc2.39，无Torch、
Transformers及nvidia-smi；未用GPU/远端作业，不报告速度/显存/模型收益。
24组输入SHA256`3aaf366af9095b401cb7517f1767b986d99093e03091adf1b3be7d9f3125f8f9`。

快照有意保留，包括失败运行；磁盘会积累，mkdtemp目录按0700创建。
清理应由所有者确认无活跃读者及需保留的verdict引用后执行，本补丁不自动GC。
同名最终OUT仍是最后一次replace获胜，不保证较新收集快照最后发布；没有
把整个分析变成事务，也不防外部主动改/删证据、断电或collector跨文件ABA。
现有旧判决被旧版删除的快照不会自动恢复；未取得真实收集快照，历史影响
及必要重评未知，不更改旧实验数据。正文/token身份迁移及GPU验证仍未完成。
