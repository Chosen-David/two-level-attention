# TL-PREFILL-SIGMA-CAP-20261019

## 冻结问题与方案

基线 main `60f469e8d085e81904417971f4367128d9d9d5f7`，fresh fetch、独立干净检出；与上次084f6b5相比只新增论文图脚本，保留不动。实验分支e138-ablation只读核验。串行flock；上次长日志/检查点保留在audit-36，不覆盖。已读当前TASK/HINTS/只读人类GUIDE和相关prefill advice，无适用AGENTS。沿用已安装实现优化skill版本，不新增依赖。

真实CPU最小探测：TLIIndexer sigma_select=mid，K2=256、bs=64、sink/SWA=64、L=1025、Q/K=0、V[t]=t、chunk128。prepare_mask最后三次实际选择896/1024/1025个token；reference末行511.9998169（理论512），candidate末行320。candidate无组路径把固定K2推导的容量当作top-sigma选集上界，stable argsort后截W遗漏合法token，且可能遗漏行级SWA。

选定方案：sigma_select!=none时不使用K2静态上界，令w_sel_cap=0，复用现有实际mask计数路径。仅host派发条件修正，不修改selector、阈值/预算、attention/kernel、排序或归约。默认普通top-k保留原静态上界及同步行为；分组路径本已用真实并集宽度，不改。代价是无组sigma每个稀疏chunk增加一次现有sum/max/.item路径；正确性优先，不宣称提速或GPU验证。

替代方案：按全前缀T作为安全上界可避同步但增加pad/gather/score工作；拒绝sigma会丢掉已有合法模式；更改sigma强制K2会改变算法。均不选。未来紧凑索引需GPU公平测量，不混入本补丁。通用非TLI selector的预算声明不在此项范围。

## 任务链与预算

- 先新增实际TLI数值红例及cap派发检查，保留修前输出。
- 修改一处容量选择，普通/-O专项、ops回归与E151c路由通过后独立结果复核。
- 覆盖far/near/mid、普通top-k不变、共享GQA、fp32/bf16、非连续K/V、非整块/单token尾块、默认outer2048、分组路径和空/短输入（相邻回归）。以独立绝对位置选集均值为oracle，不只跟自身reference比。
- 单轮一项有界正确性修复；CPU每条测试最多120s，无GPU/no新增资源，不测速。fresh fetch后显式暂存、普通push或连接工具CAS发布、回读SHA/tree关键源码。
- GPU/模型质量、prefill→decode状态及历史sigma实验重测仍待验证。CPU证据不替代生产GPU，无必要验证的高风险kernel不合入。

## 进度

已复现实际CPU错误；待正式红绿与独立复核。

已完成：修前3方法10个失败，修后3方法通过；ops87+相邻E151c21=108方法普通/-O各通过。独立上下文普通/-O各10case通过，基线负控复现320 vs512；见results目录tests.md、independent_review.md及verify.py。CPU限定host派发修复接受，GPU/性能/模型质量仍待验，任务保持未完成。

发布前再次fetch发现并发main为0e0b10b330afbb9424f66055836c5c67dc6c94d1；仅论文源/PDF/ledger五文件改变，与实现及测试无重叠，已fast-forward保留全部并发修改。发布与远端读回状态保存私有检查点，不预先声称推送成功。
