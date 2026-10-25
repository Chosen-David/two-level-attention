# Prefill top-σ 可变选集被固定 K2 打包容量截断

问题 ID：TL-PREFILL-SIGMA-CAP-20261019。代码基线 `60f469e8d085e81904417971f4367128d9d9d5f7`。fresh fetch后与上一轮 `084f6b5` 对照，新增仅论文图脚本；未修改该并发工作。实验分支 e138-ablation `2ed6a1feb78a7b5bcf3cac922b4510c3968714a5` 只读核验，未合并。

## 真实路径与根因

Qwen3 patch (`sparse_attn/patches/qwen3_attn_patch.py:168–186`) → sparse_prefill_attn → TLIIndexer.prepare_mask → compute_mask的sigma分支 (`tli_indexer.py:2149–2189`) → candidate无组gather。

top-σ按阈值选token，far/near/mid三种模式的基数均不受K2限制。原candidate在 `eager_prefill.py:387–390` 用K2+块对齐余量+sink/SWA推导w_sel_cap；`_chunk_rows_attn:167–170,210–216` 用它推导W，再对stable argsort的选中token序列取前W列。上界过小时，后部合法选择及自身SWA会被截掉。这不是浮点允许误差，也不是上一轮非法metadata：实际selector返回完整合法token mask，但消费端删掉了选集。

真实CPU探测：TLI sigma=mid、K2=256、bs=64、sink/SWA=64、L=1025、H/Hkv=2/1、D=128、Q/K=0、V[t]=t、chunk128、dense_below128。最后三次真实mask选中896/1024/1025个token。理论最后行均值512；修前reference为511.9998169、candidate为320，最大行误差192。最后单token尾块的W只有641，stable压紧后仅保留前641个token，均值正好320。

默认outer2048也存在同类问题，新测试L4097覆盖，不以缩小chunk作为修复。默认reference和group>0路径不是本bug的错误范围：后者已使用实际并集宽度。

## 修复与取舍

`eager_prefill.py:391–395` 在 `sigma_select!=none` 时传w_sel_cap=0，复用原有 `sel_e.sum(...).max().item()` 得到真实选集上界，然后沿原gather/causal/逐行SWA计算。只改变host分发条件并更新注释，selector、阈值、K2、排序、归约、chunk2048、kernel、_inc缓存、流和Graph实现均未修改。普通top-k（包括对象无sigma属性的既有调用者）保持原静态cap。

这是正确性修复，有必要成本：无组sigma每个稀疏chunk增加一次已有计数归约和host标量读取；恢复被漏掉的token还会增加实际计算/内存。GPU耗时与显存尚未测量，不能称提速。使用全前缀T静态上界虽避同步但可能放大pad/gather/score浪费；把sigma强行截到K2会改变算法；拒绝sigma会取消已有合法模式，均未采用。

## 实际验证

新增 `sparse_attn/ops/tests/test_prefill_sigma_capacity.py`，先红再修：3测试方法出现10个失败（far/near/mid×fp32/bf16无组数值6项、容量派发3项、默认2048数值1项）；合法分组数值子例及普通top-k派发当时通过。修后全部通过。

完整ops87项及相邻E151c21项，普通/-O各108项通过；独立新上下文普通/-O各10case通过，另以原源码内存载入负控独立复现320 vs512。独立复核未发现本有界修复的阻塞项。

数值oracle直接读取真实TLI mask，逐行组合绝对位置因果及sink/SWA，再对V算均值（Q=0），不复制candidate打包实现。fp32 atol/rtol=2e-6，bf16=0.004，在修复前固定；不依赖与自身reference对拍。覆盖H/Hkv=4/2、非连续Q/K、expanded V、L1025尾块、三种σ、group0/64、默认outer2048 L4097。既有ops覆盖空/短输入、mask metadata、保护带和非整块；相邻E151c测试保护真实TLI普通top-k路由。

命令、原始短日志、hash和独立复核见 `agent_doc/results/prefill-sigma-cap-20261019/`。环境：Linux6.18.55 x86_64、Intel Xeon Platinum8573C（容器9CPU）、Python3.12.14/Clang22.1.3、torch2.8.0+cpu；OMP_NUM_THREADS=1，命令各120s预算。CUDA不可用，未调用历史远端GPU；驱动、GPU架构、新kernel编译配置不适用。按实现优化skill保留红绿证据及独立复核，不安装新工具。

源文件 eager_prefill.py SHA256：修前 `adbc05a67f0653a616ec1d8f320248c2f1fa13c3acf47ecfab065a73f1780654`，修后 `bd05914d4e7f6471f2287956b665ba4138d140663e4a83654eafc98be9752352`。修前日志SHA256 `00cd00adbd985bfa857a093078ec1eaa02365d7f39159adf4f16517cb334473f`，保留本地原始日志、不提交冗长traceback。

## 历史影响与后续

仅需优先重查启用candidate、SELECT_GROUP=0、sigma_select非none，且实际选集/行保护并集超过静态W的prefill样本。此前decode-only的E87数据、默认reference、普通top-k和已按实际并集取W的group模式，不因本bug自动失效；仍需按各自证据门禁判断。无历史数据改写。

CPU通过不代表GPU/真实模型/端到端质量验收。GPU性能、峰值显存、kernel数、host同步代价、TTFT仍未测；固定前缀替换后缀及prefill→decode缓存状态不是本次修复对象。后续在已授权GPU上同预算重跑受影响sigma样本，再比较实际计数与安全静态上界/紧凑索引方案，不能把此前少算token的耗时当合法基线。
