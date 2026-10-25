# TL-E151C-MODES-20261018：拒绝静默旁路 oracle 配额的选择模式

复现基线 `b8719fa7023837dc724470eeb23795c5557232c1`；发布前于
2026-10-18 UTC fresh fetch main=`b1f50739f0c164db7d380c610f3fc2ebe4402fd2`。
新增提交只改 7 个论文归档文件，`sparse_attn/`、`tests/` 和 prefill 场景脚本
相对复现基线零 diff。在最新 main 的干净独立工作树重放有界补丁，原工作树
与他人未提交文件保留；未合并论文之外的其他改动。
实验分支 `e138-ablation` 核为 `2ed6a1feb78a7b5bcf3cac922b4510c3968714a5`，
仅作对照，未合并。无本地适用 AGENTS.md；读取任务索引与人类 GUIDE，未改 GUIDE。

## 触发、根因与修复

`TLI_E151C_ORACLE_QUOTA` 接受合法数组，不等于选定算法会消费数组。
沿 Qwen3 patch → eager/reference prefill → prepare_mask → compute_mask 核查：
配额覆盖仅在 E64（alpha>0 且 beta>0）数值 gamma 分支的 far 选择中实现。
gamma=None 自由竞争、非 E64 模式、sink/SWA-only、MoBA、sigma 选择会提前返回
或使用其他预算公式。旧构造器允许这些组合，运行结果仍可能被标成 oracle 配额臂。

实际 CPU reference prefill：seed15119、S=512、Hq/Hkv=2/1、D=128、FP32、
K1=8/K2=256、sink/SWA=64、alpha=beta=.5、gamma=None；指定 far=0 与128，
实际均选 far=65、总量256，两个 mask 逐位相同。不是仅凭静态推断。

修复位置 `sparse_attn/indexer/tli_indexer.py:352–365`：在构造期对上述
不兼容组合显式抛 ValueError，包含零配额表，不以表真值漏掉零。
消费端原分支（修后约2317–2362行）不变。选择 fail-closed 而非自动改 gamma
或替换算法；后两种方案会改变实验语义。本轮是正确性修复，不是加速实现。

新增条件只执行于构造期，O(1) 时间/空间；无逐 token 操作、kernel、同步或缓存改动。
chunk2048、GQA、归一化顺序、ties、sink/SWA、预算裁剪保持原实现。
未配置配额时保持原构造行为；合法分区的 gamma=0/1、per-query-head、cluster/
sim-greedy 构造控制通过。后两类仅证明未被错误拒绝，不等于新增 GPU 运行验收。

## 验证入口与边界

结果目录 [e151c-modes-20261018](../results/e151c-modes-20261018/)。
`baseline_runner.py` 在隔离进程加载固定基线模块，不覆盖工作树；先重现上述
真实 selector 结果，再运行新增测试。预期退出1：3个方法中1个方法的18个子例
失败（9种模式×配额0/32），另2个控制方法通过；不是18个独立bug。

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python agent_doc/results/e151c-modes-20261018/baseline_runner.py
python -m unittest discover -s tests -p 'test_e151c*.py' -v
python -m unittest discover -s tests -p 'test_e139*.py' -v
python -m unittest discover -s tests -p test_paper_runtime.py -v
```

修后后三条普通及 `-O` 均执行；包含真实 selector 配额0/32/128、选集及输出
对照、decode交接、异常清理、空/短dense与batch/graph元数据门禁。
E151c 16项、E139 13项、paper runtime 5项，普通及-O各34项通过（见regression.log）。
另 `python test_sparse_prefill_impl.py` 及其-O版本各7场景通过：S800/3000/5000、
非整chunk、前缀切片、GQA/per-q-head、因果及逐行SWA（见prefill.log）。
每种模式共41项/场景，不是全库测试；这些耗时不作性能样本。
发布前重放到 b1f5073 后重新执行固定基线复现及上述全部命令：仍为修前
18 个子例失败、修后普通/-O各41项/场景通过。新运行原始短日志见
`revalidation.txt`（注明摘录规则与完整私有日志 SHA256），未覆盖早期日志。
本次恢复相同 Torch/einops/Transformers 版本后重跑，Python/CPU/OS 与上述
测试环境一致；没有复用旧成功日志冒充本次执行。
独立只读复核未发现阻塞项，并自行重跑16项E151c普通/-O通过；其提醒的D′/空池
优先级及构造控制不等于GPU验收已在本文明确。E139旧测试打印的CPU allocation
proxy不是本轮补丁的内存改善证据。

环境：KVM x86_64、Intel Xeon Platinum 8573C、9可见CPU、Linux6.18.55、
Python3.12.14（Clang22.1.3）、Torch2.8.0+cpu、einops0.8.1、Transformers4.56.2。
CUDA=False；没有GPU/驱动/设备编译器/kernel配置验收。本轮沿用可用的实现优化
skill与unittest/真实CPU selector，不新增第三方依赖或远端资源。
q→k→v连续字节 SHA256：
`df266f3eaba6a69c245a64fc62f22790ecfaabbe8ccee34c5c9940b8cbc202ca`。
脚本及测试哈希见结果目录 SHA256SUMS。模型质量、速度、峰值内存、TTFT未测。

## 历史影响与未完成项

历史 quota 臂若使用被拒组合，需要回查配置、实际选集和路由；不能把此前静默
忽略的配置当作配额已生效。未改历史实验数据。标准数值 gamma/E64 的已有实验
不因本守卫自动失效；既有路由修复的GPU/真实模型重测仍未完成。

这不是“quota总会精确选到指定数量”的承诺：D′ skip_far 优先砍far，空far池、
短dense前缀与decode不会强行消费prefill配额；容量裁剪和零概率补位仍按既有契约。
本轮保留这些优先级，不扩大拒绝范围。短表与延后注入layer_idx的校验仍待办；
完整alpha/beta/gamma数值域验证也不在本补丁内。
候选内L2、紧凑索引、fused indexed attention/overlap的公平GPU测速仍待可用资源，
不得将此次构造期校验称为性能改善。
