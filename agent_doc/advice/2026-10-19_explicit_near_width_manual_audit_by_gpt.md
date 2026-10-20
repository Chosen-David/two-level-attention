# TL-EXPLICIT-NEAR-WIDTH-20261019：固定配额子页区宽仍被旧规则放大

## 范围与结论

用户手动触发，重点检查固定配额。重新 fetch 的 main 为
`d7d71009cb8962490b9468c63a66f44d196e4c01`；独立 worktree，持有本机
twolevel-audit.lock（不代表跨机器排他）。保留其他进程写入的字体下载临时文件，
不编辑人类 GUIDE、不修改其他仓库、不停用持续任务。

本轮新增：**合法显式 `0 <= near_len < block_size` 的 host 几何合同不一致**。
完成源码绑定整数分支复现、三行候选补丁与 host 验证；补丁仅作为
`TL-EXPLICIT-NEAR-WIDTH-20261019.pending.patch` 保存，**未应用到生产源码**。
它会改变有效配置的选集，需要真实张量及 CUDA 回归后再发布生产修复。

## 触发、定位、根因

`sparse_attn/arguments.py:304–307` 定义 near_len 为显式 token 区宽。
构造器 `tli_indexer.py:390–463` 接受 nb=0、nt=0/off、nl=0；约束是
nb·bs<=nl、nt<=nb·bs，没有要求 nl>=bs。

但是三个消费点仍把显式宽度套入 legacy 的整页保底：

- `_maybe_build_kmeans`，`tli_indexer.py:1088`：max(bs, int(nl))。
- `compute_score` 的 D′ 省算边界，`:1509`：max(bs, int(nl))。
- `compute_mask` 的 L1/L2 分区，`:2102`：max(bs, int(nl))。

`calibrate_dyn_dgate`，`:607–609` 则使用 max(int(nl),0)，没有整页保底。
这与正常的非整页边界 floor 对齐不同：以下反例全部页对齐，排除舍入和
边界保送干扰。

```text
S=1024, block=64, sink=SWA=64, K1=4, K2=256
near_pages=0, near_tokens=0, near_len=0
构造成功
D′ 标定 far=[64,960)
selector 实际 near_len=64, far=[64,896), near=[896,960)
显式区宽合同预期 far=[64,960), near 为空
```

在普通非 skip 层，L1 near 配额为0且无边界保送，最后这整页不进入候选；
far L2 取满128名后 near 预算为0。该页内高分 token 没有正常竞争机会。
这是源码推导的选集影响，**未运行 tensor topk/mask，也未测模型质量损失**。
nl>=bs 不受此整页保底问题影响；不能据此宣称所有固定配额实验失效。

## 实际复现和候选验证

`tests/repro_explicit_near_width.py` 调用已有完整构造器继承链 loader；再从当前
生产 AST 中提取上述四个整数分支和池左界 property、near_blks 赋值运行。
没有重写 Tensor 算子，没有执行完整 compute_score/compute_mask。
空/短/非对齐/长 S 仅检验 host 整数分支，不能说明完整入口支持这些输入。

矩阵：bs=32/64/128；nl=0/1/bs-1/bs/bs+1/2bs/512；nt=0/off；
S=0/2bs/2bs+1/1024/4097。共210配置、840分支检查。
输入 JSON SHA256：
`b3d3f686319e996b2b32c998ae26695a59f93ee6c66763fa7a0f2beea7cf94d9`。

```text
python tests/repro_explicit_near_width.py
原 main：failures=270, selector_far_hi=896, exit=1
三行候选：failures=0, selector_far_hi=960, exit=0
python -O tests/repro_explicit_near_width.py
原 main：failures=270；三行候选：failures=0
python [-O] -m unittest discover -s tests -p 'test_explicit*budget.py'
三行候选：普通/-O各 Ran 7 tests, OK
```

原 source SHA256 `fc6aa6a40a53cabc521c4130ffb4961a861f3deaf27f675dfb9b74720c12edf2`；
候选 `3b080cd3a4b84bccee19ee6b400f558900d1c56227166c557dc073ca888c19ca`。
恢复生产源码后再次以 -O 复现270错误，保留红证。此审计脚本故意在缺陷仍存
时返回1，文件名不匹配默认 test*.py 回归发现规则，不冒充已通过的回归测试。

候选只移除三个显式 nl 分支的 max(bs,...)；legacy 比例分支及正常页对齐
规则原样保留。构造器合法域中 nl>=bs 的表达式恒等；不能外推整个 GPU
选集的等价性。未采用拒绝所有子页 nl 的替代方案，因当前合同允许此配置。

环境：本地 x86_64 / Python 3.12.14，无可导入 Torch/Triton，PATH 无
nvidia-smi；未使用 CUDA 编译器、驱动或 kernel，无 GPU/模型/速度/显存结果。

## 固定配额性能检查（静态成本，不是测速）

调用顺序 `indexer/base.py:46–47` 先 compute_score 再 compute_mask。
普通非 skip 分支 `tli_indexer.py:1608–1690` 为整个 k_qat 前缀计算并分配
score_fine；L1 候选在 compute_mask 内选出后，`:2323–2328` 才用全宽
where/softmax/归约屏蔽。这意味着固定 K1/nb/nt 目前主要约束选集，
没有让 L2 点积工作量变为候选规模。Triton GQA 复用虽减少重复读，仍处理全前缀。

例如 S=65536、K1=128、bs=64，L1预算页只覆盖8192 token，最多另加两张
边界页；完整L2仍打分65536 token。32 query heads、bf16 时单份
全宽 score 为4MiB（形状算术，非峰值显存），另有 mask/probability 临时量。
不能把规模比直接换成8倍加速：索引、gather、归一化、topk补位、attention
均有成本，选集 ties 与零概率补位必须保持。

另外 `tli_indexer.py:2249` 的 near 分数 clone 后只交给非原地 topk，是
可删复制候选，但本轮没有实测收益，因此未当作性能修复提交。
显式配额当前禁用 batch selector 是已有 TL-BATCH-EXPLICIT-QUOTA 守卫，
本轮没有撤掉守卫或将旧比例 batch 路径视作等价加速。

## 下一步与历史重测范围

先在已授权且空闲的 GPU 上执行候选：nb=0/nl<bs、nt=0/off、skip_far
开关、边界保送开关、短/非整页长度、GQA/head/dtype/stride、prefill→decode
交接、固定前缀替换后缀；核对绝对索引与因果、预算、输出，不只比较计数。
nl>=bs 及 legacy 作为不变对照。先定位历史实际配置，仅重测使用该子页区宽的
实验，不修改已有数据。随后实现显式配额批量选择与候选内L2，在同输入hash、
同预算/语义/设备下测完整成本；继续保持外层chunk2048，不以多stream代替省算。
