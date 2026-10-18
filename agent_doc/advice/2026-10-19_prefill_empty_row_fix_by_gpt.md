# TL-PREFILL-EMPTY-20261019：空行兜底不可污染非空行

## 问题与修复

基线 main `7eb42416288f4f2643442d8044b773e31e97e835`，
`sparse_attn/ops/reference_prefill.py:98-108`。
真实入口为 `eager_prefill.sparse_prefill_attn` 默认 reference 路由，
转入 reference 公共入口，再调用 `_chunk_rows_attn`。

选择 mask 与绝对位置因果 mask 相交后，只要任一行/head为空，旧实现
`if not bool(m.any(-1).all())` 就将对角线 OR 到整个执行tile。
这会给本来非空的行/head增加未选择token，并导致结果依赖执行tile边界。
例如sink=SWA=0，chunk=[4,8)，共享选择仅有位置6：行4/5需自身兜底，
行7本应只读V6，却被改为读V6和V7。零logits、Vt=t时输出6变为6.5。
另一head始终选择位置0，也会被无关空head污染（最大偏差3.5）。

修复按 `[row, Hkv, G-or-1, 1]` 计算empty，只给对应空行/head补对角线。
候选gather实现已有相同按行契约，本轮保持其不变。没有放宽预算，
没有改变组末Q共享选择与逐query语义，也没有新增近似算法或kernel。
去掉了该处tensor-to-bool分支；新增向量mask操作的实际GPU成本未知，
不能凭少一次同步宣称提速。

## 验证及范围

新增 `sparse_attn/ops/tests/test_prefill_empty_rows.py` 经公共入口覆盖
reference/candidate × 共享KV、多KV、per-query head六场景。
独立逐行绝对索引oracle只在空选集时补自身；零logits用V位置平均给解析输出。
修前日志保留3个reference失败，candidate对照通过；修后全部通过。

命令：
```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s sparse_attn/ops/tests -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -O -m unittest discover -s sparse_attn/ops/tests -v
```
两次均 `Ran 67 tests` / `OK`。原始日志位于
`agent_doc/results/prefill-empty-20261019/`，输入为测试中显式构造的8位置张量，
没有外部私密数据；源码/测试/日志SHA绑定于该目录校验清单。
E151c相邻路由/配额21项通过。独立上下文复核1324个CPU场景通过，
包括全空/全选/混合/future-only、GQA/per-query、多种执行tile、保护区、
非连续布局、fp32/fp16/bf16、空/短输入，报告见结果目录independent.md。
归档脚本可在仓库根用 `PYTHONPATH=. python agent_doc/results/prefill-empty-20261019/independent_check.py` 重跑。
本说明不把CPU运行当作生产GPU验收。

环境：Python3.12.14，PyTorch2.8.0+cpu，Linux6.18.55 x86_64，
Intel Xeon Platinum8573C，9个可见CPU，OMP/MKL各1线程。
CUDA=False，无GPU驱动、架构、CUDA编译器或kernel配置可报告。
没有测TTFT/吞吐、显存、kernel数、真实模型质量或计时收益。

## 历史影响与后续

风险限定reference路径、同tile存在空与非空选集且非空行自身未被选中的情况。
标准非零sink/SWA使所有行非空时不会触发旧分支，不能据此判全部历史实验失效。
需复查关闭保护区或使用稀疏/特殊mask的reference实验；candidate既有按行兜底
不受本问题影响。未改历史结果。重测应覆盖实际部署GPU与上述特殊保护配置。
候选内L2、紧凑索引、fused attention以及overlap的生产性能验收仍待可用GPU。
