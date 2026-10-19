# TL-PREFILL-SIGMA-CAP-20261019 独立复核

结论：冻结方案与实现一致；在本次限定 CPU/host 派发修复范围内无阻塞发现。未改生产代码、selector、kernel 或人类 GUIDE，未访问远端或发布。

## 静态路径核验

- 基线：`60f469e8d085e81904417971f4367128d9d9d5f7`。
- `tli_indexer.py` 的真实 sigma 分支通过阈值比较选择 token；far/near/mid 均非 K2 基数上界。固定 K2 推导 cap 后截取稳定排序结果，确实可遗漏合法选择及末端保护带。
- 实现只在 `getattr(indexer, "sigma_select", "none") != "none"` 时置 `w_sel_cap=0`，并更新一处对应说明。缺失属性和普通 top-k 保留原静态 cap；分组路径原已由 `sel=None` 传 cap=0，不受改变。
- count 路径计算 `max_h |sel_h|`。任一行子块的可见并集不超过 `|sel_h| + sink + R + swa`，因此现有 W 公式是安全上界；实际count可增加 R 的内存预算准确性。额外 padding 仍被 valid 屏蔽。
- 代价是无组 sigma 每个稀疏 chunk 的现有 `.sum().max().item()` 同步；count 更大时实际合法工作量可能更多。没有性能改善或 GPU 可用性结论。

## 独立 oracle 与短日志

独立 `verify.py` 使用真实 TLIIndexer，记录原始 token mask；不用 reference attention、token_selection 或 compact/index 构造逻辑作为 oracle。Q/K=0，V 为绝对 token 位置，每行期望值由 Python 绝对位置集合和实际 dtype 输入值的算术均值计算。bf16 允许 4 的绝对误差，fp32 允许 0.002；bf16 仅用于支持集回归，不声称高精度质量验证。

环境：torch `2.8.0+cpu`，`cuda=False`，线程数1。

```text
PYTHONPATH=. timeout 120s python agent_doc/results/prefill-sigma-cap-20261019/verify.py
10 cases: PASS; exit 0
PYTHONPATH=. timeout 120s python -O agent_doc/results/prefill-sigma-cap-20261019/verify.py
10 cases: PASS; exit 0
```

两种解释器模式结果一致：

| Case | 末行实际 / oracle | 最大绝对误差 | cap集合 |
|---|---:|---:|---|
| mid, L1025, chunk128 | 511.9998169 / 512 | 0.0005493164 | {0} |
| far, L1025, chunk128 | 484.7211914 / 484.7211609 | 0.0003662109 | {0} |
| near, L1025, chunk128 | 593.2485352 / 593.2484741 | 0.0006103516 | {0} |
| none, L1025, chunk128 | 678.2734375 / 678.2734375 | 0.0005798340 | {0,512} |
| mid, 4Q/2KV, strided K/V, budget4096 | 511.9998169 / 512 | 0.0004882813 | {0} |
| mid, bf16 | 512 / 512 | 2 | {0} |
| near, group64 | 593.2484741 / 593.2484741 | 0.0003967285 | {0} |
| mid, L2051, chunk2048 | 1024.9998779 / 1025 | 0.0011596680 | {0} |
| mid, L0 | N/A | 0 | {} |
| mid, L17 | 7.9999995 / 8 | 0.0000009537 | {0} |

低预算 case 通过 wrapper 显式传入 `_row_block_size(..., budget=4096)`，确实触发多行子块，未依赖修改已绑定的默认参数常量。L1025 包含单 token 尾块；mid/far/near 的末行真实 mask count 分别为1025/832/833（每KV头）。

另外仅读取 `git show 60f469e:sparse_attn/ops/eager_prefill.py` 并在内存 module 执行，将同一独立 oracle 的 eager_prefill 替换为基线模块：

```text
baseline mid L1025: cap={0,512}, final=320, oracle_final=512,
max_abs_error=192, pass_numeric=False, pass_cap=False
EXPECTED_BASELINE_FAILURE (negative-control harness exit 0)
```

基线负控载入方式：`types.ModuleType('sparse_attn.ops.independent_baseline')`，设 `__package__='sparse_attn.ops'` 后 `exec(compile(git_show_source,...), module.__dict__)`；通过 importlib 加载 verify.py，赋 `verify.eager_prefill=module`，线程设1，调用 `verify.check('mid')`。没有检出或覆盖任何生产文件。

## 输入与源码身份

SHA256（字节哈希不含形状；形状/dtype/stride由脚本case参数定义，完整每case哈希亦由脚本JSON打印）：

```text
eager_prefill.py bd05914d4e7f6471f2287956b665ba4138d140663e4a83654eafc98be9752352
tli_indexer.py 4fd73e879ed4011dbacd37c3318304eaddd6a446738b2fdbe3b71d7903dbdb1a
verify.py 7965f3c20d0d9cc20ac5a6904a26a9213841d2a5f4bbab3da074cc2c4d992f56

L1025 fp32 2Q/1KV (mid/far/near/none/group64):
Q c4aca32864436a55dd96b60917b04b84ad22dd2765937129234b22782b9d4fd7
K 3f5acc4d73df7197ea151ac5fc829824a0fe1be817850d490d5c4c8b2f6ef800
V 7fb22e0e0bf59980ffc982a1bf4cf7f173d017f6c5cdfc59263684046157afc5
L1025 fp32 4Q/2KV strided:
Q 3a5680463a171305a81d54e1830470dd20bb0f57260fd7a3c5f179e9e1fad160
K c4aca32864436a55dd96b60917b04b84ad22dd2765937129234b22782b9d4fd7
V d90adfbdacd3d9665a0530d765589fdb02186241a18a4ff330c897926a1b06f3
L1025 bf16 2Q/1KV:
Q 3f5acc4d73df7197ea151ac5fc829824a0fe1be817850d490d5c4c8b2f6ef800
K 0249c5b2b4116c7167909be4b4ec87c5f9972cd2b1ba7b8d1dd457ebfddba085
V 7ea37c61a7bffaec40b4d1bf4974d05b3ddbe45641f80145555989afdc47b4e2
L2051 fp32 2Q/1KV:
Q c6d5f1e4877e8bb76cdbc01bacd43eea44392da94af4f5edd47ba0bae49d81c2
K b99a683eacd27c7b6b673e69406d83ecdb1ddfde49beddce8ed1f96bfeb925f8
V 9ade8bf85dcb6e2716ef7682888eb117690f1111e8fd2d386d84ec0b1bbeaf39
L0 Q/K/V e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
L17 fp32 2Q/1KV:
Q 3f1f6f76c52276c865bae097486a0ce164cd509c98c6410b677f516084ad7c3c
K e8b31e302d11fbf7da124b537ba2d44f88e165da03c6557e2b0f6dc486e025bb
V 57f69682190da328636d0184ffcbb7d08e5c94efc4cc91827632cf9b252e465e
```

过程透明性：首版 scalar tensor oracle 较慢，后将mask和值一次转为Python列表；中间一轮在已完成8个数值case后，因空张量 byte view 的stride错误退出，属独立哈希辅助器问题。修正为空字节哈希后，普通和-O完整10case均通过；报告引用最终脚本结果。

## 验收边界

这是支持集截断修复的 CPU 正确性证据，非生产 GPU 验收、吞吐/显存测量、模型质量评估、历史sigma实验结论更新或prefill→decode状态证明。通用非TLI selector预算声明不在本改动范围。建议在上述边界明确保留的前提下接受此补丁。
