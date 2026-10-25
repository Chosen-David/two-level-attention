# TL-DECODE-CU-STRIDE-20261019：长度数组布局与 fused ABI 不一致

## 问题、触发与修复

基线 `bb14bc8a0347f1e440c78d397a3e2019e4a16c82`。
`sparse_attn/ops/eager_decoding.py:162-164` 的 kernel A 固定读 `CUK`
和 `CUK+1`；`:386-387` 的 wrapper 只在非 int32 时转换类型。
所以本来就是 int32 的间隔 view 不会整理，逻辑第二元素不在 `CUK+1`。

实际链路是 `eager_decoding_attn` → `_eager_decoding_attn_gather` →
`_eager_gather_triton` → `_eager_expand_kernel`。T≥16384、block_size=1、
符合既有 GQA/KV 布局和 W×8≤T 等 fused 门时，stride=2 的 cu 可以进入。
测试保留真实 host 链路，仅将两个 kernel 替换为 CPU 地址 ABI 探针。

两个方向均已复现：

- backing `[99,0,777,16384,888]` 的 `[1:4:2]` 是合法 `[0,16384]`，
  但相邻寻址读到 `[0,777]`，错误记录 code3，导致请求校验误拒绝。
- backing `[0,16384,16383]` 的 `[::2]` 是非法 `[0,16383]`，
  相邻寻址却读到 `[0,16384]`，漏掉本应记录的 code3。

最小修复在转换之后、分配/launch 之前（修后`:388-391`）检查
`stride(0)!=1` 并 `.contiguous()`。连续 int32（含非零 storage offset）
保留对象身份、不调用 contiguous；原 int64 转换不变。没有改 kernel、
预算、选择集、chunk2048、近似算法或 stream。

## 证据与边界

新增 `sparse_attn/ops/tests/test_eager_cu_layout.py`，3个方法覆盖 public
和直接 wrapper、int32/int64、间隔/连续、非零 offset、非法逻辑长度。
`agent_doc/results/decode-cu-layout-20261019/reproduce.py` 从固定基线
读取旧模块，使用同一测试进行修前/修后对照；reproduce.log 为实际输出。

```text
baseline tests 3 errors 2 failures 2
patched tests 3 errors 0 failures 0
python -m unittest discover -s sparse_attn/ops/tests -q: Ran 70 tests / OK
python -O -m unittest discover -s sparse_attn/ops/tests -q: Ran 70 tests / OK
```

前两个 errors 是合法 cu 误拒绝，两个 failures 是非法 cu 未拒绝，
均分别来自直接/public 调用，不冒充4个独立测试方法。
独立上下文复核额外30布局（两dtype×五stride×三offset）通过，输入storage
未变、连续int32身份未变；亦复核了新增反向非法长度测试。
地址探针读取 `as_strided((2,), (1,))`，对应kernel的实际地址公式，
但不是执行Triton，不证明GPU数值、异步行为或性能通过。

复现：仓库根执行 `PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python agent_doc/results/decode-cu-layout-20261019/reproduce.py`。
合法最小cu逻辑字节SHA256为
`45055f7d65263fdd23060bb67886e2269ca996e23f5d9423ee2c445432dbfe73`，
backing/stride/offset见上文；其余输入由测试确定性构造，Q全1、K/V全0，
T=16384/Hq=2/Hkv=1/D=4、float32、mask仅首token为True、W=32。
文件哈希清单绑定实现、测试和证据。

环境：Python3.12.14，PyTorch2.8.0+cpu，Linux6.18.55 x86_64，
Xeon Platinum8573C，9可见CPU，OMP/OpenBLAS=1。
无GPU/CUDA驱动/架构/编译器可报告。kernel参数未变：expand TP=1024/8 warps，
attention BW=256/4 warps，仅源码配置，未设备执行。
没有模型质量、TTFT、吞吐、峰值显存、kernel数或速度收益测量。
非连续cu新增小复制的真实GPU开销未知，不宣称性能优化。

## 历史影响与后续门禁

生产 `sparse_attn/patches/utils.py:8-10` 的 prepare_cu_seqlens 使用 cumsum，
通常生成连续int64数组；未发现常规实验受本问题污染，不否定历史结果。
需重查自定义cu视图且实际派发fused的实验，不能仅凭无异常认可其长度契约。
GPU待验：相同合法逻辑值的连续/间隔输入应输出一致，非法逻辑长度须flag3；
核验实际fused派发、请求末尾校验和正常int64路径，然后测完整成本。
本补丁不完善cu形状/dtype/device的全部非法输入守卫，也不解决int64到int32
溢出问题；这些须独立保留复现与证据后另行处理。
候选内L2、紧凑索引、fused prefill及overlap优化继续等待可用GPU实测，不冒充完成。
