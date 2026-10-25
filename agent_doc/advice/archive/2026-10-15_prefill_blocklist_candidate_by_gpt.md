# Prefill 活跃块遍历实验候选（GPU 待验证）

问题 ID：TL-PERF-BLOCKSCAN-157。分支：`perf/prefill-compact-block-traversal`。
基于实验分支 `e138-ablation@e1edf9c316fc4190dd0c29bc5d47477136f8bdab`，
不是 main 的实现或性能结论。发布前核验 main 为
`90134a435396d7a352dd6d6dc90099567b50eef3`；本次不更新 main。

## 瓶颈与改动

基线 `sparse_attn/ops/fused_attn_triton.py:91` 的每个 query tile 循环
`range(0,T,BK)`，即使整块不可见，仍需加载位图、归约、判断。
以生产参数 G=4、BRG=64、Hkv=8、chunk2048、T=65536 为例，单 chunk
有 128×8 个 query programs，每个检查 1024 个 K blocks，共 1,048,576 次
块判定。这是控制流工作量，不是实测耗时，也不是点积次数。

新增 `_pack_group_blocks`：每个选择组/KV head 在 GPU 上用前缀和压缩活跃
64-token 块，按绝对位置升序存储索引和设备端 count。attention 只遍历这些
索引，并继续执行原来逐 query tile 的 `need` 判断。保持原先在线 softmax
的实际执行块顺序，sink/SWA、逐行 causal、选集和预算不变。chunk2048、
group64、BRG64 等调度参数均不改。不改 selector，也不引入双 stream。

同例 group64 有 32×8 个 pack programs，扫描 262,144 个块标志；随后
attention 扫描数量取决于活跃块数。**不能直接把循环次数减少折算成加速比**。
pack 是额外 kernel，缓冲仍按最大容量分配：单 chunk 索引加 count 为
1,049,600 bytes。短序列/稠密选集可能回退性能，需要实测后决定派发阈值。
当前仍保留 token mask 的 pad/stack/any，也仍计算活跃块内未选择的列。

开关 `TLI_PREFILL_BLOCK_LIST=1` 默认关闭；调用现有 prefill 路径还需原有
`TLI_PREFILL_FUSED_ATTN=1` 和组选择配置。支持 BK=BLK、grp 可整除 BR、
最多4096个块且每组一个 mask；不支持时抛错，由原调用者回退。
直接测试/测速使用显式 `use_block_list=True/False`，不经过捕获异常的调用者。

## 验证证据与边界

本机 Linux x86_64，Python3.12.14，PyTorch2.8.0+cpu；无可用 CUDA/Triton。
本次未编译、执行或测量 GPU kernel，没有性能收益或显存峰值结论。

命令：
```sh
PYTHONPATH=/tmp/twolevel-cpu-deps:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest test_prefill_blocklist -v
```
短日志：
```text
test_packing_and_attention ... skipped 'requires CUDA and Triton; CPU is not kernel validation'
test_sorted_group_superset_retains_exact_tile_sequence ... ok
Ran 2 tests in 1.785s
OK (skipped=1)
```

CPU 几何模型覆盖1512组配置、32238次 tile 序列比较；seed=20261015，
长度1/63/64/65/127/129/2048，起点0/1031/64000，grp64/128，G1/2/4/8，
sink/SWA=(0,0)/(1,1)/(128,128)，空/满/随机选集。该模型检查原 `need`
谓词在组范围压缩后能保留逐 tile 执行序列，**不是 Triton 正确性证明**。
新实现、测试、基准脚本均通过 Python AST 解析和 `git diff --check`。

GPU 测试已提供：直接核验 pack count、排序索引、未写入尾区，以及 fused
baseline/candidate 输出严格相等；覆盖fp16/bf16、D64/128、G1/4/8、尾组、
空选集、零保护区及包含未来 token 的选集。未执行项仍是未验证。

## 目标 GPU 验收

```sh
python -m unittest test_prefill_blocklist -v
compute-sanitizer --tool memcheck python -m unittest test_prefill_blocklist.GPUChecks -v
python benchmark/prefill_blocklist.py --length 65536 --rows 2048 --iters 21
```

有工具时再执行 racecheck。测试的输入由固定 seed 与完整形状生成；GPU
benchmark 输出实际输入 SHA256、kernel SHA256、Git SHA、驱动、框架/CUDA/
Triton版本、设备/架构、dtype/shape/stride与kernel配置。JIT/冷启动单列，
预热后交替顺序测设备时间和同步 wall time，保存每次样本、峰值增量显存；
另行 profiler 统计 kernel 数并检查 pack kernel 确实出现，避免静默回退。
不下载模型或启用付费资源。

基准范围是**合成块稀疏选集的 wrapper，含打包和attention，不含 selector**。
不能将其当完整 prefill/TTFT；还需相同机器同输入比较原 e1edf9c 完整链路
与本分支，覆盖8K/16K/32K/64K、实际选集、质量、dense高效后端、prefill→
decode交接，并比较单流wall与已有Graph口径。只有获得完整链路净收益后，
才考虑默认启用或合入 main。

本候选不修订历史实验数据，既有1.64x实验结果不能归因于本改动。
未实施更高风险的候选内L2融合、跨tile流水或共享缓存并发改造。
