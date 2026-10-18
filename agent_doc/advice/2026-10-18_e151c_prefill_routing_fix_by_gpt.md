# TL-E151C-ROUTE-20261018：prefill 配额随运行路径丢失

基线：`44d09d7781d235db94d7ec5ffe5a90b529221af3`，2026-10-18 新独立检出。
相对上次 `b9988a0` 只有论文文件更新，选择器源码没有变化；实验分支
`e138-ablation` 当前为 `2ed6a1feb78a7b5bcf3cac922b4510c3968714a5`，
该分支没有 `reference_prefill.py`，不能把分支上的配额实验当作当前 main 路由证据。

## 触发与根因

真实入口 `sparse_attn/patches/qwen3_attn_patch.py:173–183` 在无 padding、B=1
且 `TLI_SPARSE_PREFILL=1` 时调用 `ops/eager_prefill.py:313–319`；默认运行路径是
reference。`TLIIndexer.__init__` 读取 `TLI_E151C_ORACLE_QUOTA` 后，只有
`_e151c_on=True` 时 `tli_indexer.py:2300–2302/2331–2333` 才覆盖 far 预算。

1. 原 `reference_prefill.py:190–192` 未置标志，配置配额仍按旧 γ 预算运行。
   实际 CPU TLI 输入 S=512、sink/SWA=64、K1=8、K2=256、α/β/γ=.5：
   far 区为 [64,256)，配置 far=32 或 128，原 reference 实际都选 0。
2. `batch_select.py:_gate_ok` 没有排除该配置，而批量选择直接算自身的 L2
   预算，不调用 `prepare_mask`；candidate 开启组选择与批量选择、满足 CUDA
   等门禁时可能绕过配额。该门禁缺口以纯设备元数据测试复现；未执行 CUDA
   批量 kernel，不能把它说成已取得 GPU 输出错误实测。

## 修复与选择

- `reference_prefill.py:190–202` 使用与现有 candidate 一致的 prefill 激活范围，
  `finally` 清零，正常返回与异常均不把配额状态泄漏给后续 decode。
- `batch_select.py:257–261` 检查已加载配置是否非 None，而非尚未激活的临时
  `_e151c_on`，不支持时沿原有 None 回退逐组选择；要求 graph 的调用仍报不支持。
- 批量/graph 内新增配额算法为替代方案，暂不采用：需要新增预算与缓存身份验收，
  当前无 GPU。这里复用已有逐组路径，时间成本可能增加，未测速。

未改变外层 chunk2048、算法选择粒度、候选/L2 排序、预算公式、sink/SWA、GQA
或 Triton kernel。逐组候选内 L2、紧凑绝对索引及 overlap 性能优化仍待设备验证。
原“组末 query 共享选择”仍不是逐 query 论文语义或 prefix-invariant 保证。

## 实际证据与复现

证据目录：[`../results/e151c-route-20261018/`](../results/e151c-route-20261018/)。
测试 `tests/test_e151c_prefill_routing.py` 的最终相同版本：

```text
before-final.log: Ran 10 tests; FAILED (failures=10), errors=0
after-final.log:  Ran 10 tests; OK
after-final-O.log: Ran 10 tests; OK
```

10 个 failure 包含 subTest 失败，不能称为 10 个独立生产 bug。
覆盖真实 selector far 配额 0/32/128、总量 256、保护区必选、decode 状态交接、
异常清理、无配额控制、空/短 dense、group64 CPU 回退、reference/candidate
选集逐位一致及输出 rtol=1e-5/atol=1e-6。仅纯门禁检查使用 CudaMetadata 替身，
不将替身传入算子。首版 graph 测试替身缺 device 引发 1 个测试设施错误；原始
日志留在私有检查点，改为空组的 fail-closed 控制后重跑基线，最终证据无该错误。

相邻回归：E139 graph/GQA 13 项、paper runtime 5 项、原 prefill 7 场景，
普通及 `-O` 均通过（共 35 项/场景，每种模式一次）。原 prefill 覆盖 S=800/3000/5000、
GQA/per-query-head、非整 chunk、前缀切片、因果和逐行保护区。不是全库测试。

命令（在仓库根，以安装了 Torch/einops/Transformers 的 Python）：

```sh
python agent_doc/results/e151c-route-20261018/baseline_runner.py
python -m unittest discover -s tests -p test_e151c_prefill_routing.py -v
python -O -m unittest discover -s tests -p test_e151c_prefill_routing.py -v
python -m unittest discover -s tests -p 'test_e139*.py' -v
python -m unittest discover -s tests -p test_paper_runtime.py -v
python test_sparse_prefill_impl.py
```

baseline_runner 需要上述基线 git 对象，预期退出 1；仅在隔离进程加载原模块，
不覆盖当前源码。其余命令 `-O` 重跑亦执行；E139 的 `-O` 运行额外设置
OMP_NUM_THREADS=1/MKL_NUM_THREADS=1，因此日志时长绝不用于性能比较。

环境：KVM x86_64 / Intel Xeon Platinum 8573C（9 个可见 CPU），Linux 6.18.55，
Python 3.12.14，PyTorch 2.8.0+cpu，einops 0.8.1，Transformers 4.56.2；
专项内 Torch 线程数 1。CUDA=False，无 GPU 架构/驱动/kernel 编译或性能数据。
固定 q/k/v：seed15118、FP32、[1,512,2,128]/[1,512,1,128]/[1,512,1,128]，
按 q→k→v 连续原始字节串联 SHA256：
`d387dbabd615b83d52c903424342b6a9f4ff4e079e4c5436cca926325de8a86e`。
文件哈希见结果目录 `SHA256SUMS`；没有真实模型质量、TTFT、吞吐或显存收益主张。

## 历史影响与下一步

配额已配置却走 reference，或走符合批量门禁的 candidate batch/graph 的历史
实验，需先回查真实路由与配额生效记录，不能继续仅凭环境变量将其标作 oracle
配额臂。原 candidate 逐组路径已经激活标志，不自动判污染；无配额实验不受本修复
影响。未改旧测量、未重跑模型、未发现/控制任何已绑定远端在飞任务。

下一位授权 GPU 执行者需按原输入/预算/dtype 重验 candidate 批量请求确实回退、
逐组 mask 与 decode 交接、完整选择+attention 成本和真实质量。非法/过短 quota
表的构造期校验、γ-off/非 e64 配置是否允许配额仍属未覆盖契约，不在本补丁承诺内。
