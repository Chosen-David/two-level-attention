# Compact far skipping、KV eviction 与 direct partitioned prefill

本路径面向受支持的 Qwen3 TLI decode：被冻结为 `skip_far=True` 的层从页统计、
L1/L2 打分和最终注意力中物理排除 far KV，只保留固定 sink 与有界 Near/SWA
环形存储。普通 non-skip 层仍执行原有 TLI 全历史索引并保留完整 linear KV。

## 默认解析与启用

两个选项的 CLI 缺省均为 `auto`：

```text
--tli_far_skip_backend auto --tli_far_skip_cache auto
```

入口在模型/config 已知后、treatment snapshot 冻结前统一解析：

- 受支持的 TLI Qwen3（目标层 `head_dim=128`、无 native sliding window），且已
  显式配置固定 `near_len/near_pages`、`near_tokens=off` 及 compact 支持的普通
  shared-head 两级选择时，解析为 `compact/evict`；缺省几何或其他不兼容选择解析为
  `legacy/retain`，避免先解析成功再在 indexer 构造时失败；
- 其他模型或方法解析为 `legacy/retain`；
- 显式 `compact`/`evict` 遇到不支持的配置仍 fail-closed，不静默降级；
- 缺字段的历史 Namespace 保持 `legacy/retain`，旧 treatment identity/hash 不变；
  未解析的 `auto` 不得进入 snapshot 或 manifest。

默认 compact **不等于**默认打开 layer skip。生产冠军口径仍须显式传
`--tli_enable_layer_skip false`；只有静态 skip mask 或动态 gate 选中的层才驱逐 far。
例如：

```bash
--tli_far_skip_backend compact --tli_far_skip_cache evict \
--tli_near_len 4096 --tli_near_pages 32 --tli_near_tokens off \
--tli_sink_tokens 4 --tli_swa_size 128 \
--tli_enable_layer_skip true \
--tli_layer_skip_path exp/trace/results/tli_layer_skip_mask.json
```

动态 request-local 标定使用：

```bash
--tli_enable_layer_skip false --tli_dyn_dgate --tli_dyn_dgate_eps 0.05
```

`retain` 是保留完整 KV 的 compact 消融臂；`evict` 在 gate 冻结后释放 far
存储。显式 `legacy` 保留历史行为。

## Direct partitioned prefill 生命周期

对 compact+evict，生产静态入口默认采用：

```text
prepare_direct_prefill → one-shot dense prefill → finalize_direct_prefill
→ decode / CUDA Graph capture
```

该路径只接受 B=1、unpadded、eval/no-grad、一次完整 prompt 的 dense prefill。
chunked/padded/beam/offload/grad、自定义非 None prefill mask、未 finalize 就
decode、重复 finalize 或 prefill 后 gate/geometry 改变均 fail-closed。普通
`TLIStaticLayer.update()` 的 decode-only 守卫没有放宽。

每层 staging `update()` 直接写最终物理分区，同时把**原始完整 K/V 原样返回**
给 dense attention；attention 不消费 static/ring 拼接视图。因此 RoPE、dense
SDPA、动态 gate 标定和 prefill logits 的数学语义不变。staging 期 HF 使用
unpadded SDPA 的 `is_causal` 路径，避免构造二次方 static mask；finalize 后恢复
固定容量 decode mask 契约。

旧 `DynamicCache → fill_static_from_dynamic()` conversion 路径保留用于 legacy、
兼容调用和 direct-vs-conversion 对拍。

### Sink/Far/Near-SWA 所有权

设当前 prompt 逻辑长度为 `L`，`tail_start` 为 Near/SWA 起点：

- staging linear 独立持有 `[0, tail_start)`，即 Sink/Far 前缀；
- staging ring 独立持有固定地址 sink 前缀及 `[tail_start, L)` 的 chronological
  Near/SWA tail；
- `skip_far=True`：finalize 直接从 ring 构造 `FarSkipStaticLayer`，释放整份
  linear Sink/Far staging。decode 只继续写 ring，不再保留或复制 far；
- `skip_far=False`：保留完整 linear 历史。若 staging 因动态/未知 gate 建了
  ring，则把 chronological tail 装入 linear `[tail_start:L)` 后释放 ring；
  已确定 non-skip 的层在 prefill 前直接省掉冗余 ring。

因此，skip 层实现了“摘除 Far 节点并释放其独立存储”；ring 内另有固定 sink
副本，不依赖被释放的 linear。non-skip 层为兼容现有 L1/L2、gather 和 CUDA
Graph kernel，最终仍是一块完整 linear，而不是字面意义上的三个独立链表节点。
GPU kernel 消费固定地址段描述符，不逐节点遍历 Python 链表。

## Selection 与 ring 契约

对逻辑长度 `S`、页大小 `b`、sink `s`、window `w` 和固定 near 长度
`n >= b`，near 页边界为 `max(s // b, (S-w-n) // b) * b`。

- 完整 near 页按 `near_pages` 上限参与 avg/minmax 竞争，并在每个 KV group 内
  对 query heads 求均值；
- 保护边界部分页只把未受保护的 near token 送入 L2；sink/near/window 之外的
  token 不进入候选缓冲；
- L2 在每个 query head 上归一化候选和保护 token，再按 KV group 平均概率，
  用 near 候选填充剩余 K2 预算，不为填满 K2 扩大 near 区间或页上限；
- 最终注意力只 gather 选中的原始宽度 K/V。没有全历史 score、布尔 mask 或
  dense fallback；
- tie 选择较早的绝对页/token；可选 L2 量化只影响打分，不是 KV 存储压缩。

compact far-off 与旧 gate 不是 bitwise parity 合约：旧 sink-boundary bypass
可能额外接纳非 sink far token，compact 会排除它们，tie/arithmetic ordering
也可能不同。动态标定本身不变。

evict 层固定容量为 `s+n+w+b-1` 个槽：固定 sink 前缀加 cyclic suffix。
绝对位置 `t >= s` 映射到 `s + (t-s) % (n+w+b-1)`。只读 live Near/SWA 位置，
包括 ring wrap 后；过期槽不是注意力 key。`get_seq_length()` 仍返回逻辑 `S`，
RoPE 继续使用绝对位置，不重编号、不重新旋转 KV。后续每步只原地写一个 K
和一个 V token，不复制旧 KV。

`FarSkipWorkspace` 和 CUDA Graph 均为 request-local。下一请求必须重新
prepare/finalize、使旧 graph 失效并重新 capture。EOS/dead row 冻结逻辑长度，
不能覆盖已冻结的有效 KV。

## 限制

Eviction 在请求内不可逆。gate 改回 far-on、geometry/cache mode 改变、
crop/rewind、beam、offload、cache sharing/serialization 和 speculative decoding
均不支持。compact eager cache 与静态 direct cache 是不同入口；不要把 eager
selector 的限制误套到已实现的静态 whole-step CUDA Graph decode，也不要假设
任意配置都能 capture——不支持的组合仍须 fail-closed。

短序列上固定 ring 预留可能大于完整 cache。驱逐降低的是稳态主 KV 和
finalize 后进程 allocated；prefill 峰值必须单独实测，allocator reserved 下降
不是必要条件。外部 KV 引用或 autograd graph 会阻止释放，因此运行时需要
inference 并拒绝可微 cached KV。

## 验证与测量

```bash
python -m pytest -q \
  tests/test_far_skip_compact.py tests/test_far_skip_optimized.py \
  tests/test_far_skip_hybrid_cache.py tests/test_hybrid_batch_cache.py \
  tests/test_qwen_static_mask.py
python exp/bench/test_e127_gates.py
python exp/bench/test_tli_batch_gates.py
```

B=1 GraphRunner direct/conversion 对拍：

```bash
python exp/bench/e152v4_graph_latency.py \
  -L 65536 --repeats 2 --warmup 0 --max-gen 128 \
  --method tli --graph-mode graph --prefill-mode direct \
  --tia_level1_topk 64 --tia_level2_topk 1024 \
  --tli_near_len 4096 --tli_near_pages 32 --tli_near_tokens off \
  --tli_subspace tail --tia_level2_cmp_ratio 4 \
  --tli_sink_tokens 4 --tli_swa_size 128 \
  --tli_enable_layer_skip false --tli_far_skip_backend compact \
  --tli_far_skip_cache evict --out /tmp/direct.json
```

将 `--prefill-mode` 改为 `conversion` 得到对拍臂；将 `--graph-mode` 改为
`eager` 得到同静态 cache 的 eager 参照。Hybrid row-local direct/conversion
采集器为 `exp/bench/e182_hybrid_direct_prefill.py`。

两臂都必须记录并分别报告：

1. 稳态主 KV：逐层 `numel * element_size`、linear/ring/staging bytes、skip 与
   non-skip 层数；
2. finalize/fill 后 `torch.cuda.memory_allocated()`；
3. prefill 全过程 `max_memory_allocated()`，并注明首请求与后续请求、绝对值和
   相对模型加载后的 delta。

不能用容量公式或 `memory_reserved()` 代替 allocated，也不能把 non-skip 层的
完整 linear 计入 far 驱逐节约。质量与真实模型吞吐必须由目标 GPU 上的
RULER/LongBench 和 e2e 对拍另行闭合；CPU 结果不能当作 GPU 性能证据。
