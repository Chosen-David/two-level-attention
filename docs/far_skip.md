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

### 三态存储计划与 finalize 所有权

prefill 前按 request-local gate 为每层选择唯一的物理写入计划：

| 模式 | 适用条件 | prefill 写入 | finalize |
|---|---|---|---|
| `RING_FINAL` | 已知静态 skip | 只写固定 sink + cyclic Near/SWA ring，不分配 full linear | `FarSkipStaticLayer.from_ring()` 直接接管，数据指针不变 |
| `LINEAR_FINAL` | 已知 non-skip 或非 compact | 只写完整 linear `[0,L)`，不创建 staging ring | 只提交元数据，数据指针不变 |
| `LINEAR_PROVISIONAL` | 动态或未知 gate | 只写一份完整 provisional linear，不双写 ring | non-skip 直接保留；skip 做唯一一次 linear→ring 压实后释放 linear |

动态 gate 必须消费 dense prefill 的原始完整 K/V 才能冻结，因此“动态 skip 从不
分配 full linear”和“动态 non-skip 零回拷”不能同时成立。这里选择单份
provisional linear，并显式披露动态 skip 的 finalize 压实；不恢复双写来制造表面
零拷贝。三种路径的 finalize/所有权移交都不再把 ring tail 回填到 linear。

Hybrid batch 的 compact 行仍直接转移各自独立的 ring/linear 与 indexer snapshot。
native-dense 行则在 `prepare_row()` 时把 B=1 proxy 预绑定到最终
`batch.keys[b:b+1]` / `values[b:b+1]`；prefill 直接写该视图，finalize 校验 shape、
stride、数据指针和 storage 后只丢弃 proxy，不再回拷整行。

因此，skip 层最终只拥有固定 ring，non-skip 层最终只拥有完整 linear。这里的
“零拷贝”指 finalize/所有权移交零 KV 回拷，不指取消原始 K/V 到最终存储的首次
持久化。GPU kernel 消费固定地址段描述符，不逐节点遍历 Python 链表。

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
  --tia_level1_topk 128 --tia_level2_topk 1024 \
  --tli_near_len 4096 --tli_near_pages 32 --tli_near_tokens off \
  --tli_subspace tail --tia_level2_cmp_ratio 4 \
  --tli_sink_tokens 4 --tli_swa_size 128 \
  --tli_enable_layer_skip false --tli_far_skip_backend compact \
  --tli_far_skip_cache evict --out /tmp/direct.json
```

将 `--prefill-mode` 改为 `conversion` 得到对拍臂；将 `--graph-mode` 改为
`eager` 得到同静态 cache 的 eager 参照。Hybrid row-local direct/conversion
三口径采集器为 `exp/bench/e183_hybrid_zero_copy.py`（E182 采集器与 K1=64
历史判决保留作基线）。

两臂都必须记录并分别报告：

1. 稳态主 KV ownership：按 `(device, untyped_storage().data_ptr())` 去重物理
   storage，区分 view bytes 与 physical owner bytes，并报告 linear/ring 和
   skip/non-skip 层数；
2. 临时 staging/finalize 拷贝：静态 skip linear ownership、non-skip staging ring、
   hybrid native 独立 B=1 staging 均应为 0；动态 skip 压实单独报告；
3. 过程显存：逐行 prefill、finalize 前后、capture 后与 steady decode 的
   `memory_allocated/max_memory_allocated`，注明首请求与后续请求、绝对值和相对
   模型加载后的 delta；reserved 与 `mem_get_info` 只作辅助；
4. prefill、finalize/build、capture 和 TPOT 分开计时，正式比较至少三次取中位数。

不能用容量公式或 `memory_reserved()` 代替 allocated，也不能把 non-skip 层的
完整 linear 计入 far 驱逐节约。质量与真实模型吞吐必须由目标 GPU 上的
RULER/LongBench 和 e2e 对拍另行闭合；CPU 结果不能当作 GPU 性能证据。

E183 的 K1=128 实测已覆盖 B=1 四种 gate × 三长度 × graph/eager，以及 hybrid
B=1/2/8/32 的 direct/conversion 对拍；生成 token、最终 KV ownership 一致，KV
地址在 finalize/capture/steady 间稳定。同机 E182 static-32K 对照中，staging
linear 减少 1,798,131,712 bytes，后续请求 graph prefill 峰值下降 1.675 GiB；
首请求、graph 池和 eager 后续请求口径分别保留。动态 skip 的单次压实及 B32
共享机计时波动不作隐藏，完整三口径数据见
`exp/trace/results/e183_zero_copy_verdict.json`。
