"""【阶段 2a】TLI 固定容量 KV cache（有效在前 left-pad 布局，decode-only）。

背景：pred_ruler_batch 的 TLI batch decode（`_tl_decode_batched`）当前消费
DynamicCache 变长 KV（有效占尾、形状随步增长）——CUDA Graph 化需要
**固定形状**的 KV 输入（T=S_cap 固定、cu=[0,S_cap] 烘焙、mask 全宽）。

本模块提供固定容量缓冲：层 buffer[b, :, 0:L_b, :] 为行 b 的有效 KV，
[L_b, S_cap) 为未写 pad 区（保持零）。decode 每步新 token 写到
pos = ctx.seq_lens[b] - 1（pred_ruler_batch 在 forward 前已将 seq_lens
含本步 +1）。配合 indexer.prepare_mask(..., valid_length=L_b) 与
eager_decoding_attn（cu=[0, S_cap]、mask [1,1,Hkv,S_cap]）→ 固定形状
decode（CUDA Graph 前置）。

prefill 不直接在本 cache 上做：pred_ruler_batch 沿用 DynamicCache
（left-pad 输入整行进 cache、有效占尾），prefill 后由
fill_static_from_dynamic() 逐层把每行有效列搬进本 cache 的 [0, L_b)
（一次性，每 chunk 一次）——避免与 prefill 的 2D bool padding mask
（有效占尾语义）冲突。

仅支持 decode（q_len==1）；prefill（q_len>1）显式拒绝（fail-closed）。

【E175 W1】skip 层 ring 物理 KV 压缩：`--tli_skip_layer_ids` 静态层集
（E137 判决 eps=0.05 → Qwen3-8B {7,8,10,11,35}）内层 cache 容量收缩到
cap = sink + ring（champion 几何 4+4096+128 = 4228），物理布局 =
[sink(4) | ring(4224)]：
  - sink 槽位 [0,4) 永驻（位置 0-3，永不覆盖）；
  - ring 槽位 [4,4228) 轮转覆盖：位置 p → 槽位 4+((p-4) mod 4224)，
    即任意时刻 ring 恰驻留最近 4224 个 token。
数学等价主张（见 exp/bench/test_e175_ring_equivalence.py 门禁）：
skip 层 mask 永选集（sink + near + swa 强制区与 K2 选择域）⊆ 驻留集
[0,4)∪[L-4224,L)，ring 驱逐的 token 在全宽 skip_far 几何下本就
绝不可选 → attention 输出等价（驱逐无损）。indexer 侧
（TLIIndexer.prepare_mask ring 分支）以「虚拟位置序 gather」保持选择
链与全宽 skip_far 逐位同构（同 mask 集），物理序差仅剩 eager 归约顺序
的 bf16 ULP（对齐几何逐位一致，非对齐 ULP 口径，E127 _ulp_ok 先例）。
"""
from __future__ import annotations

import torch
from transformers.cache_utils import Cache

from ..baseline_runtime import batch_decode_ctx


def pos_to_slot(pos, sink: int, ring: int):
    """【E175 W1】位置 → ring 槽位映射（可复用工具，tensor/标量通用）。

    p < sink → 槽位 p（sink 永驻区，位置即槽位）；
    p >= sink → 槽位 sink + ((p - sink) mod ring)（ring 轮转覆盖）。

    实现必须纯张量分支（torch.where + 整除取模）——CUDA Graph capture
    期禁止 host Python 分支依赖 pos 值（capture 烘焙后 replay 读旧值）。
    p ∈ [sink, sink+ring) 时 (p-sink) mod ring = p-sink → 未满阶段
    恒等映射（槽位 == 位置），满后环绕。数学性质（门禁 cell A 穷举）：
    ① [0, cap) 双射（sink 段恒等 + ring 段模双射）；
    ② 任意 L，驻留位置集 [0,sink)∪[max(sink,L-ring), L) 与槽位集
       [0, min(L,cap)) 一一对应。
    """
    pos = torch.as_tensor(pos) if not torch.is_tensor(pos) else pos
    return torch.where(
        pos < sink, pos, sink + (pos - sink) % ring)


def e175_ring_geometry(args) -> tuple[int, int]:
    """E175 W1：skip 层 ring 几何单一事实源（cache 侧与 indexer 侧同源）。

    ring 大小 R = near_len + swa（champion 4+4096+128 中的 4224）——
    skip 层 mask 永选域 = sink[0,4) ∪ near[L-4224, L-128) ∪ swa[L-128, L)，
    R = near+swa 恰好覆盖 near∪swa 的 4224 token；sink 另列（永驻槽）。
    fail-closed：ring 压缩要求 E155 固定几何显式（nt_near_len 必须显式
    传入——legacy α 比例口径下 near 宽随 L 漂移，恒定容量 ring 无定义）。
    """
    sink = int(getattr(args, "tli_sink_tokens", 128))
    swa = int(getattr(args, "tli_swa_size", 128))
    nl = getattr(args, "tli_near_len", None)
    if nl is None:
        raise ValueError(
            "E175 ring 压缩要求 --tli_near_len 显式传入（E155 固定几何）；"
            "legacy α 比例口径下 near 区宽随 L 漂移，恒容量 ring 无定义"
            "（fail-closed，宁拒跑不静默错几何）")
    return sink, int(nl) + swa


def e175_layer_caps(args, max_cache_len: int, n_layers: int) -> list[int] | None:
    """E175 W1：per-layer 容量表构造（pred_ruler_batch TLIStaticCache 入口）。

    返回 None（层集空 = 不启用，生产不变）或 list：skip 层 =
    e175_ring_geometry 的 cap，非 skip 层 = max_cache_len（全宽）。
    层 id 越界 fail-closed（错层集会让 ring 驱逐落到非 skip 层——
    语义级错误，宁可拒跑）。"""
    raw = str(getattr(args, "tli_skip_layer_ids", "") or "").strip()
    if not raw:
        return None
    ids = {int(p) for p in raw.split(",")}
    for i in ids:
        if i < 0 or i >= n_layers:
            raise ValueError(
                f"tli_skip_layer_ids 层 id 越界：{i} ∉ [0, {n_layers})"
                "（模型层数）——错层集会让 ring 驱逐落到非 skip 层，"
                "fail-closed 拒跑")
    sink, ring = e175_ring_geometry(args)
    cap = sink + ring
    if cap >= max_cache_len:
        # 容量收缩不成立（cap >= 全宽）→ ring 无收益且绕过现有 pad 语义，
        # 显式拒绝（例如短序列 bench 配 S_cap <= 4228 时误传层集）。
        raise ValueError(
            f"E175 ring cap={cap} >= S_cap={max_cache_len}：ring 压缩在"
            "本容量配置下无收缩收益，拒绝构造（fail-closed）")
    return [cap if l in ids else max_cache_len for l in range(n_layers)]


class TLIStaticLayer:
    """固定容量 KV 层。buffer [B, Hkv, S_cap, D]，有效在前。

    【E175 W1】ring_size 非 None → ring 压缩层：S_cap = ring_sink +
    ring_size（如 4+4224），写入位置经 pos_to_slot 映射（scatter 目标
    列 = ring 槽位）。sink 段 [0, ring_sink) 永驻；ring 段轮转覆盖。
    非 ring 层（ring_size=None）行为逐位不变（铁律：默认 flags 零扰动）。

    对齐 StaticLayer 的 CacheLayerMixin 契约（is_compileable / max_batch_size
    / max_cache_len / get_seq_length / get_mask_sizes / get_max_cache_shape
    / lazy_initialization / update / reset / crop / reorder_cache）。
    """

    is_compileable = True
    is_sliding = False

    def __init__(self, max_cache_len: int, max_batch_size: int,
                 ring_size: int | None = None, ring_sink: int = 0):
        if max_cache_len < 1 or max_batch_size < 1:
            raise ValueError("TLIStaticLayer 需要 max_cache_len>=1 且 max_batch_size>=1")
        if ring_size is not None and (ring_size < 1 or ring_sink < 1
                                      or ring_sink + ring_size != max_cache_len):
            raise ValueError(
                f"ring 层契约：ring_sink({ring_sink}) + ring_size({ring_size})"
                f" 必须等于 max_cache_len({max_cache_len}) 且均 >= 1")
        self.max_cache_len = int(max_cache_len)
        self.max_batch_size = int(max_batch_size)
        # E175 W1：ring 几何（None = 全宽层，零行为变化）
        self.ring_size = None if ring_size is None else int(ring_size)
        self.ring_sink = int(ring_sink)
        self.keys = None
        self.values = None
        self.num_heads = None
        self.head_dim = None
        # 单调递增累计写长（供 get_seq_length；batch decode 下 cache_position
        # 不影响正确性——RoPE 用显式 position_ids、create_causal_mask 4D 早退、
        # TLI patch 全宽分支不切片）。
        self.cumulative_length = 0

    @property
    def is_ring(self) -> bool:
        return self.ring_size is not None

    def lazy_initialization(self, key_states: torch.Tensor):
        """按首层首个 decode 步（或搬移时）的 [B, Hkv, S, D] 形状预分配全宽 buffer。"""
        if key_states.ndim != 4:
            raise ValueError(f"TLIStaticLayer 期望 4D key_states，got {key_states.ndim}D")
        self.num_heads = key_states.shape[1]
        self.head_dim = key_states.shape[-1]
        self.dtype, self.device = key_states.dtype, key_states.device
        if key_states.shape[0] != self.max_batch_size:
            raise ValueError(
                f"TLIStaticLayer batch {key_states.shape[0]} != 构造 batch "
                f"{self.max_batch_size}")
        shape = (self.max_batch_size, self.num_heads, self.max_cache_len,
                 self.head_dim)
        self.keys = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.values = torch.zeros(shape, dtype=self.dtype, device=self.device)

    def update(self, key_states, value_states, cache_kwargs=None):
        """decode-only：新 token [B, Hkv, 1, D] → buffer 写入列。

        非 ring 层：写 pos[b] = seq_lens[b]-1 列（有效在前 left-pad 布局），
        行为逐位不变（铁律）。
        ring 层（E175 W1）：写 ring 槽位列 slot = pos_to_slot(pos)——sink
        槽 [0, ring_sink) 仅当 pos < ring_sink 时写入（即 sink 段永驻，
        位置 >= ring_sink 的 token 绝不覆盖 sink 槽）；ring 段轮转覆盖
        最老驻留 token。dead 行 seq_lens 冻结 → slot 不变 → 重复写同槽
        （被 mask 排除，无污染）。

        返回全宽 buffer [B, Hkv, S_cap, D]（固定形状）。
        """
        if key_states.ndim != 4 or key_states.shape[-2] != 1:
            raise RuntimeError(
                "TLIStaticCache 仅支持 decode（q_len==1）；prefill 须走 "
                "DynamicCache + fill_static_from_dynamic 搬移。当前 "
                f"key_states shape={tuple(key_states.shape)}")
        if self.keys is None:
            self.lazy_initialization(key_states)
        ctx = batch_decode_ctx()
        if ctx is None:
            raise RuntimeError(
                "TLIStaticCache.update 需要 batch decode ctx（batch_decode_ctx "
                "为 None）；本 cache 仅供 pred_ruler_batch batch 模式")
        if ctx.batch != self.max_batch_size:
            raise RuntimeError(
                f"TLIStaticCache ctx.batch={ctx.batch} != 构造 batch "
                f"{self.max_batch_size}")
        # 【阶段 2c CUDA Graph 前置】KV 写入位置张量化：scatter_ 单 kernel
        # 沿 dim2（S_cap 维）每行写 pos[b] 一列。pos 为 tensor 索引 →
        # replay 时 pos_buf 更新 → 写新列（CUDA Graph 兼容）。
        # dead 行 seq_lens 冻结（不递增）→ pos 不变 → 重复写同列（被
        # mask 排除，无污染）。非 graph 模式（pos_buf=None）回退 Python
        # 构造，语义与旧逐行 copy_ 逐位一致（纯内存拷贝，无算术）。
        from .cuda_graph_state import graph_state  # noqa: E402
        _gs = graph_state()
        if _gs.enabled and _gs.pos_buf is not None:
            pos = _gs.pos_buf
        else:
            pos = torch.tensor(
                [ctx.seq_lens[b] - 1 for b in range(self.max_batch_size)],
                dtype=torch.long, device=self.keys.device)
        if self.is_ring:
            # 【E175 W1】ring 槽位映射：纯张量 torch.where + 取模（graph
            # capture 安全——无 host 分支依赖 pos 值）。不做 clamp 上界：
            # 映射天然落 [0, cap)（任意大的绝对位置都被 mod 收进 ring）；
            # 保留 clamp 下界 0 防御（pos_buf 异常负值）。注意：非 ring 层
            # 的 clamp 上界语义不适用于 ring 层——clamp 会把环绕映射破坏成
            # 尾列重复写，这里必须区分（ring 分支先行映射再 clamp(0)）。
            pos = pos_to_slot(pos, self.ring_sink, self.ring_size)
            pos = torch.clamp(pos, min=0)
        else:
            pos = torch.clamp(pos, 0, self.max_cache_len - 1)
        # scatter_ 沿 dim2（S_cap 维）每行写 pos[b] 一列：index 与 src
        # 同 shape [B,Hkv,1,D]、元素值为 pos[b] → self[b,h,pos[b],d] =
        # src[b,h,0,d]（单 kernel，graph 兼容）。dead 行 pos 冻结 →
        # 重复写同列（被 mask 排除，无污染）。
        idx = pos.view(-1, 1, 1, 1).expand(
            self.max_batch_size, self.num_heads, 1, self.head_dim)
        self.keys.scatter_(2, idx, key_states)
        self.values.scatter_(2, idx, value_states)
        self.cumulative_length += 1
        return self.keys, self.values

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_mask_sizes(self, cache_position):
        return self.max_cache_len, 0

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def reset(self):
        if self.keys is not None:
            self.keys.zero_()
            self.values.zero_()
        self.cumulative_length = 0

    def crop(self, max_length):
        raise RuntimeError("TLIStaticLayer 不支持 crop（固定容量 decode-only）")

    def reorder_cache(self, beam_idx):
        raise RuntimeError("TLIStaticLayer 不支持 reorder（greedy decode-only）")


class TLIStaticCache(Cache):
    """decode 用固定容量 KV cache（每层一个 TLIStaticLayer 全宽缓冲）。

    【E175 W1】layer_caps：可选 per-layer 容量表（list[int] | None）。
    None = 全层 max_cache_len（现状，逐位不变）；list 则层 l 容量 =
    layer_caps[l]。容量 < max_cache_len 的层自动构造为 ring 层
    （cap = ring_sink + ring_size，布局 [sink|ring]，见 TLIStaticLayer）；
    ring_sink 必须显式传入且对所有 ring 层一致（几何由
    e175_layer_caps 保证 cap == sink+ring 恒等式）。"""

    def __init__(self, config, max_cache_len: int, batch_size: int,
                 layer_caps: list[int] | None = None,
                 ring_sink: int = 4):
        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            layer_types = ["full_attention" for _ in range(config.num_hidden_layers)]
        if layer_caps is None:
            layers = [TLIStaticLayer(max_cache_len, batch_size)
                      for _ in layer_types]
        else:
            if len(layer_caps) != len(layer_types):
                raise ValueError(
                    f"layer_caps 长 {len(layer_caps)} != 层数 "
                    f"{len(layer_types)}（fail-closed）")
            layers = []
            for cap in layer_caps:
                if cap == max_cache_len:
                    layers.append(TLIStaticLayer(max_cache_len, batch_size))
                elif cap < max_cache_len:
                    # E175 W1：容量收缩层 = ring 压缩层
                    layers.append(TLIStaticLayer(cap, batch_size,
                                                 ring_size=cap - ring_sink,
                                                 ring_sink=ring_sink))
                else:
                    raise ValueError(
                        f"layer_caps 含越层容量 {cap} > S_cap {max_cache_len}"
                        "（fail-closed：容量表只允许收缩）")
        super().__init__(layers=layers)
        self.batch_size = int(batch_size)
        # max_cache_len 由 Cache 基类只读 property 从 layers 推导
        # （max(layer.max_cache_len)），get_max_cache_shape 走
        # layers[layer_idx]——TF 4.56.2 基类契约如此，此处不自行赋值
        # （旧行 `self.max_cache_len = int(max_cache_len)` 会触发只读
        # property setter → AttributeError: no setter）。
        # E175 W1：ring 几何快照（patch/indexer 侧的 ring 层判定与
        # prepare_mask 虚拟轴共享——cap / sink / ring 单一事实源）。
        _rl = [l for l in layers if l.is_ring]
        self.ring_sink = _rl[0].ring_sink if _rl else None
        self.ring_size = _rl[0].ring_size if _rl else None
        self.ring_cap = (self.ring_sink + self.ring_size) if _rl else None


def fill_row_from_tail(layer: TLIStaticLayer, b: int,
                       k_row: torch.Tensor, v_row: torch.Tensor,
                       L: int):
    """单行搬移：k_row/v_row [Hkv, S_row, D] 有效占尾（尾 L 列有效，
    位置 p ↔ 列 S_row - L + p）→ layer buffer[b] 的有效区。

    非 ring 层：位置序直拷 [0, L)（现状语义逐位不变）。
    ring 层（E175 W1）：sink 槽 [0, ring_sink) ← 位置 [0, ring_sink)；
    ring 段驻留位置 [max(sink, L-ring), L) → pos_to_slot 环绕 scatter。
    L <= cap 时槽位 == 位置（未满恒等），环绕分支退化为顺序直拷。

    fill_static_from_dynamic（多行整 cache 搬移）与 pred_ruler_batch
    prefill 内联搬移共用本函数（单一事实源：两入口的 ring 语义不可
    漂移）。"""
    S_row = k_row.shape[-2]
    if L < 0 or L > S_row:
        raise RuntimeError(
            f"搬移越界：b={b} L={L} S_row={S_row}")
    if not layer.is_ring and L > layer.max_cache_len:
        # ring 层允许 L > cap（只搬驻留集）；非 ring 层 L 必须全容纳。
        raise RuntimeError(
            f"搬移越界：b={b} L={L} cap={layer.max_cache_len}")
    if not L:
        return
    if not layer.is_ring:
        layer.keys[b, :, :L].copy_(k_row[:, S_row - L:])
        layer.values[b, :, :L].copy_(v_row[:, S_row - L:])
        return
    s, r = layer.ring_sink, layer.ring_size
    # sink 段直拷（位置 0..s-1 == 槽位；有效长不足 sink 时截断）
    ns = min(s, L)
    layer.keys[b, :, :ns].copy_(k_row[:, S_row - L:S_row - L + ns])
    layer.values[b, :, :ns].copy_(v_row[:, S_row - L:S_row - L + ns])
    if L <= s:
        return
    # ring 段：驻留位置 [max(s, L-r), L) → 环绕槽位。位置升序下
    # pos_to_slot 是到 ring 槽位段的双射 → 高级索引赋值逐元素对应、
    # 无冲突（slots 与 sink 槽 [0,s) 不相交——ring 段值域 [s, cap)）。
    p_lo = max(s, L - r)
    src = torch.arange(p_lo, L, device=k_row.device)
    slots = pos_to_slot(src, s, r)
    # 【E175 门禁 cell C 修复】高级索引 getitem（layer.keys[b,:,slots]）
    # 返回的是临时张量，对它 .copy_ 只写临时不落盘（sink 段是基本切片
    # view，copy_ 有效）——ring 段必须 __setitem__ 真写 buffer。
    layer.keys[b][:, slots] = k_row[:, S_row - L + src]
    layer.values[b][:, slots] = v_row[:, S_row - L + src]


def fill_static_from_dynamic(cache: TLIStaticCache, dyn, seq_lens):
    """把 prefill 后的 DynamicCache（有效占尾）搬进 TLIStaticCache（有效在前）。

    dyn.layers[l].keys/values：[B, Hkv, S_max, D]，行 b 有效列 =
    [S_max-lens[b], S_max)（left-pad 输入整行进 cache）。搬到
    buffer[l][b, :, 0:lens[b])。逐层逐行 copy_（一次性，每 chunk 一次）。

    铁律一致性：搬移是纯内存布局重排（同一 KV 值、同一 token 顺序，
    列索引 0..L_b-1 对应 token 0..L_b-1 绝对位置）——下游
    prepare_mask(valid_length=L_b) 的有效区 [0, L_b) 与变长路径
    （直接消费 [0, L_b) 前缀）语义逐位一致。

    【E175 W1】ring 层经 fill_row_from_tail 环绕搬移（sink 头 + 尾部
    最近 ring_size token 按槽位写入）。

    【TF 4.56 适配】4.56 移除 DynamicCache.key_cache/value_cache 属性，
    改为 layers[l].keys/values（DynamicLayer 直接字段）；旧属性访问
    AttributeError（GPU 单测未覆盖 e2e 形态，4.56.2 跑批现形）。
    """
    n_layers = len(cache.layers)
    if len(dyn.layers) != n_layers:
        raise RuntimeError(
            f"DynamicCache 层数 {len(dyn.layers)} != TLIStaticCache 层数 "
            f"{n_layers}")
    for l in range(n_layers):
        k = dyn.layers[l].keys
        v = dyn.layers[l].values
        layer = cache.layers[l]
        if layer.keys is None:
            layer.lazy_initialization(k)
        for b in range(cache.batch_size):
            fill_row_from_tail(layer, b, k[b], v[b], int(seq_lens[b]))
