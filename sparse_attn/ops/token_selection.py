# -*- coding: utf-8 -*-
"""Bounded token IDs for decode. IDs are logical positions, never ring slots.

紧凑 Selection 接口（Phase 1）：compute_mask_graph 在 graph 路径产生
全宽 [1,1,Hkv,S_cap] bool mask，但消费端（operator pipeline）真正需要
的是排序去重后的 token IDs + 有效计数。本模块提供：
  - TokenSelection NamedTuple：indices [B,Hkv,K] int32 + counts [B,Hkv] int32
  - graph_middle_selection()：从 topk 输出直建 compact selection
  - canonical_selection / middle_selection：通用 compact 构建工具
解码路径全用 logical position（0..S_real-1），不做 ring slot 换算。
"""
from typing import NamedTuple
import torch


class TokenSelection(NamedTuple):
    """紧凑 decode 选择：排序去重后的 token IDs + 每 head 有效数。

    indices: [B, Hkv, K] int32，前 counts[b,h] 位有效（已排序去重），
             垃圾槽填 0（不使用 S_cap-1 哨兵——compact 接口不怕越界读，
             消费者通过 counts 知道有效范围）。
    counts:  [B, Hkv] int32，有效 token 数（不含 sink/swa 保护集——
             保护集由调用方独立处理）。
    """
    indices: torch.Tensor  # [B,Hkv,K], sorted unique valid prefix, int32
    counts: torch.Tensor   # [B,Hkv], int32, device resident

    def valid(self):
        """[B,Hkv,K] bool 掩码：arange < counts[..., None]。"""
        return torch.arange(
            self.indices.shape[-1], device=self.indices.device
        ) < self.counts[..., None]

    def to_mask(self, capacity):
        """诊断/兼容适配器：从 compact selection 重建 [B,Hkv,capacity] bool mask。

        新消费者（decode_indexed / PartialState）直接读 indices/counts，
        不走此路径。保留用于：
          (a) 单元测试对拍（mask == to_mask 验证 compact 接口无损）
          (b) 降级 / 调试可视化
        垃圾槽填 capacity（越界列）→ 被 slice [:capacity] 自然截掉。
        """
        result = torch.zeros(
            (*self.indices.shape[:-1], capacity + 1),
            dtype=torch.bool, device=self.indices.device)
        ids = torch.where(self.valid(), self.indices.long(), capacity)
        result.scatter_(-1, ids, True)
        return result[..., :capacity]


def canonical_selection(ids, valid, capacity, width=None):
    """保序去重 + 宽度压缩（correctness-first，非最优实现）。

    语义：取 ids[valid & 0<=ids<capacity]，排序去重，保留前 width 个
    （width=None 时保留全部）。两遍 sort：
      第一遍 sort：把无效值（capacity 哨兵）排到末尾，同值相邻
      标记 unique：ordered != previous（去重）
      第二遍 sort：unique=True 的在前（垃圾位 capacity 哨兵兜底）
    调用者必须保证 width ≥ 实际唯一有效 ID 数（否则截断丢 token）。
    """
    if ids.ndim != 3 or valid.shape != ids.shape or capacity < 1:
        raise ValueError(
            f'selection requires [B,Hkv,K] IDs/valid and positive capacity, '
            f'got ids.shape={ids.shape} valid.shape={valid.shape} capacity={capacity}')
    width = ids.shape[-1] if width is None else min(width, ids.shape[-1])
    # 无效值 → capacity 哨兵（排序后汇集到末尾）
    ordered = torch.where(
        valid & (ids >= 0) & (ids < capacity), ids, capacity).sort(-1).values
    previous = torch.cat(
        (torch.full_like(ordered[..., :1], -1), ordered[..., :-1]), -1)
    unique = (ordered < capacity) & (ordered != previous)
    packed = torch.where(unique, ordered, capacity).sort(-1).values[..., :width]
    count = (packed < capacity).sum(-1, dtype=torch.int32)
    return TokenSelection(
        torch.where(packed < capacity, packed, 0).to(torch.int32).contiguous(),
        count.contiguous())


def middle_selection(ids, valid, lengths, sink, swa, capacity, width=None):
    """移除保护集（sink 头 + swa 尾），不改变 L2 打分域。

    保护集 [0, sink) ∪ [lengths-swa, lengths) 被排除（这些 token 由
    protected 路径处理，不参与 L2 mid 竞争）。中段 [sink, lengths-swa)
    进入 canonical_selection。
    """
    lengths = lengths.reshape(-1, 1, 1)
    lo = lengths.clamp(max=sink)
    hi = torch.maximum(lo, lengths - swa)
    keep = valid & (ids >= lo) & (ids < hi) & (ids < lengths)
    return canonical_selection(ids, keep, capacity, width)


def graph_middle_selection(top, k2_mid, mid_mask, length, capacity, sink, swa, *, unique_top=False):
    """compute_mask_graph 路径专用：从 topk 输出直建紧凑 selection。

    复刻旧链：topk → rank gate → mid_mask & length 过滤 → middle_selection。
    输入：
      top: [Hkv, K2_static] int64（topk token IDs）
      k2_mid: 标量或 [Hkv] int（共享或每 head 有效 topk 数）
      mid_mask: [S_cap] bool（mid 域掩码）
      length: 标量（S_real，有效序列长）
      capacity: S_cap（总容量）
      sink: sink_tok
      swa: swa_tok（sliding window 大小）
    输出：TokenSelection（indices [1,Hkv,K2_static] int32）。
    """
    if isinstance(length, (int, float)):
        length = torch.tensor(length, dtype=torch.int64, device=top.device)
    if unique_top and top.is_cuda:
        from .selection_pack import pack_unique_top
        packed = pack_unique_top(top, k2_mid, mid_mask, length, capacity, sink, swa)
        if packed is not None:
            return TokenSelection(*packed)
    ranks = torch.arange(top.shape[-1], device=top.device)
    in_range = (top >= 0) & (top < capacity)
    safe = top.clamp(0, capacity - 1)
    keep = (ranks[None] < (k2_mid.reshape(-1, 1) if torch.is_tensor(k2_mid) else k2_mid)) & in_range & mid_mask[safe] & (top < length)
    if unique_top:
        # torch.topk returns distinct positions; only mask/sort is needed.
        prefix = length.clamp(max=sink)
        start = torch.maximum(prefix, length - swa)
        keep = keep & (top >= prefix) & (top < start)
        packed = torch.where(keep, top, capacity).sort(-1).values
        valid = packed < capacity
        return TokenSelection(torch.where(valid, packed, 0)[None].to(torch.int32),
                              valid.sum(-1, dtype=torch.int32)[None])
    return middle_selection(top[None], keep[None], length.reshape(1),
                            sink, swa, capacity)
