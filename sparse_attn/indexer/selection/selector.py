"""生产 selector 的紧凑逻辑 token ID 与 canonical union。"""
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
