import torch
from numbers import Integral
from typing import Tuple, Dict, Any
from .utils import prepare_pad_mask, pad_tensor
from ..metrics import get_metrics

class Indexer:

    def __init__(self, args) -> None:
        self.args = args
        self.metrics = get_metrics()

    def prepare_index(self,
        k: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
    ):
        pass

    def compute_score(self,
        q: torch.Tensor, 
        q_ids: torch.Tensor,
        index_dict: Dict[str, torch.Tensor],
        cu_seqlens_idx: torch.Tensor,
        softmax_scale: float,
    ):
        pass

    def compute_mask(self,
        q_ids: torch.Tensor,
        score_dict: Dict[str, torch.Tensor],
    ):
        pass

    def get_block_size(self):
        pass

    def prepare_mask(self,
        q: torch.Tensor,
        q_ids: torch.Tensor,
        k: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        softmax_scale: float = None,
        valid_length: int = None,
    ) -> torch.Tensor:
        # Fixed-capacity decode uses a host length for one nonempty request.
        # Reject malformed metadata before slicing K or mutating index state.
        if valid_length is not None:
            if isinstance(valid_length, bool) or not isinstance(valid_length, Integral):
                raise TypeError("valid_length must be an integer or None")
            if not 1 <= valid_length <= k.shape[1]:
                raise ValueError("valid_length must be within [1, K capacity]")
            valid_length = int(valid_length)
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        # 【阶段 2a 固定宽度】valid_length：真实序列长（固定全宽 KV 下
        # k.shape[1]=S_cap 容量而非有效长）。可变宽路径 None → 维持
        # 旧口径（valid_length==k.shape[1]）逐位不变。
        # 修复（主 AI 审计 10-21）：quest/tia/twi/base 的 prepare_index
        # 签名无 valid_length 形参——按 prepare_index 是否声明支持
        # （supports_valid_length 类属性）决定是否传 kwarg：不支持时不传
        # 保持旧调用形态（None 分支逐位不变）；TLIIndexer 支持时传
        # valid_length 供其覆盖 index_dict（固定全宽 KV 真实有效长）。
        k_full_w = k.shape[1]
        if valid_length is None or not getattr(self, "supports_valid_length", False):
            index_dict = self.prepare_index(k, cu_seqlens_k)
        else:
            if valid_length < k.shape[1]:
                # 【阶段 2a 固定宽度】k 有效区切片：固定全宽 KV（k.shape[1]=
                # S_cap 容量 > 有效长）下，prepare_index 的块统计/k_qat 若从
                # 全宽算会混入 pad 列 → score_coarse 漂移（G5 aavg 反例）；
                # 且 softmax 归约宽度（score_fine 宽 S_cap vs 变长宽 S_real）
                # 不同 → fp32 求和顺序差跨 topk 边界。切片到有效区后块统计
                # 与 fine 宽度均与变长路径逐位一致；mask 产出后下方 pad
                # False 回全宽（eager 消费端固定形状）。切片改局部 k/cu，
                # 调用方全宽 k 不变。
                k = k[:, :valid_length]
                cu_seqlens_k = torch.tensor(
                    [0, valid_length], dtype=cu_seqlens_k.dtype,
                    device=cu_seqlens_k.device)
            index_dict = self.prepare_index(k, cu_seqlens_k,
                                            valid_length=valid_length)
        score_dict = self.compute_score(q, q_ids, index_dict, softmax_scale)
        mask = self.compute_mask(q_ids, score_dict)
        # 【阶段 2a 固定宽度】valid_length 非 None（固定全宽 KV，k.shape[1]=
        # S_cap 容量 > 有效长）时统一清 pad 区：compute_mask 各分支的
        # sink 强制区（L2939 `m[..., :sink_tok]=True` 用全量 sink_tok，短序列
        # sink_tok>有效长）与「配额空余时 topk 选 p=0/-inf pad token」都会把
        # [S_real, S_width) 标 True，eager 会 gather 到 pad 垃圾。此处统一
        # `mask[..., valid_length:] = False` 一次覆盖 moba/sinkswa/γ/legacy
        # 全部返回分支（075 精神：padding token 永不入选）。变长路径
        # valid_length=None 条件不触发，逐位零回归。
        if valid_length is not None and valid_length < mask.shape[-1]:
            mask[..., valid_length:] = False
        elif mask.shape[-1] < k_full_w:
            # 【阶段 2a 固定宽度】k 有效区切片后 mask 宽 = valid_length < 原始
            # 全宽（eager 消费端要求 [B,1,Hkv,S_cap] 固定形状）——pad False
            # 回全宽（有效区 [0, valid_length) 已由上方/内部与变长逐位一致，
            # pad 区恒 False 防泄漏）。
            mask = torch.cat([mask, mask.new_zeros(
                (mask.shape[0], mask.shape[1], mask.shape[2],
                 k_full_w - mask.shape[-1]), dtype=torch.bool)], dim=-1)
        block_size = self.get_block_size()
        self.metrics.add_select_result(q_ids, mask, block_size)
        return mask, block_size

    def clear(self):
        pass

