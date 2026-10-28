import os

import torch
from einops import rearrange, einsum

class Metrics:

    def __init__(self, threshold: int = 2048) -> None:
        self.threshold = threshold
        self.select_tokens = []
        self.k_max = []
        self.k_min = []
        self.k_delta_max = []
        self.k_delta_mean = []

    def clear(self):
        self.select_tokens.clear()
        self.k_delta_max.clear()
        self.k_delta_mean.clear()

    def _add_select_result_lazy(self,
        q_ids: torch.Tensor,
        mask: torch.Tensor,
        block_size: int,
    ):
        """【E127 iter1】惰性路径：消除每层每 decode step 的 2 次 .item()
        host 同步（q_ids[i].item() + b_mask.sum().item()——36 层模型每 token
        72 次同步，是 decode 流水线的最大 CPU 停顿源之一）。

        数值契约（逐位一致，门禁 test_e127_metrics_lazy.py）：
          - 旧路径：cond = (q_ids[i].item()+1 >= threshold)；cond 真时记录
            Python float = int(b_mask[i].sum().item()) * block_size / H。
          - 惰性路径：GPU 端存 0-dim int64 张量 s = where(cond, sum, -1)；
            读取端 get_select_tokens 一次性逐条 .item()（读取时刻同步一次，
            生产路径只在生成结束后读一次），过滤 -1 哨兵后在 Python float64
            里做与旧路径完全相同的 s*bs/H 逐条算术——整数精确、除法同序，
            均值逐位等于旧路径（旧路径 sum(list)/len 的加法序也按记录序）。
        mask 的 H（q-head 数）每条记录可能不同（per_q_head 臂 [H,tk] vs
        共享 [Hkv,tk]），故 bs/H 随条目一起存。
        """
        mask = mask.squeeze(0)  # [tq, h, tk]
        for i in range(q_ids.shape[0]):
            cond = (q_ids[i] + 1) >= self.threshold          # GPU bool，无同步
            s = mask[i].sum()                                # GPU int64
            self.select_tokens.append((torch.where(cond, s, s.new_full((), -1)),
                                       block_size, mask[i].shape[0]))

    def add_select_result(self,
        q_ids: torch.Tensor,
        mask: torch.Tensor,
        block_size: int,
    ):
        # E127 iter1→inplace 合并（2026-10-12）：惰性路径（数值逐位一致）
        # 并为默认——env 门反转，TLI_E127_FAST=0 显式回退原路径（紧急
        # 逃生口，归档复现用）。
        if os.environ.get("TLI_E127_FAST", "1") != "0":
            return self._add_select_result_lazy(q_ids, mask, block_size)
        mask = mask.squeeze(0) # [tq, h, tk]
        for i in range(q_ids.shape[0]):
            num_tokens = q_ids[i].item() + 1
            if num_tokens >= self.threshold:
                b_mask = mask[i] # [h, tk]
                num_select_tokens = b_mask.sum().item() * block_size / b_mask.shape[0]
                self.select_tokens.append(num_select_tokens)

    def add_select_result_batch(self, lengths, mask, block_size):
        """One reduction for [B,1,H,S], with deferred host accounting."""
        counts = mask.sum(dim=(1, 2, 3))
        counts = torch.where(lengths >= self.threshold, counts, -1)
        self.select_tokens.append((counts, block_size, mask.shape[2]))

    def get_select_tokens(self):
        if len(self.select_tokens) > 0:
            # E127 iter1：惰性条目 = (tensor, bs, H) 三元组；旧条目 = float。
            # 两者按记录序做完全相同的 Python float64 算术 → 均值逐位一致。
            total = 0.0
            cnt = 0
            for e in self.select_tokens:
                if isinstance(e, tuple):
                    s, bs, h = e
                    if s.ndim:
                        for count in s.tolist():
                            if count >= 0:
                                total += count * bs / h
                                cnt += 1
                        continue
                    s = int(s.item())          # 读取端唯一同步点（逐条）
                    if s < 0:                  # -1 哨兵 = 记录时阈值不满足
                        continue
                    total += s * bs / h
                else:
                    total += e
                cnt += 1
            result = total / cnt if cnt > 0 else None
        else:
            result = None
        return result
    
    def add_k_delta(self, k: torch.Tensor, block_size: int):
        k = rearrange(k, 'b (t bs) h d -> b t bs h d', bs=block_size)
        k_max = k.amax(dim=2)
        k_min = k.amin(dim=2)
        # 【B6 修复（kimi3 清单 F10，2026-10-08）】块内极差应为减法：
        # 原 (k_max + k_min).abs() 是 min 与 max 绝对值之和（有符号下无
        # 几何意义），按块跨度语义应为 k_max - k_min（极差，与 L1 minmax
        # 上界分数所刻画的「块内散布」一致）。死代码防御修复：当前无
        # 生产者调用本方法，无历史数据口径需要迁移。
        k_delta = k_max - k_min
        k_delta_max  = k_delta.amax(dim=0).amax(dim=1)
        k_delta_mean = k_delta.mean(dim=0).mean(dim=1)
        self.k_delta_max.append(k_delta_max.cpu())
        self.k_delta_mean.append(k_delta_mean.cpu())

    def get_k_delta_max(self):
        if len(self.k_delta_max) > 0:
            k_delta_max = torch.stack(self.k_delta_max).amax(dim=0)
            return k_delta_max.amax()
        else:
            return None

    def get_k_delta_mean(self):
        if len(self.k_delta_mean) > 0:
            k_delta_mean = torch.stack(self.k_delta_mean).mean(dim=0)
            return k_delta_mean.mean()
        else:
            return None

GLOBAL_METRICS = Metrics()

def get_metrics():
    return GLOBAL_METRICS

