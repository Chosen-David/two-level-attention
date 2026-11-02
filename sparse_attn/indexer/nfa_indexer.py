"""GraphReady NFA 的配置、子空间索引和当前 Q 打分 kernel。

旧 DynamicCache/StaticCache 的 prepare/compute/mask 管线已删除；生产选择由
``nfa_layer_batch_selector.select_topk_for_layer_batched`` 在层内执行。
"""
import operator

import torch

from .tia_indexer import TIAIndexer

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover - CPU-only 环境
    triton = None
    tl = None
    _HAS_TRITON = False


class NFAIndexer(TIAIndexer):
    """单层 Req-Batched selector 使用的不可变配置与维度映射。"""

    _TILE_FINE = {32: (16, 1), 64: (64, 4), 128: (64, 4)}
    _TILE_COARSE = {32: (128, 8), 64: (64, 4), 128: (64, 4)}

    def __init__(self, args, snapshot=None) -> None:
        super().__init__(args)
        self.layer_idx = None
        self.subspace = getattr(args, 'nfa_subspace', 'full')
        if self.subspace not in (
                'full', 'rope', 'nope', 'tail', 'random', 'highfreq', 'mla_rope'):
            raise ValueError(f'nfa_subspace 非法取值: {self.subspace}')
        self.enable_subspace = bool(
            getattr(args, 'nfa_enable_subspace', True)) and self.subspace != 'full'
        if self.subspace in ('rope', 'nope', 'mla_rope'):
            self.enable_subspace = True
        self.l2_subspace = getattr(args, 'nfa_l2_subspace', 'same')
        if self.l2_subspace not in ('same', 'full'):
            raise ValueError(f'nfa_l2_subspace 非法取值: {self.l2_subspace}')
        self.head_dim = 192 if self.subspace == 'mla_rope' else 128

        self.far_method = getattr(args, 'nfa_far_method', 'minmax')
        self.near_method = getattr(args, 'nfa_near_method', 'avg')
        self.sink_tokens = int(getattr(args, 'nfa_sink_tokens', 128))
        if self.sink_tokens < 0:
            raise ValueError('nfa_sink_tokens 必须 >= 0')
        if self.sliding_window_size < 1:
            raise ValueError('nfa_swa_size 必须 >= 1')
        block_size = int(args.tia_block_size)
        if self.sliding_window_size % block_size:
            raise ValueError('nfa_swa_size 必须是 tia_block_size 的整数倍')

        def _quota(name, value, *, allow_off=False):
            if allow_off and isinstance(value, str) and value.strip().lower() == 'off':
                return None
            if value is None:
                return None
            if isinstance(value, bool):
                raise ValueError(f'{name} 须为整数计数，收到 {value!r}')
            try:
                return int(value) if isinstance(value, str) else operator.index(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'{name} 须为整数计数，收到 {value!r}') from exc

        near_tokens_raw = getattr(args, 'nfa_near_tokens', None)
        self.nt_near_pages = _quota(
            'nfa_near_pages', getattr(args, 'nfa_near_pages', None))
        self.nt_near_tokens = _quota(
            'nfa_near_tokens', near_tokens_raw, allow_off=True)
        self.nt_near_len = _quota(
            'nfa_near_len', getattr(args, 'nfa_near_len', None))
        near_tokens_off = (
            isinstance(near_tokens_raw, str)
            and near_tokens_raw.strip().lower() == 'off')
        if self.nt_near_pages is None or self.nt_near_len is None:
            raise ValueError('生产 NFA 必须显式传 nfa_near_pages 与 nfa_near_len')
        if self.nt_near_tokens is None and not near_tokens_off:
            raise ValueError("nfa_near_tokens 须传非负整数或 'off'")
        if self.nt_near_pages < 0 or self.nt_near_len < 0:
            raise ValueError('nfa_near_pages/nfa_near_len 必须非负')
        if self.nt_near_pages > args.tia_level1_topk:
            raise ValueError('nfa_near_pages 不可超过 tia_level1_topk')
        if self.nt_near_pages * block_size > self.nt_near_len:
            raise ValueError('网格约束破坏：nfa_near_pages * block_size <= nfa_near_len')
        k2 = int(args.tia_level2_topk)
        if k2 < self.sink_tokens + self.sliding_window_size:
            raise ValueError('tia_level2_topk 不可小于 sink+swa')
        if not near_tokens_off:
            if self.nt_near_tokens < 0:
                raise ValueError('nfa_near_tokens 必须非负')
            if self.nt_near_tokens > self.nt_near_pages * block_size:
                raise ValueError('网格约束破坏：nt <= nb * bs')
            if k2 - self.sink_tokens - self.sliding_window_size - self.nt_near_tokens < 0:
                raise ValueError('网格约束破坏：far = K2-sink-swa-nt >= 0')

        # 这些旧实验只有 HF Cache 管线实现；当前 Q selector 不接受静默降级。
        if getattr(args, 'nfa_proj_basis', None):
            raise ValueError('nfa_proj_basis 旧 HF Cache 路径已删除')
        if getattr(args, 'nfa_static_pair', False):
            raise ValueError('nfa_static_pair 旧 prefill 标定路径已删除')
        if getattr(args, 'nfa_far_skip_backend', 'legacy') != 'legacy':
            raise ValueError('仅支持 GraphReadyPagedRegionCache')
        if getattr(args, 'nfa_far_skip_cache', 'retain') != 'retain':
            raise ValueError('仅支持 GraphReadyPagedRegionCache')

    @property
    def _sink_pool_lo_blk(self):
        return max(0, self.sink_tokens // self.args.tia_block_size)

    def _tail_indices(self, device):
        if self.head_dim == 256:
            # GLM MLA：[NoPE192 | RoPE64]，RoPE 两个半区各取低频 16 维。
            return torch.cat((torch.arange(208, 224, device=device),
                              torch.arange(240, 256, device=device)))
        half = self.head_dim // 2
        delta = half // self.cmp_ratio
        return torch.cat((
            torch.arange(half - delta, half, device=device),
            torch.arange(self.head_dim - delta, self.head_dim, device=device)))

    def _subspace_indices(self, device):
        cache_key = (self.head_dim, device)
        cached = getattr(self, '_idx_sub_cache', None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        if self.subspace == 'rope':
            out = torch.arange(0, 64, device=device)
        elif self.subspace == 'nope':
            out = (torch.arange(0, 64, device=device) if self.head_dim == 256
                   else torch.arange(64, 128, device=device))
        elif self.subspace == 'mla_rope':
            out = torch.arange(128, 192, device=device)
        elif self.subspace == 'random':
            generator = torch.Generator().manual_seed(20261002)
            out = torch.randperm(128, generator=generator)[:32].sort().values.to(device)
        elif self.subspace == 'highfreq':
            delta = 64 // self.cmp_ratio
            out = torch.tensor(
                list(range(delta)) + list(range(64, 64 + delta)), device=device)
        else:
            out = self._tail_indices(device)
        self._idx_sub_cache = (cache_key, out)
        return out

    def _qat_indices(self, device):
        key = (self.head_dim, self.cmp_ratio, device)
        cached = getattr(self, '_qat_idx_cache', None)
        if cached is not None and cached[0] == key:
            return cached[1]
        indices = self._tail_indices(device)
        self._qat_idx_cache = (key, indices)
        return indices


if _HAS_TRITON:
    @triton.jit
    def _score_fine_kernel(
        KQ, Q, OUT, MOUT,
        T, stride_kt, stride_kh, stride_qh, stride_oh, stride_ot,
        stride_mh, stride_mt,
        Hkv: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
        HAS_MEAN: tl.constexpr,
        BATCHED: tl.constexpr = False, STRIDE_KB: tl.constexpr = 0,
    ):
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        kv = pid_h
        if BATCHED:
            pid_t = pid_t.to(tl.int64)
            pid_h = pid_h.to(tl.int64)
            batch = pid_h // Hkv
            kv = pid_h % Hkv
            KQ += batch * STRIDE_KB
        offs_t = pid_t * BT + tl.arange(0, BT)
        offs_d = tl.arange(0, D)
        mask_t = offs_t < T
        if HAS_MEAN:
            macc = tl.zeros((BT,), dtype=tl.float32)
        k = tl.load(
            KQ + offs_t[:, None] * stride_kt + kv * stride_kh + offs_d[None, :],
            mask=mask_t[:, None], other=0.0).to(tl.float32)
        for j in tl.static_range(G):
            h = pid_h * G + j
            q = tl.load(Q + h * stride_qh + offs_d)
            s = tl.sum(k * q[None, :], axis=1)
            s_o = s.to(OUT.dtype.element_ty)
            tl.store(OUT + h * stride_oh + offs_t * stride_ot, s_o, mask=mask_t)
            if HAS_MEAN:
                macc += s_o.to(tl.float32)
        if HAS_MEAN:
            tl.store(MOUT + pid_h * stride_mh + offs_t * stride_mt,
                     (macc / G).to(MOUT.dtype.element_ty), mask=mask_t)

    @triton.jit
    def _score_coarse_kernel(
        KMIN, KMAX, Q, OUT, MOUT,
        T, stride_kt, stride_kh, stride_kt2, stride_kh2,
        stride_qh, stride_oh, stride_ot, stride_mh, stride_mt,
        Hkv: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
        HAS_MEAN: tl.constexpr,
        BATCHED: tl.constexpr = False,
        STRIDE_KB: tl.constexpr = 0, STRIDE_KB2: tl.constexpr = 0,
    ):
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        kv = pid_h
        if BATCHED:
            pid_t = pid_t.to(tl.int64)
            pid_h = pid_h.to(tl.int64)
            batch = pid_h // Hkv
            kv = pid_h % Hkv
            KMIN += batch * STRIDE_KB
            KMAX += batch * STRIDE_KB2
        offs_t = pid_t * BT + tl.arange(0, BT)
        offs_d = tl.arange(0, D)
        mask_t = offs_t < T
        if HAS_MEAN:
            macc = tl.zeros((BT,), dtype=tl.float32)
        kmin = tl.load(
            KMIN + offs_t[:, None] * stride_kt + kv * stride_kh + offs_d[None, :],
            mask=mask_t[:, None], other=0.0).to(tl.float32)
        kmax = tl.load(
            KMAX + offs_t[:, None] * stride_kt2 + kv * stride_kh2 + offs_d[None, :],
            mask=mask_t[:, None], other=0.0).to(tl.float32)
        for j in tl.static_range(G):
            h = pid_h * G + j
            q = tl.load(Q + h * stride_qh + offs_d)
            s = (tl.sum(kmin * tl.minimum(q, 0.0)[None, :], axis=1)
                 + tl.sum(kmax * tl.maximum(q, 0.0)[None, :], axis=1))
            s_o = s.to(OUT.dtype.element_ty)
            tl.store(OUT + h * stride_oh + offs_t * stride_ot, s_o, mask=mask_t)
            if HAS_MEAN:
                macc += s_o.to(tl.float32)
        if HAS_MEAN:
            tl.store(MOUT + pid_h * stride_mh + offs_t * stride_mt,
                     (macc / G).to(MOUT.dtype.element_ty), mask=mask_t)
else:
    _score_fine_kernel = None
    _score_coarse_kernel = None
