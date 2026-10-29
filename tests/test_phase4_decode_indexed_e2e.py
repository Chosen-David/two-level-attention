# -*- coding: utf-8 -*-
"""Phase 4: decode_indexe2e 多 run CPU 正确性（torch backend）。

mock indexer 模拟 compute_selection_graph 输出，验证 decode_indexed（含
protected + selected + merge）在变长区间 S_cap∈{16K,24K,32K} 下多 run
无错误且输出形状正确。CPU torch 路径（非 SIMT Triton）覆盖；GPU simt
路径留 GPU smoke test。

链接：exp/trace/results/e182_phase4_serial_smoke_verdict.json
"""
import pytest
import torch

from sparse_attn.ops.token_selection import graph_middle_selection
from sparse_attn.ops.partial_attention import partial_attention, merge_partials


class MockIndexer:
    """dummy TLI indexer，compute_selection_graph 语义近似生产。"""

    def __init__(self, K2=1024, sink=4, swa=128, group_size=4):
        self.sink_tokens = sink
        self.sliding_window_size = swa
        self.K2 = K2
        self.group_size = group_size

    def prepare_index(self, k, cu):
        B, S, H, D = k.shape
        x = k[..., :32].contiguous()
        p = x.reshape(B, S // 64, 64, H, 32)
        return dict(k_min=p.amin(2), k_max=p.amax(2), k_avg=p.mean(2), k_qat=x)

    def compute_selection_graph(self, q, idx_dict, scale, vl):
        B, _, Hq, D = q.shape
        Hkv = Hq // self.group_size
        S_real = vl[0].item()
        top = torch.arange(self.K2, device=q.device).unsqueeze(0).expand(Hkv, -1).contiguous()
        k2_mid = torch.full((Hkv,), self.K2, dtype=torch.long, device=q.device)
        mid_mask = torch.zeros(S_real, dtype=torch.bool, device=q.device)
        mid_mask[self.sink_tokens:S_real - self.sliding_window_size] = True
        return graph_middle_selection(top, k2_mid, mid_mask, S_real, S_real,
                                      self.sink_tokens, self.sliding_window_size)


def _run_decode_indexed(indexer, S_cap, Hq=32, D=128, dtype=torch.float32,
                        _q=None, _k=None, _v=None):
    """等价于 sparse_attn.ops.operator_pipeline.decode_indexed 的 torch backend 版。

    可选传入 _q/_k/_v 用于确定性测试；未传时内部随机生成。
    """
    B, Hkv, Dv = 1, Hq // indexer.group_size, D
    if _q is None:
        q = torch.randn(B, 1, Hq, D, dtype=dtype)
    else:
        q = _q
    if _k is None:
        k = torch.randn(B, S_cap, Hkv, D, dtype=dtype)
    else:
        k = _k
    if _v is None:
        v = torch.randn(B, S_cap, Hkv, Dv, dtype=dtype)
    else:
        v = _v
    cu = torch.tensor([0, S_cap], dtype=torch.int32)
    vl = torch.tensor([S_cap], dtype=torch.int64)
    scale = 1.0 / (D ** 0.5)

    index_dict = indexer.prepare_index(k, cu)
    selection = indexer.compute_selection_graph(q, index_dict, scale, vl)
    ids_counts = (selection.indices, selection.counts)

    protected = partial_attention(q, k, v, ids_counts, vl,
                                 sink=indexer.sink_tokens,
                                 swa=indexer.sliding_window_size,
                                 scale=scale, protected=True, splits=4,
                                 backend='torch')
    selected = partial_attention(q, k, v, ids_counts, vl,
                                sink=indexer.sink_tokens,
                                swa=indexer.sliding_window_size,
                                scale=scale, protected=False, splits=4,
                                backend='torch')
    return merge_partials(protected, selected, dtype=dtype, backend='torch')


@pytest.mark.parametrize("S_cap", [16384, 24576, 32768])
def test_decode_indexed_multi_run(S_cap):
    """decode_indexed 在目标 S_cap 区间下多 run 不报误且形状正确。"""
    idx = MockIndexer(K2=1024)
    for run in range(3):
        out = _run_decode_indexed(idx, S_cap)
        assert out.shape == (1, 1, 32, 128), f'S_cap={S_cap} run={run}: shape={out.shape}'
        assert torch.isfinite(out).all(), f'S_cap={S_cap} run={run}: non-finite output'


def test_decode_indexed_alignment_full_selection():
    """全选（sink+swa+mid = all positions）下 decode_indexed 与 dense attention bitexact（rtol=1e-3）。"""
    torch.manual_seed(77)
    B, Hq, Hkv, D, Dv = 1, 8, 4, 32, 32
    S_cap = 64
    sink, swa = 2, 8
    q = torch.randn(B, 1, Hq, D)
    k = torch.randn(B, S_cap, Hkv, D)
    v = torch.randn(B, S_cap, Hkv, Dv)
    ql = torch.tensor([S_cap], dtype=torch.int64)
    scale = 1.0 / (D ** 0.5)

    idx = MockIndexer(K2=S_cap, sink=sink, swa=swa, group_size=Hq // Hkv)
    out = _run_decode_indexed(idx, S_cap, Hq=Hq, D=D, _q=q, _k=k, _v=v)

    # 参考：dense full attention
    qg = (q[:, 0] * scale).float().reshape(B, Hkv, Hq // Hkv, D)
    scores = torch.einsum('bhgd,bthd->bhgt', qg, k.float()).reshape(B, Hq, -1)
    weights = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    vg = v.float().repeat_interleave(Hq // Hkv, dim=2)
    ref = torch.einsum('bht,bthd->bhd', weights, vg).unsqueeze(1)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
