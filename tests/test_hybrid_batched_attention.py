from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from sparse_attn.ops.far_skip import FarSkipLayout
from sparse_attn.ops.hybrid_attention import attend_rows


@pytest.mark.parametrize('batch',[1,2,8,32])
def test_mixed_ring_linear_matches_independent_sdpa(batch):
    torch.manual_seed(44)
    q=torch.randn(batch,1,4,16)
    rows, selections, refs = [], [], []
    for b in range(batch):
        length=17 if b%2 else 65
        row=SimpleNamespace(keys=torch.randn(1,2,length,16),values=torch.randn(1,2,length,16))
        if b%2:
            row.layout=FarSkipLayout(2,15)
        ids=torch.randint(0,60 if b%2 else length,(2,7+b%3))
        valid=torch.rand(ids.shape)>.25
        heads=torch.arange(2)[:,None]
        safe=torch.where(valid,ids,0)
        if b%2:
            safe=row.layout.map(safe)
        k=row.keys[0][heads,safe].masked_fill(~valid[...,None],0)
        v=row.values[0][heads,safe].masked_fill(~valid[...,None],0)
        ref=torch.nn.functional.scaled_dot_product_attention(q[b,0].reshape(2,1,2,16),
            k[:,None],v[:,None],attn_mask=valid[:,None,None],scale=.25)
        refs.append(ref.reshape(1,1,4,16))
        rows.append(row);selections.append((ids,valid))
    real=torch.nn.functional.scaled_dot_product_attention
    with patch('sparse_attn.ops.hybrid_attention.F.scaled_dot_product_attention',wraps=real) as spy:
        actual=attend_rows(q,rows,selections,.25)
    assert spy.call_count==1
    torch.testing.assert_close(actual,torch.cat(refs),rtol=2e-5,atol=2e-6)
