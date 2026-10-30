"""Independent FP32 selected attention oracle; selector reused, not revalidated."""
from types import SimpleNamespace
import pytest
import torch
from sparse_attn.ops.frozen_region_cache import FrozenRegionCache,RegionGeometry
from sparse_attn.ops.frozen_region_tli import decode,region_mask,tli_args
from sparse_attn.indexer.tli_indexer import TLIIndexer

@pytest.mark.parametrize('near,far',[(False,False),(False,True),(True,False),(True,True)])
@torch.inference_mode()
def test_glm_d256_consumer_against_selected_fp32(monkeypatch,near,far):
    monkeypatch.setenv('TLI_FINE_GATHER','0');monkeypatch.setenv('TLI_OPERATOR_MODE','off')
    monkeypatch.setattr('sparse_attn.indexer.tli_indexer._HAS_TRITON',False)
    torch.set_num_threads(2);torch.manual_seed(417)
    cache=FrozenRegionCache(1,RegionGeometry(max_new_tokens=80));cache.tli_regions=True
    k=torch.randn(1,2,4352,256);v=torch.randn_like(k)
    cache.update(k,v,0);cache.freeze({0:dict(skip_near=near,skip_far=far)})
    idx=TLIIndexer(tli_args(subspace='nope'));idx.layer_idx=0;idx.head_dim=256
    module=SimpleNamespace(layer_idx=0,indexer=idx,scaling=.0625)
    projection=torch.randn(512,19)/512**.5
    for step in range(66):
        q=torch.randn(1,2,1,256);kn=torch.randn_like(q);vn=torch.randn_like(q)
        actual=decode(module,q,kn,vn,cache,torch.tensor([4352+step]))
        if step not in (0,63,65):continue
        kk,vv,positions=cache.layers[0].gather()
        mask,_=region_mask(idx,q.transpose(1,2),kk.transpose(1,2),positions,4353+step,module.scaling)
        logits=(q.float()@kk.float().transpose(-1,-2))*module.scaling
        logits.masked_fill_(~mask.transpose(1,2), -torch.inf)
        expected=(logits.softmax(-1)@vv.float()).transpose(1,2)
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=1e-5)
        torch.testing.assert_close(actual.reshape(1,1,512)@projection,expected.reshape(1,1,512)@projection,atol=2e-6,rtol=1e-5)
