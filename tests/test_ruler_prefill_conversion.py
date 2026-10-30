"""Execute the real RULER chunk entry without importing its CLI program."""
from types import SimpleNamespace

import pytest
import torch

from sparse_attn.baseline_runtime import BatchCtx, batch_decode_scope
from sparse_attn.ops.tli_static_cache import TLIStaticLayer
from exp.bench.ruler_chunk_entry import load_ruler_chunk


@pytest.mark.parametrize('batch',[1,2,4,8])
def test_ruler_preserves_unpadded_inputs_logits_and_defers_static_allocation(monkeypatch,batch):
    lengths=[5+b for b in range(batch)];width=max(lengths)
    ids=torch.zeros(batch,width,dtype=torch.long);positions=ids.clone()
    for b,n in enumerate(lengths):
        ids[b,-n:]=torch.arange(n)+10*b+1
        positions[b,-n:]=torch.arange(n)
    sources=[];calls=[];allocations=[]
    class Model:
        config=SimpleNamespace(num_hidden_layers=2)
        def modules(self):return []
        def parameters(self):yield torch.zeros(1)
        def __call__(self,*,input_ids,position_ids,use_cache):
            b=len(calls);n=lengths[b]
            assert not allocations, 'entire static batch must not precede later prefills'
            assert use_cache and torch.equal(input_ids,ids[b:b+1,-n:])
            assert torch.equal(position_ids,positions[b:b+1,-n:])
            calls.append(b)
            logits=torch.zeros(1,n,32);logits[0,-1,b+1]=1
            layers=[SimpleNamespace(keys=input_ids.float().view(1,1,n,1)+li,
                                    values=-input_ids.float().view(1,1,n,1)-li) for li in range(2)]
            cache=SimpleNamespace(layers=layers);sources.append(cache)
            return SimpleNamespace(logits=logits,past_key_values=cache)
    original_init=TLIStaticLayer.lazy_initialization
    def allocate(self,key):
        assert len(calls)==batch
        allocations.append(self)
        original_init(self,key)
    monkeypatch.setattr(TLIStaticLayer,'lazy_initialization',allocate)
    original_to=torch.Tensor.to
    def cpu_to(self,*args,**kwargs):
        return self if args==('cuda',) and not kwargs else original_to(self,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'to',cpu_to)
    def decode(model,ctx,past,gen,lens,*args):
        assert gen==[[b+1] for b in range(batch)]
        assert lens==lengths and ctx.seq_lens==lengths
        for li,layer in enumerate(past.layers):
            assert layer.cumulative_length==max(lengths)
            for b,n in enumerate(lengths):
                expected=ids[b,-n:].float()+li
                assert torch.equal(layer.keys[b,0,:n,0],expected)
                assert torch.equal(layer.values[b,0,:n,0],-expected)
                assert torch.count_nonzero(layer.keys[b,0,n:])==0
        assert all(layer.keys is None and layer.values is None for row in sources for layer in row.layers)
        return gen
    chunk=load_ruler_chunk(16)
    chunk.__globals__['_decode_batch_graph']=decode
    with torch.no_grad():
        result=chunk(Model(),SimpleNamespace(pad_token_id=0,eos_token_id=31),
            [None]*batch,ids,positions,None,lengths,set(),False,fixed_cache=True)
    assert result==[[b+1] for b in range(batch)]
