import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

spec = importlib.util.spec_from_file_location('graph_fa2', Path(__file__).parents[1]/'exp/bench/graph_fa2.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


def setup(device='cpu', kernel=None, capacity=257):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    from transformers import Qwen3Config
    config = Qwen3Config(hidden_size=512, num_attention_heads=4, num_key_value_heads=2, head_dim=128)
    config._attn_implementation = 'flash_attention_2'
    model = torch.nn.Module(); model.attention = Qwen3Attention(config,0); model.eval()
    mask = torch.zeros(1,1,1,capacity,device=device,dtype=torch.bfloat16)
    lengths = torch.tensor([capacity],device=device)
    adapter = mod.GraphFA2(model,mask,lengths,kernel=kernel)
    q = torch.randn(1,4,1,128,device=device,dtype=torch.bfloat16)
    k = torch.randn(1,2,capacity,128,device=device,dtype=q.dtype)
    return model, adapter, lengths, q, k, torch.randn_like(k)


def test_interface_restore_and_no_duplicate_kv_append():
    calls=[]
    def kernel(q,k,v,**kw):
        calls.append((q.shape,k.shape,v.shape,kw))
        return torch.zeros_like(q)
    model,a,lengths,q,k,v=setup(kernel=kernel)
    original=a.module.ALL_ATTENTION_FUNCTIONS
    with pytest.raises(RuntimeError,match='body'):
        with a.activate(lengths):
            out,_=a.attention(model.attention,q,k,v,a.mask,scaling=.125)
            assert out.shape==(1,1,4,128)
            assert a.module.ALL_ATTENTION_FUNCTIONS['flash_attention_2']==a.attention
            raise RuntimeError('body')
    assert a.module.ALL_ATTENTION_FUNCTIONS is original
    assert len(calls)==1
    kw=calls[0][-1]
    assert 'k' not in kw and 'v' not in kw and kw['num_splits']==0 and kw['causal']
    assert kw['cache_seqlens'] is a.lengths and a.lengths.dtype==torch.int32
    assert a.calls=={'eager':1,'capture':0}


def test_mask_and_length_metadata_guards():
    model,a,lengths,q,k,v=setup(kernel=lambda *args,**kwargs:None)
    with pytest.raises(ValueError,match='untrusted mask'):
        a.attention(model.attention,q,k,v,a.mask.clone())
    with pytest.raises(ValueError,match='int64'):
        with a.activate(lengths.int()):pass
    with pytest.raises(ValueError,match='unsupported'):
        a.attention(model.attention,q,k,v,a.mask,softcap=1.)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_native_graph_live_lengths_matches_dense_fa2():
    from flash_attn import flash_attn_func
    model,a,lengths,q,k,v=setup(device='cuda')
    def run():
        with a.activate(lengths):
            return a.attention(model.attention,q,k,v,a.mask,scaling=128**-.5)[0]
    run();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual=run()
    for valid in (1,2,127,257,17):
        lengths.fill_(valid);q.normal_();k.normal_();v.normal_()
        # A future/padded suffix must not enter the dense denominator.
        k[:,:,valid:].fill_(1000);v[:,:,valid:].fill_(-1000)
        graph.replay()
        ref=flash_attn_func(q.transpose(1,2),k[:,:,:valid].transpose(1,2),
                           v[:,:,:valid].transpose(1,2),causal=True,softmax_scale=128**-.5)
        torch.testing.assert_close(actual,ref,atol=5e-3,rtol=5e-3)
    assert a.calls['capture']==1


@pytest.mark.parametrize('length', [0, -1, 258, 2**40])
def test_invalid_length_rejected_before_cast_or_registry_change(length):
    model,a,_,q,k,v=setup(kernel=lambda *args,**kwargs:None)
    old=a.module.ALL_ATTENTION_FUNCTIONS
    a.lengths.fill_(123)
    with pytest.raises(RuntimeError,match='outside cache capacity'):
        with a.activate(torch.tensor([length])):
            raise AssertionError('invalid length reached attention')
    assert a.module.ALL_ATTENTION_FUNCTIONS is old
    assert a.lengths.item()==123
