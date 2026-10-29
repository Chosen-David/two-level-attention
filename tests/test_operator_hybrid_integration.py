"""Native operator Graph replay vs independent-cache eager torch attention."""
import copy
from functools import partial
from unittest.mock import patch
import pytest
import torch
from test_far_skip_hybrid_cache import make_model, isolated_runtime
from test_hybrid_batch_cache import build, buffers
from sparse_attn.ops.cuda_graph_state import graph_state
from sparse_attn.ops.tli_static_cache import TLIStaticLayer


def row_buffers(cache,b):
    for layer in cache.layers:
        if hasattr(layer,'rows'):
            row=layer.rows[b]
            yield row.keys
            yield row.values
        else:
            yield layer.keys[b:b+1]
            yield layer.values[b:b+1]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires actual CUDA Graph')
@pytest.mark.parametrize('batch',[2,4,8])
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('mode',['serial','overlap'])
@pytest.mark.parametrize('backend',['simt','tc'])
def test_cuda_operator_mixed_batch(monkeypatch,batch,dtype,mode,backend):
    import sparse_attn.ops.partial_attention as ops
    import sparse_attn.ops.kv_row_update as kv_ops
    native_append=kv_ops.append_kv_row
    native_update=TLIStaticLayer.update
    native_partial,native_merge=ops.partial_attention,ops.merge_partials
    def reference_partial(*args,**kwargs):
        kwargs['backend']='torch'
        return native_partial(*args,**kwargs)
    monkeypatch.setenv('TLI_OPERATOR_MODE',mode)
    monkeypatch.setenv('TLI_OPERATOR_BACKEND',backend)
    model=make_model(device='cuda').to(dtype)
    reference_model=copy.deepcopy(model)
    for repeat in range(2):
        lengths=[17+(b*7+repeat*3)%20 for b in range(batch)]
        skips=[(b+repeat)%2==0 for b in range(batch)]
        actual_runner=build(model,lengths,skips)
        reference_runner=build(reference_model,lengths,skips)
        pointers=[x.data_ptr() for x in buffers(actual_runner)]
        states=[]
        for layer in actual_runner.cache.layers:
            for b in range(batch):
                if hasattr(layer,'rows'):
                    row=layer.rows[b]
                    states.append((b,row.keys,row.values,getattr(row,'layout',None)))
                else:
                    states.append((b,layer.keys[b:b+1],layer.values[b:b+1],None))
        writes={}
        def append(k,v,nk,nv,slot,active):
            # Strong refs retain capture-time tensor addresses; replay updates
            # these tensors. Do not clone their capture-time values here.
            writes[k.data_ptr()]=(nk,nv,slot,active)
            return native_append(k,v,nk,nv,slot,active)
        def update(layer,k,v,cache_kwargs=None):
            gs=graph_state()
            for b in range(batch):
                writes[layer.keys[b:b+1].data_ptr()]=(k[b:b+1],v[b:b+1],gs.pos_buf[b:b+1],gs.alive_buf[b:b+1])
            return native_update(layer,k,v,cache_kwargs)
        for step in range(38):
            alive=[not(step>4 and b%3==0) for b in range(batch)]
            before=[(k.clone(),v.clone()) for _,k,v,_ in states]
            tokens=[(b+step)%32 for b in range(batch)]
            lengths=[n+int(active) for n,active in zip(lengths,alive)]
            for runner in (actual_runner,reference_runner):runner.inject(tokens,lengths,alive)
            if step==0:
                with patch.object(kv_ops,'append_kv_row',append),patch.object(TLIStaticLayer,'update',update):
                    actual_runner.capture()
            actual=actual_runner.run().logits.clone()
            assert len(writes)==len(states)
            for (b,k,v,layout),(expected_k,expected_v) in zip(states,before):
                nk,nv,slot,active=writes[k.data_ptr()]
                logical=lengths[b]-1
                expected_slot=logical if layout is None or logical<layout.sink else layout.sink+(logical-layout.sink)%layout.tail_capacity
                assert slot.item()==expected_slot and active.item()==alive[b]
                if alive[b]:
                    expected_k[:,:,expected_slot:expected_slot+1].copy_(nk)
                    expected_v[:,:,expected_slot:expected_slot+1].copy_(nv)
                torch.testing.assert_close(k,expected_k,rtol=0,atol=0,equal_nan=True)
                torch.testing.assert_close(v,expected_v,rtol=0,atol=0,equal_nan=True)
            with patch.object(ops,'partial_attention',reference_partial),patch.object(
                    ops,'merge_partials',partial(native_merge,backend='torch')):
                expected=reference_runner.run().logits
            close=torch.isclose(actual,expected,rtol=.02,atol=.002 if dtype==torch.bfloat16 else .0003)
            if not close.all():
                print(dict(repeat=repeat,step=step,alive=alive,bad=[
                    (i,actual[tuple(i)].item(),expected[tuple(i)].item())
                    for i in (~close).nonzero()[:5].tolist()]),flush=True)
            torch.testing.assert_close(actual,expected,rtol=.02,atol=.002 if dtype==torch.bfloat16 else .0003,
                                       msg=f'repeat={repeat},step={step}')
        assert pointers==[x.data_ptr() for x in buffers(actual_runner)]
        # First-layer KV inputs are identical across numerical consumers.
        # Later layers depend on their own attention outputs; their writes were
        # verified exactly against captured incoming tensors above instead.
        for actual_row,expected_row in zip(actual_runner.cache.layers[0].rows,reference_runner.cache.layers[0].rows):
            torch.testing.assert_close(actual_row.keys,expected_row.keys,rtol=0,atol=0,equal_nan=True)
            torch.testing.assert_close(actual_row.values,expected_row.values,rtol=0,atol=0,equal_nan=True)
