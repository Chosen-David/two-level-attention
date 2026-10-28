"""Physical storage and bounded-write contracts; CPU checks are not GPU memory."""
import weakref
from types import SimpleNamespace
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.cache_utils import DynamicCache
from sparse_attn.ops.far_skip import FarSkipConfig
from sparse_attn.ops.far_skip_cache import FarSkipLayer
from sparse_attn.ops.far_skip_static import FarSkipStaticLayer
from sparse_attn.ops.tli_static_cache import TLIStaticCache, fill_static_from_dynamic


class WriteCounter(TorchDispatchMode):
    def __init__(self):self.elements=[]
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        if func==torch.ops.aten.copy_.default:self.elements.append(args[0].numel())
        if func==torch.ops.aten.index_copy_.default:self.elements.append(args[3].numel())
        return func(*args,**(kwargs or {}))


@pytest.mark.parametrize('length',[8192,32768])
@torch.inference_mode()
def test_one_time_compaction_and_constant_append(length,monkeypatch):
    monkeypatch.setenv('TLI_FAR_SKIP_KERNELS','torch')
    cfg=FarSkipConfig(64,4096,32,1024,4,128)
    cache=DynamicCache()
    cache.update(torch.randn(1,2,length,128),torch.randn(1,2,length,128),0)
    old=[weakref.ref(cache.layers[0].keys),weakref.ref(cache.layers[0].values)]
    cache.layers[0]=FarSkipLayer.from_layer(cache.layers[0],cfg)
    layer=cache.layers[0]
    assert all(ref() is None for ref in old)
    assert layer.layout.capacity==4291
    assert layer.keys.untyped_storage().nbytes()==4291*2*128*4
    pointers=layer.keys.data_ptr(),layer.values.data_ptr()
    # Multiple wraps, without constructing a full reference cache each step.
    incoming=torch.ones(1,2,1,128)
    counter=WriteCounter()
    with counter:
        for _ in range(2*layer.layout.tail_capacity+3):
            layer.update(incoming,incoming)
    assert set(counter.elements)=={2*128}
    assert len(counter.elements)==2*(2*layer.layout.tail_capacity+3)
    assert pointers==(layer.keys.data_ptr(),layer.values.data_ptr())


@torch.inference_mode()
def test_frozen_gate_per_prompt_and_released_prefill():
    from test_tli_batch_parallel import make_indexer
    cfg=SimpleNamespace(num_hidden_layers=2)
    modules=[]
    for li in range(2):
        idx=make_indexer(tli_far_skip_backend='compact',tli_far_skip_cache='evict',
                         tli_dyn_dgate=True)
        idx.layer_idx=li
        modules.append(SimpleNamespace(indexer=idx,layer_idx=li,sliding_window=None))
    model=SimpleNamespace(training=False,modules=lambda:iter(modules))
    static=TLIStaticCache(cfg,512,1)
    for skip_layer in (0,1):
        dyn=DynamicCache()
        for li,m in enumerate(modules):
            m.indexer.clear()
            m.indexer._dyn_skip_pending=li==skip_layer
            dyn.update(torch.randn(1,2,400,128),torch.randn(1,2,400,128),li)
        old=[weakref.ref(t) for layer in dyn.layers for t in (layer.keys,layer.values)]
        fill_static_from_dynamic(static,dyn,[400],model=model)
        del dyn
        assert all(ref() is None for ref in old)
        for li,m in enumerate(modules):
            assert m.indexer._dyn_frozen
            assert m.indexer.skip_far==(li==skip_layer)
            assert isinstance(static.layers[li],FarSkipStaticLayer)==(li==skip_layer)
            if li==skip_layer:assert static.layers[li].keys.shape[-2]<400


@pytest.mark.parametrize('device',['cpu','cuda'])
@torch.inference_mode()
def test_qwen_mixed_ring_and_parallel_full_kv(monkeypatch,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('requires actual CUDA')
    from test_far_skip_compact import make_model
    from sparse_attn.baseline_runtime import BatchCtx,batch_decode_scope
    from sparse_attn.ops.cuda_graph_state import disable_graph
    monkeypatch.setenv('TLI_BATCH_INDEXER','parallel')
    monkeypatch.setenv('TLI_BATCH_SCORE','triton' if device=='cuda' else 'torch')
    monkeypatch.setenv('TLI_BATCH_FUSIONS','1' if device=='cuda' else '0')
    monkeypatch.setenv('TLI_BATCH_GPU_PIPELINE','1' if device=='cuda' else '0')
    monkeypatch.setenv('TLI_FAR_SKIP_KERNELS','triton' if device=='cuda' else 'torch')
    disable_graph()
    model=make_model('evict',dynamic=True).to(device)
    if device=='cuda':model=model.bfloat16()
    dyn=DynamicCache()
    model(input_ids=torch.arange(79,device=device)[None]%32,past_key_values=dyn)
    modules=[layer.self_attn for layer in model.model.layers]
    modules[0].indexer._dyn_skip_pending=True
    modules[1].indexer._dyn_skip_pending=False
    static=TLIStaticCache(model.config,256,1)
    fill_static_from_dynamic(static,dyn,[79],model=model)
    del dyn
    mask=torch.full((1,1,1,256),-torch.inf,device=device,dtype=next(model.parameters()).dtype)
    static._far_skip_attention_mask=mask
    ctx=BatchCtx(1,[80],[1],[True],0)
    with batch_decode_scope(ctx):
        for length in range(80,84):
            ctx.seq_lens[0]=length
            mask.fill_(-torch.inf);mask[...,:length]=0
            pos=torch.tensor([length-1],device=device)
            output=model(input_ids=torch.tensor([[3]],device=device),position_ids=pos[None],
                         cache_position=pos,attention_mask=mask,past_key_values=static)
            assert torch.isfinite(output.logits).all()
    assert isinstance(static.layers[0],FarSkipStaticLayer)
    assert getattr(modules[0].indexer,'_batch_tli',None) is None
    assert modules[1].indexer._batch_tli.skip_far is False
