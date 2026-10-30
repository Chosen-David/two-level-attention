"""Independent host semantics for device-only decode input preparation."""
import pytest
import torch
from types import SimpleNamespace

from sparse_attn.ops.quality_decode_inputs import DeviceDecodeInputs


def test_warmup_streams_reuse_only_thread_device_without_request_owners(monkeypatch):
    import threading
    from sparse_attn.ops import quality_graph
    monkeypatch.setattr(quality_graph,'_warmup_streams',threading.local())
    created=[]
    def create(device):
        result=object();created.append(result);return result
    monkeypatch.setattr(torch.cuda,'Stream',create)
    monkeypatch.setenv('TLI_REUSE_WARMUP_STREAM','1')
    first,reused=quality_graph._request_warmup_stream(torch.device('cuda:0'))
    assert not reused
    assert quality_graph._request_warmup_stream(torch.device('cuda:0'))==(first,True)
    assert quality_graph._request_warmup_stream(torch.device('cuda:1'))[0] is not first
    found=[]
    thread=threading.Thread(target=lambda:found.append(quality_graph._request_warmup_stream(torch.device('cuda:0'))))
    thread.start();thread.join()
    assert found[0][0] is not first and not found[0][1]
    monkeypatch.setenv('TLI_REUSE_WARMUP_STREAM','0')
    assert quality_graph._request_warmup_stream(torch.device('cuda:0'))[0] is not first
    assert len(created)==4


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires reserved GPU0')
@pytest.mark.parametrize('batch,capacity', [(1,16384),(8,32785)])
@pytest.mark.parametrize('minimum', [1,2,4])
def test_ragged_eos_reset_and_device_inputs(batch,capacity,minimum):
    device='cuda'
    lens=[9000+3*b for b in range(batch)]
    initial_live=[b != 3 for b in range(batch)]
    tokens=torch.empty(batch,1,device=device,dtype=torch.long)
    positions=torch.empty_like(tokens)
    lengths=torch.empty(batch,device=device,dtype=torch.long)
    pos=torch.empty_like(lengths)
    alive=torch.empty(batch,device=device,dtype=torch.bool)
    mask=torch.empty(batch,1,1,capacity,device=device,dtype=torch.bfloat16)
    buffers=DeviceDecodeInputs(tokens,positions,pos,lengths,alive,mask,
        lens,lens,[1]*batch,initial_live,0,{2,7},minimum)
    # Both warmup and a fresh replay loop must reset exactly on step zero.
    for _ in range(2):
        host_len=list(lens);host_live=list(initial_live);generated=[1]*batch
        for step in range(6):
            previous=[2 if (step+b)%2==0 else 7 if b%3==0 else 4 for b in range(batch)]
            if step:
                host_live=[live and not (token in {2,7} and n>=minimum)
                           for live,token,n in zip(host_live,previous,generated)]
            host_len=[n+int(live) for n,live in zip(host_len,host_live)]
            generated=[n+int(live) for n,live in zip(generated,host_live)]
            buffers.inject(step,torch.tensor(previous,device=device))
            assert tokens.flatten().tolist()==[t if live else 0 for t,live in zip(previous,host_live)]
            assert positions.flatten().tolist()==[n+step for n in lens]
            assert lengths.tolist()==host_len
            assert pos.tolist()==[n-1 for n in host_len]
            assert alive.tolist()==host_live
            expected=torch.full_like(mask,torch.finfo(mask.dtype).min)
            for b,n in enumerate(host_len):expected[b,0,0,:n]=0
            torch.testing.assert_close(mask,expected,rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires reserved GPU0')
@pytest.mark.parametrize('minimum,eos', [(1,set()),(2,{2}),(4,{2,7})])
def test_static_decode_device_and_original_host_match(monkeypatch,minimum,eos):
    from sparse_attn.baseline_runtime import BatchCtx
    from sparse_attn.ops.quality_graph import static_decode
    from sparse_attn.ops.tli_static_cache import TLIStaticCache

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.zeros(1,device='cuda',dtype=torch.bfloat16))
            self.config=SimpleNamespace(num_hidden_layers=1,model_type='toy')
        def forward(self,input_ids,past_key_values,**kw):
            values=input_ids.to(torch.bfloat16).view(4,1,1,1).expand(4,1,1,4)
            past_key_values.update(values,values,0)
            logits=torch.nn.functional.one_hot((input_ids+1)%8,8).float()
            return SimpleNamespace(logits=logits)

    results=[]
    for graph_mode,input_mode in [('0','0'),('1','0'),('1','1')]:
        monkeypatch.setenv('TLI_CUDA_GRAPH',graph_mode)
        monkeypatch.setenv('TLI_DEVICE_DECODE_INPUTS',input_mode)
        model=Toy();lens=[10000,10003,10007,10011]
        cache=TLIStaticCache(model.config,16384,4)
        cache.layers[0].lazy_initialization(torch.zeros(4,1,1,4,device='cuda',dtype=torch.bfloat16))
        ctx=BatchCtx(4,list(lens),[0]*4,[True,True,False,True],0)
        gen=[[1],[5,6],[3],[7]]
        with torch.inference_mode():
            output=static_decode(model,cache,gen,lens,eos,7,ctx,min_new_tokens=minimum)
        result=(output,list(ctx.seq_lens),list(ctx.gen_lens),list(ctx.alive),
                cache.layers[0].keys.cpu())
        if results:
            assert result[:4]==results[0][:4]
            torch.testing.assert_close(result[4],results[0][4],rtol=0,atol=0)
        results.append(result)
        receipt=model._quality_graph_execution
        assert receipt['input_update_route']==('device_row_fused' if input_mode=='1' else 'host_reference')
        assert receipt['input_update_fallback'] is None
