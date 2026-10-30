"""Independent host semantics for device-only decode input preparation."""
import pytest
import torch

from sparse_attn.ops.quality_decode_inputs import DeviceDecodeInputs


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
