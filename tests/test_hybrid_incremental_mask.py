"""Runner mask recurrence equals full reconstruction, including EOS/capacity."""
from types import SimpleNamespace
import pytest
import torch
from sparse_attn.ops.hybrid_batch_runner import HybridBatchRunner

@pytest.mark.parametrize('batch', [1, 8, 32])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@torch.inference_mode()
def test_mask_sequence(batch, dtype, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA required')
    model = torch.nn.Linear(1, 1, dtype=dtype, device=device).eval()
    cap = 129
    initial = [1+b%8 for b in range(batch)]
    for _ in range(2):
        cache = SimpleNamespace(_hybrid_batch=True, batch_size=batch, max_cache_len=cap, _hybrid_lengths=initial)
        runner = HybridBatchRunner(model, cache)
        pointer = runner.mask.data_ptr()
        lengths = initial.copy()
        for step in range(128):
            alive = [lengths[b]<cap and (b%3==0 or step<12+b) for b in range(batch)]
            lengths = [n+int(a) for n,a in zip(lengths,alive)]
            runner.inject([0]*batch, lengths, alive)
            expected = torch.zeros_like(runner.mask).masked_fill(
                torch.arange(cap,device=device)[None,None,None,:] >= torch.tensor(lengths,device=device)[:,None,None,None], torch.finfo(dtype).min)
            torch.testing.assert_close(runner.mask, expected, rtol=0, atol=0)
            assert runner.mask.data_ptr()==pointer
        # Invalid rewind must leave all model inputs intact.
        saved = runner.mask.clone()
        with pytest.raises(ValueError): runner.inject([0]*batch,[n-1 for n in lengths],[False]*batch)
        torch.testing.assert_close(runner.mask,saved,rtol=0,atol=0)

@torch.inference_mode()
def test_steady_inject_does_not_materialize_capacity_comparison():
    model=torch.nn.Linear(1,1).eval()
    cache=SimpleNamespace(_hybrid_batch=True,batch_size=2,max_cache_len=4096,_hybrid_lengths=[10,20])
    runner=HybridBatchRunner(model,cache)
    runner.inject([0,0],[11,21])
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
        runner.inject([0,0],[12,21],[True,False])
    names={e.key for e in p.key_averages()}
    assert 'aten::scatter_' in names
    assert 'aten::ge' not in names and 'aten::masked_fill_' not in names
