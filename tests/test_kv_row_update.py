import pytest
import torch
from sparse_attn.ops.kv_row_update import append_kv_row


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_append_live_dead_replay(device, dtype):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('requires CUDA')
    torch.manual_seed(883)
    k = torch.randn(1, 3, 17, 128, device=device, dtype=dtype)
    v = torch.randn_like(k)
    # Slice a larger batch: incoming head stride differs from storage.
    nk, nv = torch.randn(2, 4, 3, 1, 128, device=device, dtype=dtype)
    nk, nv = nk[2:3], nv[2:3]
    slot = torch.tensor([16], device=device)
    alive = torch.tensor([True], device=device)
    run = lambda: append_kv_row(k, v, nk, nv, slot, alive)
    if device == 'cuda':
        run(); torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): run()
        execute = graph.replay
    else:
        execute = run
    for pos, live in [(16, True), (0, True), (7, False), (8, True), (16, False)]:
        oldk, oldv = k.clone(), v.clone()
        slot.fill_(pos); alive.fill_(live); nk.normal_(); nv.normal_()
        execute()
        if live:
            oldk[:, :, pos:pos+1].copy_(nk)
            oldv[:, :, pos:pos+1].copy_(nv)
        assert torch.equal(k, oldk)
        assert torch.equal(v, oldv)
