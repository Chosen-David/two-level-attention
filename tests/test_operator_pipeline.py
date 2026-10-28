"""Independent full-domain oracle, intermediate IDs, real model and Graph gates."""
import os
import pytest
import torch
from sparse_attn.ops.far_skip import FarSkipLayout
from sparse_attn.ops.token_selection import TokenSelection, canonical_selection
from sparse_attn.ops.partial_attention import partial_attention, merge_partials
from sparse_attn.ops.operator_pipeline import OperatorPlan


def fixture(device='cpu', dtype=torch.float32, *, length=93, batch=2, dq=32, dv=24, ring=False, group=4):
    hk = 8 // group
    gen = torch.Generator(device=device).manual_seed(728)
    q = torch.randn(batch, 1, 8, dq, generator=gen, device=device, dtype=dtype)
    k = torch.randn(batch, max(length, 1), hk, dq, generator=gen, device=device, dtype=dtype)
    v = torch.randn(batch, max(length, 1), hk, dv, generator=gen, device=device, dtype=dtype)
    lengths = torch.full((batch,), length, device=device, dtype=torch.int64)
    lo = max(4, length - 31) if ring else 4
    tokens = torch.arange(lo, max(lo, length - 9), 3, device=device).reshape(1, 1, -1).expand(batch, hk, -1)
    sel = canonical_selection(tokens, torch.ones_like(tokens, dtype=torch.bool), max(1, length))
    full = (k, v)
    layout = FarSkipLayout(4, 31) if ring else None
    if ring:
        kr = torch.full((batch, layout.capacity, hk, dq), float('nan'), device=device, dtype=dtype)
        vr = torch.full((batch, layout.capacity, hk, dv), float('nan'), device=device, dtype=dtype)
        live = list(range(min(4, length))) + list(range(max(4, length-31), length))
        for p in live:
            kr[:, layout.slot(p)] = k[:, p]; vr[:, layout.slot(p)] = v[:, p]
        k, v = kr, vr
    return q, k, v, sel, lengths, layout, full


def oracle(q, full, selection, lengths, sink=4, swa=9, scale=.17):
    """Full logical-domain attention in float64; does not split or call kernels."""
    k, v = full
    b, _, h, d = q.shape
    hk, n = k.shape[2], k.shape[1]
    out = torch.zeros(b, 1, h, v.shape[-1], device=q.device, dtype=q.dtype)
    for row in range(b):
        length = int(lengths[row])
        for head in range(h):
            group = head // (h // hk)
            ids = set(selection.indices[row, group, :int(selection.counts[row, group])].tolist())
            ids.update(range(min(sink, length)))
            ids.update(range(max(0, length-swa), length))
            ids = sorted(i for i in ids if 0 <= i < min(length, n))
            if ids:
                scores = k[row, ids, group].double() @ (q[row, 0, head]*scale).double()
                out[row, 0, head] = (scores.softmax(0) @ v[row, ids, group].double()).to(q.dtype)
    return out


@pytest.mark.parametrize('length', [0, 1, 3, 9, 17, 93])
@pytest.mark.parametrize('ring', [False, True])
@pytest.mark.parametrize('splits', [1, 4])
def test_partial_oracle_empty_overlap_and_ring(length, ring, splits):
    q,k,v,sel,lens,layout,full = fixture(length=length, ring=ring)
    opts = dict(sink=4, swa=9, scale=.17, backend='torch', layout=layout)
    protected = partial_attention(q,k,v,sel,lens,protected=True,splits=2,**opts)
    selected = partial_attention(q,k,v,sel,lens,splits=splits,**opts)
    out = merge_partials(protected,selected,q.dtype,backend='torch')
    torch.testing.assert_close(out,oracle(q,full,sel,lens),atol=2e-6,rtol=2e-5)


def test_canonical_ids_preserve_mask_scan_and_ignore_bad_padding():
    ids=torch.tensor([[[7,4,7,999,-1,2]]]);valid=torch.tensor([[[1,1,1,0,0,1]]],dtype=torch.bool)
    sel=canonical_selection(ids,valid,16)
    assert sel.counts.tolist()==[[3]]
    assert sel.indices[0,0,:3].tolist()==[2,4,7]
    assert sel.to_mask(16).flatten().nonzero().flatten().tolist()==[2,4,7]


def test_heterogeneous_qk_v_dimensions():
    q,k,v,sel,lens,layout,full=fixture(dq=192,dv=128,group=1)
    plan=OperatorPlan(q.device,'serial','torch',4,1)
    got=plan.run(q,k,v,lens,lambda:sel,sink=4,swa=9,scale=.17)
    torch.testing.assert_close(got,oracle(q,full,sel,lens),atol=3e-6,rtol=2e-5)


def test_unused_id_slots_cannot_read_nan_or_out_of_range_kv():
    q,k,v,sel,lens,layout,full=fixture()
    padded=torch.cat((sel.indices,torch.full((2,2,3),999,dtype=torch.int32)),-1)
    selection=TokenSelection(padded,sel.counts)
    plan=OperatorPlan(q.device,'serial','torch',4,1)
    got=plan.run(q,k,v,lens,lambda:selection,sink=4,swa=9,scale=.17)
    torch.testing.assert_close(got,oracle(q,full,sel,lens),atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('length',[1,15,31,63])
@pytest.mark.parametrize('bypass',[False,True])
def test_graph_selector_same_selected_set(length,bypass,monkeypatch):
    from test_far_skip_hybrid_cache import make_model
    monkeypatch.setenv('TLI_L2_BYPASS','1' if bypass else '0')
    idx=make_model().model.layers[0].self_attn.indexer
    idx._skip_ids=set();idx.skip_far=False
    torch.manual_seed(88)
    k=torch.randn(1,64,1,128);q=torch.randn(1,1,2,128)
    cu=torch.tensor([0,64],dtype=torch.int32);lens=torch.tensor([length])
    with torch.no_grad():
        index=idx.prepare_index(k,cu)
        mask=idx.compute_mask_graph(q,index,.1,lens)
        selected=idx.compute_selection_graph(q,index,.1,lens)
    full=selected.to_mask(64)
    pos=torch.arange(64)
    protected=((pos<idx.sink_tokens)|(pos>=length-idx.sliding_window_size))&(pos<length)
    assert torch.equal(full|protected,mask)
    assert not bool((full&protected).any())



CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='native CUDA required')


@CUDA
@pytest.mark.parametrize('backend',['simt','tc'])
@pytest.mark.parametrize('ring',[False,True])
@pytest.mark.parametrize('group',[1,4])
@pytest.mark.parametrize('dq,dv',[(128,128),(192,128)])
def test_cuda_kernels_and_multistream_graph(backend,ring,group,dq,dv):
    pytest.importorskip('triton')
    q,k,v,sel,lens,layout,full=fixture('cuda',torch.bfloat16,dq=dq,dv=dv,ring=ring,group=group)
    plan=OperatorPlan(q.device,'overlap',backend,4,1)
    def run():
        # Put selection work inside the capturing branch; IDs remain on-device.
        return plan.run(q,k,v,lens,lambda:TokenSelection(sel.indices.clone(),sel.counts.clone()),sink=4,swa=9,scale=.17,layout=layout)
    side=torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):out=run()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    torch.testing.assert_close(out,oracle(q,full,sel,lens),atol=.025,rtol=.025)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=run()
    for step in range(3):
        q.add_(.03125)
        lens.fill_(93-step)  # replay must not freeze protected boundaries
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out,oracle(q,full,sel,lens),atol=.025,rtol=.025)


