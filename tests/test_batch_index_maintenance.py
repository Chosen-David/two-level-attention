"""Metadata and queued native maintenance gates. Never timing evidence."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from sparse_attn.ops.batch_index_maintenance import validate_maintenance,maintain_batched,maintenance_route


def _metadata(device='cpu',batch=2,capacity=65):
    k=torch.empty(batch,capacity,2,128,device=device,dtype=torch.bfloat16)
    summaries=[torch.empty(batch,(capacity+63)//64,2,32,device=device,dtype=k.dtype) for _ in range(3)]
    fine=torch.empty(batch,capacity,2,32,device=device,dtype=k.dtype)
    return k,torch.zeros(batch,device=device,dtype=torch.int64),torch.ones(batch,device=device,dtype=torch.bool),torch.arange(32,device=device),*summaries,fine


def test_cpu_schema_and_no_implicit_launch():
    values=_metadata();validate_maintenance(*values)
    with pytest.raises(ValueError,match='requires CUDA'):maintain_batched(*values)


@pytest.mark.parametrize('kind',['dtype','positions','alive','indices','summary','fine','alias','block'])
def test_unsupported_metadata_rejected(kind):
    v=list(_metadata());kw={}
    if kind=='dtype':v[0]=v[0].float()
    elif kind=='positions':v[1]=v[1].to(torch.int32)
    elif kind=='alive':v[2]=torch.ones(2,1,dtype=torch.bool)
    elif kind=='indices':v[3]=torch.arange(16)
    elif kind=='summary':v[4]=v[4][...,:16]
    elif kind=='fine':v[7]=v[7].transpose(1,2)
    elif kind=='alias':v[5]=v[4]
    elif kind=='block':kw['block_size']=32
    with pytest.raises(ValueError):validate_maintenance(*v,**kw)


def test_large_input_batch_stride_is_validated_without_allocating_storage():
    v=list(_metadata(device='meta'))
    v[0]=torch.empty_strided((2,65,2,128),(2**32,256,128,1),device='meta',dtype=torch.bfloat16)
    validate_maintenance(*v)


def test_kernel_widens_before_batch_products_and_preserves_reduction_nodes():
    source=Path(__file__).parents[1]/'sparse_attn/indexer/tli_indexer.py'
    tree=ast.parse(source.read_text());fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_iter3_maint_kernel')
    text=ast.unparse(fn)
    assert 'tl.program_id(0).to(tl.int64)' in text
    for stride in ('STRIDE_KB','STRIDE_MINB','STRIDE_MAXB','STRIDE_AVGB','STRIDE_QB'):
        assert 'b * '+stride in text
    assert 'tl.min(v, axis=0)' in text and 'tl.max(v, axis=0)' in text
    assert 'tl.sum(v.to(tl.float32), axis=0) / BS' in text
    assert 'rv = tl.load(rp, mask=active, other=0.0)' in text


@pytest.mark.parametrize('field,value',[('subspace','full'),('cmp_ratio',2),('l2_subspace','full'),('l2_quant',True),('_triton_ok',False)])
def test_configuration_has_explicit_row_fallback(field,value):
    idx=SimpleNamespace(enable_subspace=True,subspace='tail',cmp_ratio=4,head_dim=128,
        l2_subspace='same',l2_quant=False,args=SimpleNamespace(tia_block_size=64),_triton_ok=True)
    setattr(idx,field,value);v=_metadata()
    route=maintenance_route(idx,v[0],v[1],v[2],v[3],dict(zip(('k_min','k_max','k_avg'),v[4:7])),v[7])
    assert route=='row_fallback:configuration'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='queued native GPU gate')
@pytest.mark.parametrize('batch',[2,4,8])
@pytest.mark.parametrize('capacity',[65,129,32768])
@pytest.mark.parametrize('with_average',[False,True])
def test_native_batched_maintenance_exact_primary_rows(batch,capacity,with_average):
    from sparse_attn.indexer.tli_indexer import _iter3_maint
    torch.manual_seed(320304)
    values=list(_metadata('cuda',batch,capacity));k,pos,alive,indices,lo,hi,avg,fine=values
    # Both supported raw-K layouts must use batch strides, not flattened heads.
    if batch%4==0:k=k.transpose(1,2).contiguous().transpose(1,2)
    k.normal_();indices.copy_(torch.cat((torch.arange(48,64,device='cuda'),torch.arange(112,128,device='cuda'))))
    for t in (lo,hi,avg,fine):t.fill_(-7)
    if not with_average:avg=None
    reference=[t.clone() if t is not None else None for t in (lo,hi,avg,fine)]
    positions=[min(capacity-1,62+b%2) for b in range(batch)]
    pos.copy_(torch.tensor(positions,device='cuda'))
    def run():return maintain_batched(k,pos,alive,indices,lo,hi,avg,fine)
    run();torch.cuda.synchronize()
    # Reset identical starting state before capture/replay/reference comparison.
    for target,expected in zip((lo,hi,avg,fine),reference):
        if target is not None:target.copy_(expected)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):run()
    for step in range(4):
        if step==1:alive[-1]=False
        positions=[min(capacity-1,p+1) if b!=batch-1 or step==0 else p for b,p in enumerate(positions)]
        pos.copy_(torch.tensor(positions,device='cuda'))
        # All raw K may change; the primitive must nevertheless leave dead
        # arena rows byte-for-byte unchanged, including arbitrary initial data.
        k.normal_()
        graph.replay();torch.cuda.synchronize()
        for b in range(batch):
            if b==batch-1 and step>=1:continue
            _iter3_maint(k[b],pos[b:b+1],indices,reference[0][b],reference[1][b],
                reference[2][b] if with_average else None,reference[3][b],capacity,64,with_average)
        for actual,expected in zip((lo,hi,avg,fine),reference):
            if actual is not None:torch.testing.assert_close(actual,expected,rtol=0,atol=0,equal_nan=True)
    # Same arena refill/reinitialize, including the partial final storage block.
    for target,expected in zip((lo,hi,avg,fine),reference):
        if target is not None:target.fill_(3);expected.fill_(3)
    pos.fill_(capacity-1);alive.fill_(True);k.normal_()
    graph.replay();torch.cuda.synchronize()
    for b in range(batch):
        _iter3_maint(k[b],pos[b:b+1],indices,reference[0][b],reference[1][b],
            reference[2][b] if with_average else None,reference[3][b],capacity,64,with_average)
    for actual,expected in zip((lo,hi,avg,fine),reference):
        if actual is not None:torch.testing.assert_close(actual,expected,rtol=0,atol=0,equal_nan=True)
