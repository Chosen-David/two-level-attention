import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch


def test_observer_copies_without_changing_output_and_restores_after_exception(tmp_path,monkeypatch):
    from sparse_attn.patches import qwen3_attn_patch as patch
    from sparse_attn.ops import exact_linear_batch as exact
    from sparse_attn.ops.cuda_graph_state import graph_state
    path=Path(__file__).parents[1]/'exp/bench/batch_decode_trace.py'
    spec=importlib.util.spec_from_file_location('batch_trace',path)
    trace=importlib.util.module_from_spec(spec);spec.loader.exec_module(trace)
    q=torch.randn(2,4,1,8);k=torch.randn(2,7,2,8);mask=torch.ones(2,1,2,7,dtype=torch.bool)
    output=q.transpose(1,2).contiguous()
    def attention(*args,**kw):return output
    def outer(module,indexer,q,k,v,ctx):
        return patch.eager_decoding_attn(q.transpose(1,2),k,v,mask)
    monkeypatch.setattr(patch,'_tl_decode_batched',outer)
    monkeypatch.setattr(patch,'eager_decoding_attn',attention)
    monkeypatch.setattr(graph_state(),'enabled',False)
    originals=exact.partial_attention,torch.cuda.CUDAGraph.replay
    with pytest.raises(RuntimeError,match='after observation'):
        with trace.record_batch_decode_trace(tmp_path/'trace'):
            result=patch._tl_decode_batched(SimpleNamespace(layer_idx=3),None,q,k,k,None)
            assert result is output
            raise RuntimeError('after observation')
    assert patch._tl_decode_batched is outer and patch.eager_decoding_attn is attention
    assert (exact.partial_attention,torch.cuda.CUDAGraph.replay)==originals
    row=torch.load(tmp_path/'trace/eager_layer003_step000.pt',weights_only=True)[0]
    torch.testing.assert_close(row['q'],q.transpose(1,2),rtol=0,atol=0)
    torch.testing.assert_close(row['output'],output,rtol=0,atol=0)
    assert torch.equal(row['mask'],mask) and row['layer']==3
