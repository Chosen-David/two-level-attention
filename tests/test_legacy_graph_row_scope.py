"""Legacy Graph must slice all live metadata before each per-row indexer call."""
import pytest
import torch
from test_exact_linear_batch_gpu import CONFIG,_case
from sparse_attn.ops.exact_linear_batch import _row_scope

NATIVE=pytest.mark.skipif(not torch.cuda.is_available(),reason='queued native GPU required')


@NATIVE
@pytest.mark.parametrize('batch',[2,4])
def test_legacy_graph_ragged_and_dead_rows_use_their_own_positions(monkeypatch,batch):
    for key,value in CONFIG.items():monkeypatch.setenv(key,value)
    for key,value in dict(TLI_BATCH_INDEXER='legacy',TLI_OPERATOR_MODE='off',
                          TLI_LIVE_NORMALIZATION='1',TLI_SPARSE_STATS='owner').items():
        monkeypatch.setenv(key,value)
    with torch.no_grad(),_case(batch) as case:
        positions,lengths,alive=case.gs.pos_buf,case.gs.valid_len_buf,case.gs.alive_buf
        prepare,mask=case.source.prepare_index,case.source.compute_mask_graph
        calls=[];captured=[]
        def checked_prepare(k,cu):
            row=len(calls)%batch
            assert case.gs.bs==1, 'legacy per-row indexer received whole-batch graph state'
            for current,base in zip((case.gs.pos_buf,case.gs.valid_len_buf,case.gs.alive_buf),
                                    (positions,lengths,alive)):
                assert current.shape==(1,) and current.data_ptr()==base[row:].data_ptr()
            calls.append(row)
            return prepare(k,cu)
        def observed_mask(*args,**kwargs):
            result=mask(*args,**kwargs)
            if torch.cuda.is_current_stream_capturing():captured.append(result.clone())
            return result
        monkeypatch.setattr(case.source,'prepare_index',checked_prepare)
        monkeypatch.setattr(case.source,'compute_mask_graph',observed_mask)
        for _ in range(3):case.run()
        torch.cuda.synchronize();graph,out=case.capture()
        assert len(captured)==batch
        for step in range(3):
            case.alive[-1]=False
            case.lengths=[n+int(active) for n,active in zip(case.lengths,case.alive)]
            case.inject_metadata();case.q.normal_()
            graph.replay();torch.cuda.synchronize()
            for b,ref in enumerate(case.refs):
                with _row_scope(case.gs,b):
                    index=ref.prepare_index(case.k[b:b+1],case.cu)
                    expected=ref.compute_mask_graph(case.q[b:b+1],index,case.scale,case.gs.valid_len_buf)
                    assert torch.equal(captured[b],expected)
            torch.testing.assert_close(out,case.reference(),rtol=.02,atol=.003)
        assert case.gs.bs==batch and case.gs.pos_buf is positions
        assert case.gs.valid_len_buf is lengths and case.gs.alive_buf is alive
