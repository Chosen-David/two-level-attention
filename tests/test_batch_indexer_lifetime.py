from types import SimpleNamespace
import torch
from test_far_skip_hybrid_cache import make_model, isolated_runtime  # noqa: F401
from sparse_attn.patches.qwen3_attn_patch import _tl_decode_batched, _batch_row_indexers
from sparse_attn.baseline_runtime import BatchCtx, batch_decode_scope


def test_eager_rows_keep_incremental_state_and_reset_at_request(monkeypatch):
    monkeypatch.setattr('sparse_attn.indexer.tli_indexer._HAS_TRITON',False)
    monkeypatch.setenv('TLI_BATCH_INDEXER','legacy')
    model=make_model()
    idx=model.model.layers[0].self_attn.indexer
    idx._skip_ids=set();idx.skip_far=False
    idx.far_skip_backend='legacy'
    ctx=BatchCtx(batch=2,seq_lens=[23,31],gen_lens=[0,0],alive=[True,True],pad_id=0)
    q=torch.randn(2,2,1,128);k=torch.randn(2,1,65,128);v=torch.randn_like(k)
    with batch_decode_scope(ctx):
        _tl_decode_batched(SimpleNamespace(scaling=.1),idx,q,k,v,ctx)
        rows=_batch_row_indexers(idx,k.transpose(1,2))
        assert rows[0] is not rows[1]
        def reject(*a,**kw): raise AssertionError('unnecessary full rebuild')
        for row in rows:
            monkeypatch.setattr(row,'_prepare_index_full',reject)
        ctx.seq_lens[:]=[24,32]
        _tl_decode_batched(SimpleNamespace(scaling=.1),idx,q,k,v,ctx)
        assert rows is _batch_row_indexers(idx,k.transpose(1,2))
        idx.clear()
        assert rows is not _batch_row_indexers(idx,k.transpose(1,2))
