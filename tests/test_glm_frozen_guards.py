"""No GPU: unsupported requests must fail before projections or KV mutation."""
from types import SimpleNamespace as NS
import pytest
import torch
from sparse_attn.patches.glm4_moe_lite_attn_patch import glm4_moe_lite_attn_forward
from sparse_attn.baseline_runtime import batch_decode_scope
from sparse_attn.ops.cuda_graph_state import graph_state,disable_graph

@pytest.mark.parametrize('case',['outer_batch','graph','operator','batch','mask','chunk','unfrozen'])
def test_region_rejects_before_projection(monkeypatch,case):
    monkeypatch.setenv('TLI_BATCH_INDEXER','legacy')
    monkeypatch.setenv('TLI_SPARSE_PREFILL','0')
    monkeypatch.setenv('TLI_DECODE_DENSE','0')
    monkeypatch.setenv('TLI_OPERATOR_MODE','off')
    disable_graph()
    # Intentionally no q_proj or cache.update: touching either would fail.
    module=NS(indexer=NS(dyn_dgate=False),layer_idx=0)
    layer=NS(logical_length=8,frozen=True)
    cache=NS(tli_regions=True,layers=[layer])
    hidden=torch.zeros(1,1,8);mask=None;ctx=None
    if case=='outer_batch':ctx=NS(batch=2)
    if case=='graph':graph_state().enabled=True
    if case=='operator':monkeypatch.setenv('TLI_OPERATOR_MODE','auto')
    if case=='batch':hidden=torch.zeros(2,1,8)
    if case=='mask':mask=torch.zeros(1,1,1,9)
    if case=='chunk':hidden=torch.zeros(1,2,8)
    if case=='unfrozen':layer.frozen=False
    try:
        with batch_decode_scope(ctx),pytest.raises(ValueError,match='GLM frozen regions require'):
            glm4_moe_lite_attn_forward(module,hidden,None,mask,cache)
        assert layer.logical_length==8
    finally:
        disable_graph()
