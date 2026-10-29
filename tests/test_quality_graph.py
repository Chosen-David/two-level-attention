import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sparse_attn.baseline_runtime import BatchCtx
from sparse_attn.ops.quality_graph import (
    graph_policy, from_dynamic, static_decode, owned_decode_mask,
)
from sparse_attn.ops.tli_static_cache import TLIStaticCache


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(num_hidden_layers=1, model_type="toy")

    def forward(self, input_ids, past_key_values, **kwargs):
        batch = input_ids.shape[0]
        states = input_ids.float().view(batch,1,1,1).expand(batch,1,1,4)
        past_key_values.update(states,states,0)
        logits = torch.nn.functional.one_hot((input_ids+1)%8,8).float()
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("value,expected", [(None,"auto"),("0","off"),("1","on"),
                                           ("auto","auto"),("off","off"),("on","on")])
def test_policy(monkeypatch,value,expected):
    monkeypatch.delenv("TLI_CUDA_GRAPH",raising=False)
    if value is not None:monkeypatch.setenv("TLI_CUDA_GRAPH",value)
    assert graph_policy()==expected


@pytest.mark.parametrize("policy",["0","auto"])
def test_ragged_eos_rows_freeze_and_buffers_are_request_local(monkeypatch,policy):
    monkeypatch.setenv("TLI_CUDA_GRAPH",policy)
    model=Toy()
    for lens in ([2,5],[1,3]):
        cache=TLIStaticCache(model.config,16,2)
        cache.layers[0].lazy_initialization(torch.zeros(2,1,1,4))
        ctx=BatchCtx(2,list(lens),[0,0],[True,True],0)
        result=static_decode(model,cache,[[1],[3]],lens,{2},5,ctx)
        assert result==[[1,2],[3,4,5,6,7]]
        assert ctx.seq_lens==[lens[0]+1,lens[1]+4]
        assert cache.layers[0].keys[0,0,lens[0],0].item()==1
        assert cache.layers[0].keys[0,0,lens[0]+1:,0].eq(0).all()
        assert model._quality_graph_execution['captures']==0
        assert model._quality_graph_execution['actual']=='staged_static'
        assert cache._quality_decode_ctx is None


def test_strict_graph_rejects_cpu_before_cache_write(monkeypatch):
    monkeypatch.setenv("TLI_CUDA_GRAPH","1")
    model=Toy(); cache=TLIStaticCache(model.config,32,1)
    ctx=BatchCtx(1,[2],[0],[True],0)
    with pytest.raises(RuntimeError,match='non_cuda'):
        static_decode(model,cache,[[1]],[2],set(),3,ctx)
    assert cache.layers[0].keys is None


def test_no_decode_needs_no_graph_even_when_required(monkeypatch):
    monkeypatch.setenv("TLI_CUDA_GRAPH","1")
    model=Toy()
    assert from_dynamic(model,None,2,1,{2})==[2]
    assert from_dynamic(model,None,2,9,{2},min_new_tokens=1)==[2]
    assert model._quality_graph_execution['actual']=='no_decode'


def test_owned_mask_requires_exact_buffer_and_context():
    ctx=BatchCtx(1,[2],[0],[True],0)
    mask=torch.zeros(1,1,1,16)
    cache=SimpleNamespace(_quality_decode_ctx=ctx,_quality_decode_mask=mask,
                          get_max_cache_shape=lambda:16)
    assert owned_decode_mask(cache,mask,ctx)
    assert not owned_decode_mask(cache,mask.clone(),ctx)
    assert not owned_decode_mask(cache,mask,BatchCtx(1,[2],[0],[True],0))


def test_batch_generation_guard_exits_before_any_prediction_write():
    path=Path(__file__).parents[1]/'benchmark/RULER/pred_ruler_batch.py'
    tree=ast.parse(path.read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    guards=[n for n in ast.walk(main) if isinstance(n,ast.With) and any(
        isinstance(i.context_expr,ast.Call) and isinstance(i.context_expr.func,ast.Name)
        and i.context_expr.func.id=='eager_decoding_request_guard' for i in n.items)]
    assert len(guards)==1
    guard=guards[0]
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
               and n.func.id=='_decode_batch_chunk' for n in ast.walk(guard))
    writes=[n for n in ast.walk(main) if isinstance(n,ast.Call)
            and isinstance(n.func,ast.Attribute) and n.func.attr=='write']
    assert writes and all(n.lineno>guard.end_lineno for n in writes)


def test_error_flag_rejects_output_before_commit_and_next_request_recovers(tmp_path):
    from sparse_attn.ops.eager_decoding import _ERR_FLAG, eager_decoding_request_guard
    key=torch.device('cpu');before=_ERR_FLAG.get(key)
    try:
        for code in (1,2,3):
            with pytest.raises(RuntimeError):
                with eager_decoding_request_guard():
                    _ERR_FLAG[key]=torch.tensor([code],dtype=torch.int32)
                (tmp_path/'complete').write_text('invalid')
            assert not (tmp_path/'complete').exists()
        with eager_decoding_request_guard():
            pass
    finally:
        if before is None:_ERR_FLAG.pop(key,None)
        else:_ERR_FLAG[key]=before


@pytest.mark.parametrize("timing", ["0", "1"])
def test_timing_preserves_ragged_eos_and_counts_only_live_tokens(monkeypatch, timing):
    monkeypatch.setenv("TLI_CUDA_GRAPH", "0")
    monkeypatch.setenv("TLI_QUALITY_TIMING", timing)
    model = Toy()
    cache = TLIStaticCache(model.config, 16, 2)
    cache.layers[0].lazy_initialization(torch.zeros(2, 1, 1, 4))
    ctx = BatchCtx(2, [2, 5], [0, 0], [True, True], 0)
    assert static_decode(model, cache, [[1], [3]], [2, 5], {2}, 5, ctx) == [
        [1, 2], [3, 4, 5, 6, 7]]
    record = model._quality_graph_execution
    if timing == "0":
        assert "timing" not in record
    else:
        measured = record["timing"]
        assert measured["decode_steps"] == 4
        assert measured["generated_decode_tokens"] == 5
        assert measured["replay_gpu_ms"] == []
        assert measured["replay_gpu_mean_ms"] is None
        assert measured["decode_wall_ms"] >= sum(measured["step_wall_ms"]) > 0
        assert measured["tpot_wall_ms"] == sum(measured["step_wall_ms"]) / 4
