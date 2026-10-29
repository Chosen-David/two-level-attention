"""Unsupported gate scopes must fail before cache or calibration changes."""
from pathlib import Path
import runpy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sparse_attn.baseline_runtime import BatchCtx, batch_decode_scope
from sparse_attn.patches.glm4_moe_lite_attn_patch import glm4_moe_lite_attn_forward


@pytest.mark.parametrize("batch,outer_batch,masked", [(2, None, False),
                                                       (1, 2, False),
                                                       (1, 1, True)])
def test_gate_rejects_unsupported_request_before_any_cache_update(batch, outer_batch, masked):
    indexer = SimpleNamespace(dyn_dgate=True, clear=Mock(), calibrate_dyn_dgate=Mock())
    attn = SimpleNamespace(indexer=indexer)
    cache = SimpleNamespace(update=Mock())
    ctx = None if outer_batch is None else BatchCtx(
        outer_batch, [4]*outer_batch, [0]*outer_batch, [True]*outer_batch, 0)
    # No projection/config fields: reaching any computation before the guard
    # also fails this test, rather than merely checking the error wording.
    with batch_decode_scope(ctx), pytest.raises(RuntimeError, match="GLM dynamic far-skip"):
        glm4_moe_lite_attn_forward(
            attn, torch.zeros(batch, 4, 16), (torch.ones(1), torch.zeros(1)),
            torch.ones(batch, 4, dtype=torch.bool) if masked else None,
            past_key_values=cache)
    cache.update.assert_not_called()
    indexer.clear.assert_not_called()
    indexer.calibrate_dyn_dgate.assert_not_called()


def test_batch_cli_rejects_glm_gate_before_loading_weights(monkeypatch):
    import benchmark.RULER.pred_ruler_batch as runner
    helpers = runpy.run_path(str(Path(__file__).parents[1] /
                                "exp/bench/test_glm47_d256_gates.py"))
    args = helpers["make_prod_args"]()
    args.tli_dyn_dgate = True
    args.batch_size = 2
    args.pred_postfix = "_guard"
    args.model_path = "unused"
    args.yarn = False
    args.yarn_factor = None
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "baseline_preflight", lambda _: None)
    monkeypatch.setattr(runner, "seed_everything", lambda: None)
    monkeypatch.setattr(runner.AutoConfig, "from_pretrained", lambda *a, **k:
                        SimpleNamespace(model_type="glm4_moe_lite", max_position_embeddings=202752))
    load = Mock(side_effect=AssertionError("weights must not load"))
    monkeypatch.setattr(runner.AutoModelForCausalLM, "from_pretrained", load)
    with pytest.raises(SystemExit, match="dynamic far-skip"):
        runner.main()
    load.assert_not_called()


def test_shared_batch_helper_preserves_indexers_without_dynamic_gate_api():
    from sparse_attn.patches.qwen3_attn_patch import _tl_decode_batched
    length, heads, dim = 8, 2, 4
    q = torch.arange(heads*dim, dtype=torch.float32).reshape(1,heads,1,dim) / 10
    k = torch.arange(heads*length*dim, dtype=torch.float32).reshape(1,heads,length,dim) / 50
    v = k.flip(-1).contiguous()
    indexer = SimpleNamespace(clear=Mock(), prepare_mask=Mock(return_value=(
        torch.ones(1,1,heads,length,dtype=torch.bool), 1)))
    ctx = BatchCtx(1,[length],[0],[True],0)
    actual = _tl_decode_batched(SimpleNamespace(scaling=dim**-0.5), indexer,q,k,v,ctx)
    expected = torch.nn.functional.scaled_dot_product_attention(q,k,v).transpose(1,2)
    torch.testing.assert_close(actual,expected)
    indexer.clear.assert_called_once()
