"""Actual-main two-layer hybrid geometry build counts; no GPU timing."""
import copy
import json
from pathlib import Path
from types import MethodType
from unittest.mock import patch
import torch
from test_hybrid_batch_cache import build, make_model
import sparse_attn.indexer.tli_indexer as tli
from sparse_attn.patches.qwen3_attn_patch import qwen3_attn_forward

torch.set_num_threads(1)
rows = []
with torch.no_grad(), patch.object(tli, '_HAS_TRITON', False):
    for batch in (1, 2, 8, 9, 10, 32):
        tli._GRAPH_GEO_CACHE.clear()
        model = make_model()
        first, second = [layer.self_attn for layer in model.model.layers]
        second.indexer = copy.deepcopy(first.indexer)
        second.indexer.layer_idx = 1
        second.indexer._skip_ids = None
        second.indexer.dyn_dgate = True
        second.indexer.dyn_dgate_eps = -1.0
        second.forward = MethodType(qwen3_attn_forward, second)
        lengths = [23+b % 5 for b in range(batch)]
        runner = build(model, lengths, [False]*batch)
        calls = []
        original = tli.TLIIndexer._build_graph_geometry
        def track(self, *args, **kwargs):
            calls.append(self.layer_idx)
            return original(self, *args, **kwargs)
        with patch.object(tli.TLIIndexer, '_build_graph_geometry', track):
            for step in range(2):
                calls.clear()
                runner.inject([3]*batch, [n+step+1 for n in lengths])
                assert torch.isfinite(runner.run().logits).all()
                rows.append(dict(batch=batch, step=step, layer0=calls.count(0),
                                 layer1=calls.count(1), cache_entries=len(tli._GRAPH_GEO_CACHE)))
receipt = dict(main='8f213e48926431ea2d56cf3e7bc1a667d1105c74',
               torch=torch.__version__, cuda_available=torch.cuda.is_available(), records=rows)
Path(__file__).with_name('geometry.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))
