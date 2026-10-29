"""Both BF16-only fused mean call sites must preserve other input dtypes."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_fused_gqa_mean_dispatch_dtype(dtype):
    source = Path(__file__).parents[1] / 'sparse_attn/indexer/tli_indexer.py'
    tree = ast.parse(source.read_text())
    guards = [n.test for n in ast.walk(tree) if isinstance(n, ast.If)
              and '_HAS_TRITON' in ast.unparse(n.test)
              and any(isinstance(x, ast.Expr) and '_ako_gqa_mean_kernel[' in ast.unparse(x)
                      for x in n.body)]
    assert len(guards) == 2
    sm = torch.tensor([.2501, .2502, .4997], dtype=dtype)
    for guard in guards:
        scope = dict(torch=torch, self=SimpleNamespace(per_q_head=False),
                     _HAS_TRITON=True, _G_=1, group_size=1,
                     _sm=sm.reshape(1,1,1,3), sm=sm.reshape(1,3))
        assert bool(eval(compile(ast.Expression(guard), 'guard', 'eval'), scope)) == (dtype == torch.bfloat16)
