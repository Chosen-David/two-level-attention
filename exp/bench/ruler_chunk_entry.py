"""Load the production chunk functions without executing RULER's CLI body.

The function ASTs are compiled unchanged; this is an entry integration probe,
not a replacement implementation of prefill or decode.
"""
import ast
from pathlib import Path

import torch

from sparse_attn.baseline_runtime import BatchCtx, batch_decode_scope


def load_ruler_chunk(max_gen):
    source = Path(__file__).resolve().parents[2] / 'benchmark/RULER/pred_ruler_batch.py'
    names = {'_decode_batch_chunk', '_decode_batch_graph', '_decode_hybrid_batch_chunk'}
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise RuntimeError('RULER production chunk functions changed')
    import os
    namespace = dict(torch=torch, os=os, BatchCtx=BatchCtx,
                     batch_decode_scope=batch_decode_scope, MAX_GEN=max_gen)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['_decode_batch_chunk']


def run_ruler_chunk(model, tokenizer, ids, rows, max_gen):
    lengths = [row.shape[-1] for row in ids]
    width = max(lengths)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    inputs = torch.full((len(ids), width), pad_id, dtype=torch.long, device=ids[0].device)
    positions = torch.zeros_like(inputs)
    for b, (row, length) in enumerate(zip(ids, lengths)):
        inputs[b, -length:] = row[0]
        positions[b, -length:] = torch.arange(length, device=inputs.device)
    return load_ruler_chunk(max_gen)(model, tokenizer, rows, inputs, positions,
                                    None, lengths, set(), False, fixed_cache=True)
