"""Opt-in B1 Qwen decode trace. Adds copies/readbacks; never use for timing.

Captured clones are updated by replay. No host access occurs during capture.
The observer returns the original attention output and restores all patches.
"""
from contextlib import contextmanager
from pathlib import Path

import torch


@contextmanager
def record_decode_trace(directory, max_steps=32):
    from sparse_attn.patches import qwen3_attn_patch as patch
    from sparse_attn.ops.cuda_graph_state import graph_state

    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    attention = patch.eager_decoding_attn
    replay = torch.cuda.CUDAGraph.replay
    buffers = []
    graph_steps = {}
    eager_call = 0

    def snapshot(q, k, v, mask, output):
        if q.shape[:2] != (1, 1) or mask.dtype != torch.bool:
            raise ValueError("trace requires B1 single-token boolean selection")
        return {"q": q.detach().clone(), "mask": mask.detach().clone(),
                "output": output.detach().clone()}

    def save(name, records):
        # Each CPU copy synchronizes its source. Files are never overwritten.
        rows = [{key: value.detach().cpu() for key, value in row.items()}
                for row in records]
        with (root / name).open("xb") as stream:
            torch.save(rows, stream)

    def observe(q, k, v, mask, *args, **kwargs):
        nonlocal eager_call
        output = attention(q, k, v, mask, *args, **kwargs)
        if graph_state().enabled:
            if torch.cuda.is_current_stream_capturing():
                buffers.append(snapshot(q, k, v, mask, output))
        else:
            # Eager records are ordered by actual invocation, not an assumed
            # layer count; consumers must bind them to the model's layers.
            save(f"eager_{eager_call:06d}.pt", [snapshot(q, k, v, mask, output)])
            eager_call += 1
        return output

    def observed_replay(graph):
        result = replay(graph)
        key = graph  # retain identity across requests; Python ids can be reused
        if key not in graph_steps:
            graph_steps[key] = [len(graph_steps), 0, list(buffers)]
            buffers.clear()
        request, step, records = graph_steps[key]
        if step < max_steps:
            save(f"graph_{request:03d}_{step:03d}.pt", records)
        graph_steps[key][1] += 1
        return result

    patch.eager_decoding_attn = observe
    torch.cuda.CUDAGraph.replay = observed_replay
    try:
        yield
    finally:
        patch.eager_decoding_attn = attention
        torch.cuda.CUDAGraph.replay = replay
