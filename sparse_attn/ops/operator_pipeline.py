"""Opt-in selector/protected fork-join, captured as one CUDA Graph DAG.

The enclosing runner owns capture and request lifetime. This module does not
reserve SMs, change Top-K policy or claim concurrency from streams alone.
"""
import os
import torch
from .partial_attention import partial_attention, merge_partials
from .token_selection import middle_selection


def operator_mode():
    mode = os.environ.get('TLI_OPERATOR_MODE', 'off')
    if mode not in ('off', 'serial', 'overlap'):
        raise ValueError('TLI_OPERATOR_MODE must be off, serial or overlap')
    return mode


class OperatorPlan:
    def __init__(self, device, mode, backend, selected_splits, protected_splits):
        self.device = torch.device(device)
        self.mode, self.backend = mode, backend
        self.selected_splits, self.protected_splits = selected_splits, protected_splits
        self.stream = None
        if mode == 'overlap':
            if self.device.type != 'cuda':
                raise ValueError('overlap requires CUDA; use serial/torch for CPU reference tests')
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('warm up the operator plan before CUDA Graph capture')
            self.stream = torch.cuda.Stream(device=self.device)

    def run(self, q, k, v, lengths, selector, *, sink, swa, scale, layout=None):
        from .token_selection import TokenSelection
        # Protected loads need no selector result. Empty IDs only satisfy the
        # shared kernel ABI; protected token positions are generated in-kernel.
        empty = TokenSelection(torch.empty((q.shape[0], k.shape[2], 0), device=q.device, dtype=torch.int32),
                               torch.zeros((q.shape[0], k.shape[2]), device=q.device, dtype=torch.int32))
        options = dict(sink=sink, swa=swa, scale=scale, backend=self.backend, layout=layout)
        current = torch.cuda.current_stream(q.device) if q.is_cuda else None
        if self.stream is None:
            protected = partial_attention(q, k, v, empty, lengths, protected=True,
                                          splits=self.protected_splits, **options)
        else:
            # Fork AFTER the caller's current KV update and Q/RoPE work.
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                for t in (q, k, v, lengths, *empty):
                    t.record_stream(self.stream)
                protected = partial_attention(q, k, v, empty, lengths, protected=True,
                                              splits=self.protected_splits, **options)
        try:
            selection = selector()  # index maintenance, L1, L2, exact K2 barrier
            selected = partial_attention(q, k, v, selection, lengths,
                                         splits=self.selected_splits, **options)
        finally:
            # Also join on exceptions; never leave hidden work reading inputs.
            if self.stream is not None:
                current.wait_stream(self.stream)
                for t in protected:
                    t.record_stream(current)
        return merge_partials(protected, selected, q.dtype, backend=self.backend)


def plan_for(owner, q):
    mode = operator_mode()
    if mode == 'off':
        raise ValueError('operator pipeline is disabled')
    backend = os.environ.get('TLI_OPERATOR_BACKEND', 'simt')
    if backend not in ('torch', 'simt', 'tc'):
        raise ValueError('TLI_OPERATOR_BACKEND must be torch, simt or tc')
    splits = int(os.environ.get('TLI_OPERATOR_SPLITS', '4'))
    protected_splits = int(os.environ.get('TLI_PROTECTED_SPLITS', '1'))
    if not 1 <= splits <= 64 or not 1 <= protected_splits <= 64:
        raise ValueError('operator split counts must be in [1,64]')
    key = (q.device, mode, backend, splits, protected_splits)
    existing = getattr(owner, '_operator_plan', None)
    if existing is None or existing[0] != key:
        plan = OperatorPlan(q.device, mode, backend, splits, protected_splits)
        owner._operator_plan = (key, plan)
    return owner._operator_plan[1]


def decode_indexed(owner, indexer, q, k, v, cu, lengths, scale):
    """One row of the existing supported fixed-geometry graph selector."""
    if q.shape[0] != 1:
        raise ValueError('indexer adapter requires one request; runner dispatches rows')
    def select():
        index = indexer.prepare_index(k, cu)
        return indexer.compute_selection_graph(q, index, scale, lengths)
    return plan_for(owner, q).run(q, k, v, lengths, select,
                                 sink=indexer.sink_tokens, swa=indexer.sliding_window_size,
                                 scale=scale)


def decode_ring(workspace, q, k, v, length, scale):
    cfg = workspace.cfg
    def select():
        positions, valid = workspace.select(q, k, length, scale)
        # L2 normalization still included the original protected candidates.
        # Only the attention consumption is split into disjoint sets here.
        return middle_selection(positions[None], valid[None], length,
                                cfg.sink_tokens, cfg.window_tokens,
                                torch.iinfo(torch.int32).max)
    return plan_for(workspace, q).run(q, k, v, length, select,
                                     sink=cfg.sink_tokens, swa=cfg.window_tokens,
                                     scale=scale, layout=workspace.layout)
