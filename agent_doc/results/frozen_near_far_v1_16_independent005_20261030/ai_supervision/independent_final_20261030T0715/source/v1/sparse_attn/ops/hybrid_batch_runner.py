"""Fixed-address eager/CUDA Graph driver for one immutable hybrid batch.

Python controls EOS and validates append-only lengths before launching a step;
all changing model inputs are device tensors. A new request batch always gets
a new runner, workspaces and graph (including when its shape matches the old).
"""
from contextlib import contextmanager

import torch

from ..baseline_runtime import BatchCtx, batch_decode_scope
from .cuda_graph_state import graph_state


class HybridBatchRunner:
    def __init__(self, model, cache, pad_id=0):
        if not getattr(cache, '_hybrid_batch', False):
            raise ValueError('finish a HybridBatchCacheBuilder before creating its runner')
        if model.training or torch.is_grad_enabled():
            raise ValueError('hybrid runner requires eval and no_grad')
        self.model, self.cache = model, cache
        self.batch, self.capacity = cache.batch_size, cache.max_cache_len
        parameter = next(model.parameters())
        device, dtype = parameter.device, parameter.dtype
        self.ctx = BatchCtx(batch=self.batch, seq_lens=list(cache._hybrid_lengths),
                            gen_lens=[0]*self.batch, alive=[True]*self.batch, pad_id=pad_id)
        self.tokens = torch.full((self.batch, 1), pad_id, dtype=torch.long, device=device)
        self.pos = torch.zeros(self.batch, dtype=torch.long, device=device)
        self.lengths = torch.zeros_like(self.pos)
        self.alive = torch.ones(self.batch, dtype=torch.bool, device=device)
        self.mask = torch.zeros(self.batch, 1, 1, self.capacity, dtype=dtype, device=device)
        self.positions = torch.arange(self.capacity, device=device)
        cache._far_skip_attention_mask = self.mask
        self.graph = self.output = None
        self.injected = False

    @contextmanager
    def scope(self):
        gs = graph_state()
        saved = gs.enabled, gs.bs, gs.pos_buf, gs.valid_len_buf, gs.alive_buf
        gs.enabled, gs.bs = True, self.batch
        gs.pos_buf, gs.valid_len_buf, gs.alive_buf = self.pos, self.lengths, self.alive
        try:
            with batch_decode_scope(self.ctx):
                yield
        finally:
            gs.enabled, gs.bs, gs.pos_buf, gs.valid_len_buf, gs.alive_buf = saved

    def inject(self, tokens, lengths, alive=None):
        alive = list(self.ctx.alive if alive is None else alive)
        lengths = list(lengths)
        if len(tokens) != self.batch or len(lengths) != self.batch or len(alive) != self.batch:
            raise ValueError('tokens, lengths and alive must match the captured batch')
        if any(type(active) is not bool or type(length) is not int
               or not 1 <= length <= self.capacity
               or length != old + int(active)
               or (active and not previous)
               for length, old, active, previous in
               zip(lengths, self.ctx.seq_lens, alive, self.ctx.alive)):
            raise ValueError('append exactly one token per live row; finished rows stay frozen')
        device = self.tokens.device
        self.tokens.copy_(torch.as_tensor(tokens, dtype=torch.long, device=device).view(self.batch, 1))
        self.lengths.copy_(torch.tensor(lengths, dtype=torch.long, device=device))
        self.pos.copy_(self.lengths-1)
        self.alive.copy_(torch.tensor(alive, dtype=torch.bool, device=device))
        self.mask.zero_().masked_fill_(
            self.positions.view(1, 1, 1, -1) >= self.lengths.view(-1, 1, 1, 1),
            torch.finfo(self.mask.dtype).min)
        self.ctx.seq_lens[:] = lengths
        self.ctx.gen_lens[:] = [n+int(a) for n, a in zip(self.ctx.gen_lens, alive)]
        self.ctx.alive[:] = alive
        self.injected = True

    def forward(self):
        if not self.injected:
            raise ValueError('inject the first decode step before forward/capture')
        with self.scope():
            return self.model(input_ids=self.tokens, position_ids=self.pos[:, None],
                              cache_position=self.pos[:1], attention_mask=self.mask,
                              past_key_values=self.cache, use_cache=True)

    def capture(self):
        if self.tokens.device.type != 'cuda':
            raise ValueError('CUDA Graph capture requires a CUDA device')
        if self.graph is not None:
            raise ValueError('this batch is already captured')
        stream = torch.cuda.Stream(device=self.tokens.device)
        stream.wait_stream(torch.cuda.current_stream(self.tokens.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.forward()
        torch.cuda.current_stream(self.tokens.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        from sparse_attn.ops.cuda_graph_state import capture_graph
        with capture_graph(graph):
            output = self.forward()
        self.graph, self.output = graph, output

    def run(self):
        if self.graph is None:
            return self.forward()
        self.graph.replay()
        return self.output

    def kv_storage_bytes(self):
        return sum(t.numel()*t.element_size()
                   for layer in self.cache.layers
                   for row in getattr(layer, 'rows', [layer])
                   for t in (row.keys, row.values))
