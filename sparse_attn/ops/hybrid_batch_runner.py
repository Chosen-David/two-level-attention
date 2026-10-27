"""Fixed-address eager/CUDA Graph driver for one immutable hybrid batch.

The manual inject/run API validates append-only lengths on the host. generate()
keeps greedy tokens, EOS and mask updates on the device between polling points.
A new request batch always gets
a new runner, workspaces and graph (including when its shape matches the old).
"""
from contextlib import contextmanager

import torch
from sparse_attn.ops.decode_geometry import decode_geometry_scope

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
        self.generation_graph = None
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
        with self.scope(), decode_geometry_scope():
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
        with decode_geometry_scope(), torch.cuda.graph(graph):
            output = self.forward()
        self.graph, self.output = graph, output

    def run(self):
        if self.graph is None:
            return self.forward()
        self.graph.replay()
        return self.output

    def generate(self, first_tokens, max_new_tokens, eos_ids=(), *, use_graph=False,
                 poll_interval=16):
        """Greedy continuation, including the prefill's first token in the result.

        Owns this runner for one generation. Capture includes argmax, EOS and
        next-step bookkeeping; replay never copies tokens through the CPU.
        As in the RULER driver, EOS is honored from the second generated token.
        """
        if self.injected:
            raise ValueError('generate requires a fresh runner')
        if (type(max_new_tokens) is not int or max_new_tokens < 1
                or type(poll_interval) is not int or poll_interval < 1):
            raise ValueError('generation length and poll interval must be positive integers')
        if max(self.ctx.seq_lens)+max_new_tokens-1 > self.capacity:
            raise ValueError('generation exceeds cache capacity')
        if use_graph and self.tokens.device.type != 'cuda':
            raise ValueError('CUDA Graph capture requires a CUDA device')
        first = torch.as_tensor(first_tokens, device=self.tokens.device, dtype=torch.long)
        if first.shape != (self.batch,):
            raise ValueError('one first token is required per batch row')
        if max_new_tokens == 1:
            return first[:, None].tolist()
        self.inject(first, [length+1 for length in self.ctx.seq_lens])
        generated = torch.full((self.batch, max_new_tokens), self.ctx.pad_id,
                               dtype=torch.long, device=self.tokens.device)
        generated[:, 0].copy_(first)
        counts = torch.ones_like(self.lengths)
        step = torch.ones(1, dtype=torch.long, device=self.tokens.device)
        eos = torch.tensor(tuple(eos_ids), dtype=torch.long, device=self.tokens.device)

        def advance():
            predicted = self.forward().logits[:, -1].argmax(-1)
            generated.index_copy_(1, step, torch.where(
                self.alive, predicted, self.ctx.pad_id)[:, None])
            counts.add_(self.alive)
            continuing = self.alive & ~(predicted[:, None] == eos).any(-1)
            # No next input is needed after the last output. In particular,
            # never scatter a mask column beyond an exactly filled cache.
            append = continuing & (step[0]+1 < max_new_tokens)
            self.tokens.copy_(torch.where(continuing, predicted, self.ctx.pad_id)[:, None])
            self.lengths.add_(append)
            self.pos.copy_(self.lengths-1)
            self.mask.scatter_(-1, self.pos[:, None, None, None], 0)
            self.alive.copy_(continuing)
            step.add_(1)

        if use_graph:
            stream = torch.cuda.Stream(device=self.tokens.device)
            stream.wait_stream(torch.cuda.current_stream(self.tokens.device))
            with torch.cuda.stream(stream):
                # Warm up at the same input position: never advance ring KV
                # into future slots and then rewind only the scalar lengths.
                for _ in range(2):
                    self.forward()
            torch.cuda.current_stream(self.tokens.device).wait_stream(stream)
            inputs = (self.tokens, self.lengths, self.pos, self.alive, self.mask)
            saved = [value.clone() for value in inputs]
            graph = torch.cuda.CUDAGraph()
            with decode_geometry_scope(), torch.cuda.graph(graph):
                advance()
            for value, initial in zip(inputs, saved):
                value.copy_(initial)
            generated.fill_(self.ctx.pad_id)
            generated[:, 0].copy_(first)
            counts.fill_(1)
            step.fill_(1)
            self.generation_graph = graph
        for n in range(max_new_tokens-1):
            if self.generation_graph is None:
                advance()
            else:
                self.generation_graph.replay()
            if (n+1) % poll_interval == 0 and not bool(self.alive.any()):
                break
        # Keep tensors referenced for captured-address lifetime, even after
        # returning Python results. No host conversion occurs per token.
        self._generation_buffers = (generated, counts, step, eos)
        result, sizes = generated.tolist(), counts.tolist()
        self.ctx.seq_lens[:] = self.lengths.tolist()
        self.ctx.gen_lens[:] = [length-prompt for length, prompt in
                               zip(self.ctx.seq_lens, self.cache._hybrid_lengths)]
        self.ctx.alive[:] = self.alive.tolist()
        return [row[:size] for row, size in zip(result, sizes)]

    def kv_storage_bytes(self):
        return sum(t.numel()*t.element_size()
                   for layer in self.cache.layers
                   for row in getattr(layer, 'rows', [layer])
                   for t in (row.keys, row.values))
