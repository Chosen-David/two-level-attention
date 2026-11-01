"""Graph-ready continuous-batching scheduler.

Keeps ``active_req_ids`` and pads each decode step to the next power-of-two
batch bucket.  Every per-step tensor is preallocated; the scheduler only
rewrites contents, never allocates tensors on the hot path.
"""
import torch


class ContinuousBatchScheduler:
    def __init__(self, *, cache, max_batch, eos_ids, pad_token_id):
        self.cache = cache
        self.max_batch = max_batch
        self.eos_ids = set(eos_ids)
        self.pad_token_id = pad_token_id
        self.active_req_ids = []
        self.next_tokens = {}

        self.req_ids_buf = torch.full(
            (max_batch,), -1, dtype=torch.int64, device='cpu')
        self.flat_input_ids_buf = torch.empty(
            (max_batch, 1), dtype=torch.long, device='cpu')
        self.flat_position_ids_buf = torch.empty(
            (max_batch, 1), dtype=torch.long, device='cpu')

    def add_request(self, request_id):
        if request_id not in self.active_req_ids:
            self.active_req_ids.append(request_id)

    def finish_request(self, request_id):
        self.cache.remove_request(request_id)
        self.active_req_ids.remove(request_id)
        self.next_tokens.pop(request_id, None)

    def _pad_to_bucket(self, req_ids):
        n = len(req_ids)
        bucket = 1
        while bucket < n:
            bucket <<= 1
        bucket = min(bucket, self.max_batch)
        return req_ids + [-1] * (bucket - n)

    def prepare_step(self):
        padded = self._pad_to_bucket(self.active_req_ids)
        batch = len(padded)

        self.req_ids_buf.zero_()
        self.req_ids_buf[:batch] = torch.tensor(padded, dtype=torch.int64)
        self.flat_input_ids_buf.fill_(self.pad_token_id)
        self.flat_position_ids_buf.zero_()

        for b, req_id in enumerate(padded):
            if req_id < 0:
                continue
            state = self.cache.states[req_id]
            self.flat_input_ids_buf[b, 0] = self.next_tokens.get(req_id, self.pad_token_id)
            self.flat_position_ids_buf[b, 0] = state.logical_length

        self.cache.new_step(padded)
        return padded, self.flat_input_ids_buf[:batch], self.flat_position_ids_buf[:batch]

    def record_next_tokens(self, req_ids, logits):
        logits = getattr(logits, 'logits', logits)
        next_tokens = logits[:, -1].argmax(dim=-1).tolist()
        for req_id, token in zip(req_ids, next_tokens):
            if req_id < 0:
                continue
            self.next_tokens[req_id] = token
            if token in self.eos_ids:
                self.finish_request(req_id)
