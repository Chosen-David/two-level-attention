"""Graph-ready continuous-batching scheduler.

The CUDA path keeps next-token, input-id and position state on device.  Argmax
never leaves the GPU on the decode hot path; the previous step's tokens are
copied back through a dedicated stream and drained at the start of the next
step.
"""
import torch


def greedy_next_tokens(logits, eos_ids, *, generated_counts=0, min_new_tokens=0):
    """在 argmax 前屏蔽未达最小生成长度请求的 EOS。

    首 token 与 GraphRunner 解码共用此函数，避免 Scheduler 消费的 token
    与请求边界返回的 token 不一致。``generated_counts`` 是候选 token 之前
    已生成的数量，可按 batch row 传入整数、列表或 tensor。
    """
    logits = getattr(logits, 'logits', logits)
    if logits.ndim != 3:
        raise ValueError('greedy decoding expects [B,T,V] logits')
    if min_new_tokens < 0:
        raise ValueError('min_new_tokens must be non-negative')

    next_logits = logits[:, -1]
    batch = next_logits.shape[0]
    counts = torch.as_tensor(
        generated_counts, dtype=torch.long, device=next_logits.device)
    if counts.ndim == 0:
        counts = counts.expand(batch)
    if counts.ndim != 1 or counts.numel() != batch:
        raise ValueError('generated_counts must be scalar or [B]')

    suppress = counts < min_new_tokens
    eos_ids = tuple(int(token) for token in eos_ids)
    if eos_ids and min_new_tokens > 0:
        # 仅复制最后一位置 logits，避免改写 GraphRunner 持有的静态输出。
        next_logits = next_logits.clone()
        eos = torch.tensor(
            eos_ids, dtype=torch.long, device=next_logits.device)
        eos_logits = next_logits[:, eos]
        next_logits[:, eos] = torch.where(
            suppress[:, None], next_logits.new_full((), -torch.inf),
            eos_logits)
    return next_logits.argmax(dim=-1).to(torch.int64)


class ContinuousBatchScheduler:
    def __init__(self, *, cache, max_batch, eos_ids, pad_token_id):
        self.cache = cache
        self.max_batch = max_batch
        self.eos_ids = set(eos_ids)
        self.pad_token_id = pad_token_id
        self.active_req_ids = []
        self.next_tokens = {}

        self.device = self.cache.layers[0].kv_pool_k.device
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('ContinuousBatchScheduler requires CUDA')
        self._step_counter = 0
        self._seen_active = set()

        self._cpu_req_staging = torch.full(
            (max_batch,), -1, dtype=torch.int64).pin_memory()
        self._device_req_ids_buf = torch.full(
            (max_batch,), -1, dtype=torch.int64, device=self.device)
        self._device_input_ids_buf = torch.full(
            (max_batch, 1), pad_token_id, dtype=torch.long,
            device=self.device)
        self._device_positions_buf = torch.zeros(
            (max_batch, 1), dtype=torch.long, device=self.device)
        self._device_alive_buf = torch.zeros(
            (max_batch,), dtype=torch.bool, device=self.device)
        self._device_req_input_ids = torch.full(
            (max_batch,), pad_token_id, dtype=torch.long,
            device=self.device)
        self._device_req_positions = torch.zeros(
            (max_batch,), dtype=torch.long, device=self.device)
        self._device_req_alive = torch.zeros(
            (max_batch,), dtype=torch.bool, device=self.device)
        self._zero_buf = torch.zeros(
            (max_batch,), dtype=torch.long, device=self.device)

        self._eos_tensor = torch.tensor(
            list(self.eos_ids), dtype=torch.int64, device=self.device)
        self._eos_cpu_tensor = torch.tensor(
            list(self.eos_ids), dtype=torch.int64)

        self._token_dev = [
            torch.zeros((max_batch,), dtype=torch.int64, device=self.device)
            for _ in range(2)
        ]
        self._token_cpu = [
            torch.empty((max_batch,), dtype=torch.int64).pin_memory()
            for _ in range(2)
        ]
        self._copy_events = [torch.cuda.Event() for _ in range(2)]
        for event in self._copy_events:
            event.record()
        self._copy_stream = torch.cuda.Stream()
        self._pending_copy_idx = None
        self._pending_batch = 0
        self._pending_stop_on_eos = True

    def add_request(self, request_id):
        if request_id not in self.active_req_ids:
            self.active_req_ids.append(request_id)
            self._seen_active.discard(request_id)

    def finish_request(self, request_id):
        self.cache.transition_request(request_id, 'FINISHED')
        self.cache.remove_request(request_id)
        self.active_req_ids.remove(request_id)
        self.next_tokens.pop(request_id, None)
        self._seen_active.discard(request_id)

    def _pad_to_bucket(self, req_ids):
        n = len(req_ids)
        bucket = 1
        while bucket < n:
            bucket <<= 1
        bucket = min(bucket, self.max_batch)
        return req_ids + [-1] * (bucket - n)

    def _drain_pending_next_tokens(self):
        if self._pending_copy_idx is None:
            return

        idx = self._pending_copy_idx
        self._copy_events[idx].synchronize()

        batch = self._pending_batch
        tokens = self._token_cpu[idx][:batch]
        req_ids = self._cpu_req_staging[:batch]
        valid = req_ids >= 0

        if valid.any():
            ids = req_ids[valid].tolist()
            toks = tokens[valid].tolist()
            self.next_tokens.update(zip(ids, toks))

        eos = valid & torch.isin(tokens, self._eos_cpu_tensor)
        if self._pending_stop_on_eos:
            finished_ids = req_ids[eos].tolist()
            for request_id in finished_ids:
                self.finish_request(request_id)

        self._pending_copy_idx = None
        self._pending_batch = 0
        self._pending_stop_on_eos = True

    def _seed_new_requests(self, padded):
        if not padded:
            return

        request_ids = []
        tokens = []
        positions = []
        for slot, request_id in enumerate(padded):
            if request_id < 0 or request_id in self._seen_active:
                continue
            state = self.cache.states[request_id]
            request_ids.append(request_id)
            tokens.append(self.next_tokens.get(request_id, self.pad_token_id))
            positions.append(state.logical_length)
            self._seen_active.add(request_id)

        if not request_ids:
            return

        req_idx = torch.tensor(request_ids, dtype=torch.long, device=self.device)
        self._device_req_input_ids[req_idx] = torch.tensor(
            tokens, dtype=torch.long, device=self.device)
        self._device_req_positions[req_idx] = torch.tensor(
            positions, dtype=torch.long, device=self.device)
        self._device_req_alive[req_idx] = True

    def prepare_step(self):
        self._drain_pending_next_tokens()
        padded = self._pad_to_bucket(self.active_req_ids)
        batch = len(padded)

        self._cpu_req_staging.numpy()[:batch] = padded
        self._device_req_ids_buf[:batch].copy_(
            self._cpu_req_staging[:batch], non_blocking=True)
        self._device_req_ids_buf[batch:].fill_(-1)

        self._seed_new_requests(padded)

        valid = self._device_req_ids_buf[:batch] >= 0
        safe_req = torch.where(
            valid, self._device_req_ids_buf[:batch],
            self._zero_buf[:batch])

        self._device_input_ids_buf[:batch, 0].copy_(
            self._device_req_input_ids[safe_req], non_blocking=True)
        self._device_positions_buf[:batch, 0].copy_(
            self._device_req_positions[safe_req], non_blocking=True)
        self._device_alive_buf[:batch].copy_(
            self._device_req_alive[safe_req] & valid, non_blocking=True)

        self._device_input_ids_buf[:batch, 0].masked_fill_(
            ~valid, self.pad_token_id)
        self._device_positions_buf[:batch, 0].masked_fill_(~valid, 0)

        self.cache.new_step(padded)
        return (
            padded,
            self._device_input_ids_buf[:batch],
            self._device_positions_buf[:batch],
        )

    def record_next_tokens(self, req_ids, logits, *, generated_counts=0,
                           min_new_tokens=0, forced_tokens=None,
                           stop_on_eos=True):
        logits = getattr(logits, 'logits', logits)

        if logits.device.type != 'cuda':
            raise RuntimeError('ContinuousBatchScheduler requires CUDA logits')

        batch = len(req_ids)
        if forced_tokens is None:
            next_tokens = greedy_next_tokens(
                logits, self.eos_ids, generated_counts=generated_counts,
                min_new_tokens=min_new_tokens)
        else:
            if len(forced_tokens) != batch:
                raise ValueError('forced token count must match the padded batch')
            next_tokens = torch.as_tensor(
                forced_tokens, dtype=torch.long, device=logits.device)
        req_ids_tensor = self._device_req_ids_buf[:batch]
        valid = req_ids_tensor >= 0
        eos_hit = (valid & torch.isin(next_tokens, self._eos_tensor)
                   if stop_on_eos else torch.zeros_like(valid))
        alive_by_slot = valid & ~eos_hit

        valid_req_ids = req_ids_tensor[valid].to(torch.long)
        valid_next_tokens = next_tokens[valid]
        valid_alive = alive_by_slot[valid]

        self._device_req_alive[valid_req_ids] = valid_alive
        self._device_req_input_ids[valid_req_ids] = torch.where(
            valid_alive, valid_next_tokens, self.pad_token_id)
        self._device_req_positions[valid_req_ids] += (
            valid_alive.to(torch.long))

        idx = self._step_counter & 1
        self._step_counter += 1

        torch.cuda.current_stream().wait_event(self._copy_events[idx])
        device_buf = self._token_dev[idx]
        device_buf[:batch].copy_(next_tokens, non_blocking=True)

        self._copy_stream.wait_stream(torch.cuda.current_stream())
        self._token_cpu[idx][:batch].copy_(
            device_buf[:batch], non_blocking=True)
        self._copy_events[idx].record(self._copy_stream)

        self._pending_copy_idx = idx
        self._pending_batch = batch
        self._pending_stop_on_eos = stop_on_eos
        return next_tokens
