"""CUDA Graph capture/replay for fixed batch-size buckets."""
import torch


class GraphRunner:
    """Replay only with the model/cache objects used to capture that bucket.

    Tensor storage and model execution mode must stay fixed within each owner.
    In-place request admission/EOS updates do not change cache ownership;
    replacing storage or changing model configuration requires recapture.
    """

    def __init__(self, *, scheduler, buckets=(1, 2, 4, 8), device='cuda'):
        self.scheduler = scheduler
        self.graphs = {bucket: None for bucket in buckets}
        self.static_logits = {}
        self._capture_owners = {}
        self.device = torch.device(device)
        self.input_ids_buf = {}
        self.position_ids_buf = {}
        for bucket in buckets:
            self.input_ids_buf[bucket] = torch.empty(
                (bucket, 1), dtype=torch.long, device=self.device)
            self.position_ids_buf[bucket] = torch.empty(
                (bucket, 1), dtype=torch.long, device=self.device)

    def bucket_for(self, batch):
        bucket = 1
        while bucket < batch:
            bucket <<= 1
        return bucket

    def _prepare_metadata(self, cache, padded_ids):
        cache.new_step(padded_ids)
        for layer_id in range(cache.n_layers):
            cache.prepare_metadata(padded_ids, layer_id)

    def _copy_inputs(self, bucket, input_ids, position_ids):
        self.input_ids_buf[bucket].copy_(input_ids.to(self.device), non_blocking=True)
        self.position_ids_buf[bucket].copy_(
            position_ids.to(self.device), non_blocking=True)

    def _validate_cache(self, cache):
        if cache is not self.scheduler.cache:
            raise ValueError('runner cache must be scheduler.cache')

    def capture(self, model, cache, bucket):
        self._validate_cache(cache)
        if bucket not in self.input_ids_buf:
            raise ValueError(f'unsupported bucket {bucket}')
        # A failed recapture must not leave an old graph available for replay.
        self.graphs[bucket] = None
        self.static_logits.pop(bucket, None)
        self._capture_owners.pop(bucket, None)
        padded_ids = [-1] * bucket
        self._prepare_metadata(cache, padded_ids)
        self.input_ids_buf[bucket].zero_()
        self.position_ids_buf[bucket].zero_()

        model(
            input_ids=self.input_ids_buf[bucket],
            position_ids=self.position_ids_buf[bucket],
            past_key_values=cache,
            use_cache=True,
            attention_mask={'full_attention': None},
        )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_logits = model(
                input_ids=self.input_ids_buf[bucket],
                position_ids=self.position_ids_buf[bucket],
                past_key_values=cache,
                use_cache=True,
                attention_mask={'full_attention': None},
            )
        self.graphs[bucket] = graph
        self.static_logits[bucket] = static_logits
        # Keep captured storage owners alive for as long as this graph.
        self._capture_owners[bucket] = (model, cache)
        return graph

    def capture_all(self, model, cache):
        for bucket in sorted(self.graphs):
            self.capture(model, cache, bucket)
        return self.graphs

    def step(self, model, cache):
        self._validate_cache(cache)
        # Check before prepare_step mutates cache metadata or copies any inputs.
        bucket = self.bucket_for(len(self.scheduler.active_req_ids))
        graph = self.graphs.get(bucket)
        if graph is not None:
            captured_model, captured_cache = self._capture_owners[bucket]
            if model is not captured_model or cache is not captured_cache:
                raise ValueError(
                    'captured model/cache changed; recapture this bucket before replay')
        req_ids, input_ids, position_ids = self.scheduler.prepare_step()
        bucket = self.bucket_for(len(req_ids))
        padded_ids = req_ids + [-1] * (bucket - len(req_ids))
        self._prepare_metadata(cache, padded_ids)
        self._copy_inputs(bucket, input_ids, position_ids)

        graph = self.graphs.get(bucket)
        if graph is not None:
            graph.replay()
            logits = self.static_logits[bucket]
        else:
            logits = model(
                input_ids=self.input_ids_buf[bucket],
                position_ids=self.position_ids_buf[bucket],
                past_key_values=cache,
                use_cache=True,
                attention_mask={'full_attention': None},
            )
        cache.commit_step()
        self.scheduler.record_next_tokens(padded_ids, logits)
        return logits

