"""CUDA Graph capture/replay for batch-size and visible-length buckets."""
import torch


class GraphRunner:
    def __init__(
        self,
        *,
        scheduler,
        buckets=(1, 2, 4, 8, 16, 32, 64),
        visible_buckets=(512, 1024, 2048, 4096),
        device='cuda',
    ):
        self.scheduler = scheduler
        self.buckets = tuple(buckets)
        self.visible_buckets = tuple(visible_buckets)
        self.graphs = {}
        self.static_logits = {}
        self.device = torch.device(device)
        self.input_ids_buf = {}
        self.position_ids_buf = {}
        for bucket in buckets:
            self.input_ids_buf[bucket] = torch.empty(
                (bucket, 1), dtype=torch.long, device=self.device)
            self.position_ids_buf[bucket] = torch.empty(
                (bucket, 1), dtype=torch.long, device=self.device)

    @staticmethod
    def _key(bucket, visible_bucket):
        return (int(bucket), int(visible_bucket))

    def bucket_for(self, batch, max_requests=None):
        for bucket in self.buckets:
            if max_requests is not None and bucket > max_requests:
                continue
            if bucket >= batch:
                return bucket
        if max_requests is not None:
            if batch > max_requests:
                raise ValueError('active batch exceeds fixed cache capacity')
            return max_requests
        return max(self.buckets)

    def visible_bucket_for(self, cache, padded_ids):
        width = cache.choose_visible_bucket(padded_ids, self.visible_buckets)
        return min(width, cache.max_visible_len)

    def _prepare_metadata(self, cache, padded_ids,
                          selected_ids_by_layer=None, visible_bucket=None):
        cache.prepare_all_layers(
            padded_ids,
            selected_ids_by_layer=selected_ids_by_layer,
            visible_bucket=visible_bucket)

    def _copy_inputs(self, bucket, input_ids, position_ids):
        self.input_ids_buf[bucket].copy_(input_ids, non_blocking=True)
        self.position_ids_buf[bucket].copy_(
            position_ids, non_blocking=True)

    def capture(self, model, cache, bucket, visible_bucket=None):
        if visible_bucket is None:
            visible_bucket = cache.max_visible_len
        if bucket not in self.input_ids_buf:
            raise ValueError(f'unsupported batch bucket {bucket}')
        if visible_bucket > cache.max_visible_len:
            raise ValueError(
                f'visible bucket {visible_bucket} exceeds cache {cache.max_visible_len}')

        key = self._key(bucket, visible_bucket)
        padded_ids = [-1] * bucket
        cache.ensure_visible_buckets((visible_bucket,))
        self._prepare_metadata(
            cache, padded_ids, visible_bucket=visible_bucket)
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
        self.graphs[key] = graph
        self.static_logits[key] = static_logits
        return graph

    def capture_all(self, model, cache):
        if not all(state.frozen for state in cache.states.values()):
            raise RuntimeError(
                'all active requests must be frozen before CUDA Graph capture')
        valid_visible_buckets = tuple(
            width for width in self.visible_buckets
            if width <= cache.max_visible_len)
        cache.ensure_visible_buckets(valid_visible_buckets)
        for bucket in self.buckets:
            if bucket > cache.max_requests:
                continue
            for visible_bucket in valid_visible_buckets:
                self.capture(model, cache, bucket, visible_bucket)
            # Always include the full-capacity graph for batches whose
            # visible length exceeds the largest fine-grained bucket.
            if cache.max_visible_len not in valid_visible_buckets:
                self.capture(model, cache, bucket, cache.max_visible_len)
        return self.graphs

    def step(self, model, cache):
        req_ids, input_ids, position_ids = self.scheduler.prepare_step()
        bucket = self.bucket_for(len(req_ids), cache.max_requests)
        padded_ids = req_ids + [-1] * (bucket - len(req_ids))

        # Plan common metadata first so Top-K selection sees this step's
        # planned decode length rather than the previous committed length.
        cache.new_step(padded_ids)
        cache.plan_common(padded_ids)

        selected_ids_by_layer = cache.compute_selected_ids_by_layer(
            model, padded_ids)
        visible_bucket = self.visible_bucket_for(cache, padded_ids)
        self._prepare_metadata(
            cache, padded_ids,
            selected_ids_by_layer=selected_ids_by_layer,
            visible_bucket=visible_bucket)
        self._copy_inputs(bucket, input_ids, position_ids)

        key = self._key(bucket, visible_bucket)
        graph = self.graphs.get(key)
        if graph is not None:
            graph.replay()
            logits = self.static_logits[key]
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
