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

    def _prepare_metadata(self, cache, padded_ids, visible_bucket):
        """只准备公共页表/长度；各层 gather 由当前 Q 在 attention 内产生。"""
        cache.plan_common(padded_ids)
        cache.set_active_visible_bucket(visible_bucket)

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
        if key in self.graphs:
            # 图只依赖固定 bucket 与设备 buffer 地址；真实 request/length 均由
            # replay 前的 metadata 刷新，重复 capture 只会增加私有池与 JIT 开销。
            return self.graphs[key]
        padded_ids = [-1] * bucket
        cache.ensure_visible_buckets((visible_bucket,))
        cache.new_step(padded_ids)
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

    def step(self, model, cache, *, generated_counts=0, min_new_tokens=0,
             return_next_tokens=False, forced_tokens=None, stop_on_eos=True):
        req_ids, input_ids, position_ids = self.scheduler.prepare_step()
        bucket = self.bucket_for(len(req_ids), cache.max_requests)
        padded_ids = req_ids + [-1] * (bucket - len(req_ids))

        # _prepare_metadata 统一准备本步页表/长度，避免同一 step 重复
        # plan_common；selector 看到的就是本步 planned decode length。
        cache.new_step(padded_ids)

        # selector 不能放在整步 forward 前：各层必须在 post-RoPE 当前 Q 上
        # 独立选择。这里只刷新固定元数据和本图使用的 visible bucket。
        visible_bucket = self.visible_bucket_for(cache, padded_ids)
        self._prepare_metadata(cache, padded_ids, visible_bucket)
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
        # 第 N 步 GPU 工作已提交；此时处理第 N-1 步已就绪的 CPU 回执，
        # 不让上一帧 D2H 等待阻塞本帧 replay，也不依赖其完成来组装设备输入。
        self.scheduler._drain_pending_next_tokens()
        # 回执只在 step 边界记录一次；具体设备数据延迟到请求结束时读取，
        # 不在各层 forward 中更新 Python 对象或触发 D2H。
        cache.nfa_decode_steps += 1
        # 所有层均完成当前 Q 选择后，才统一消费 first-decode seed。
        from ..indexer.nfa_layer_batch_selector import mark_first_decode_seeded
        mark_first_decode_seeded(cache, padded_ids)
        cache.commit_step()
        next_tokens = self.scheduler.record_next_tokens(
            padded_ids, logits, generated_counts=generated_counts,
            min_new_tokens=min_new_tokens, forced_tokens=forced_tokens,
            stop_on_eos=stop_on_eos)
        if return_next_tokens:
            return logits, next_tokens
        return logits
