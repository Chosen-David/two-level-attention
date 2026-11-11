"""CUDA Graph capture/replay for batch-size and visible-length buckets."""
import weakref

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
        self.capture_count = 0
        self.replay_count = 0
        self.eager_fallback_count = 0
        for bucket in buckets:
            self.input_ids_buf[bucket] = torch.empty(
                (bucket, 1), dtype=torch.long, device=self.device)
            self.position_ids_buf[bucket] = torch.empty(
                (bucket, 1), dtype=torch.long, device=self.device)

    @staticmethod
    def _key(bucket, visible_bucket, skip_signature=()):
        return (int(bucket), int(visible_bucket), skip_signature)

    def _capture_request_ids(self, cache, bucket):
        active = list(getattr(self.scheduler, 'active_req_ids', ()) or ())
        if not active:
            active = [rid for rid in getattr(cache, 'active_req_ids', ())
                      if rid is not None and rid >= 0 and rid in cache.states]
        if len(active) > bucket:
            raise ValueError('capture bucket is smaller than the frozen batch')
        return active + [-1] * (bucket - len(active))

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

    @staticmethod
    def _attention_mask_arg(model):
        if getattr(getattr(model, 'config', None), 'model_type', None) == 'qwen3':
            return {'full_attention': None}
        return None

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

        capture_req_ids = self._capture_request_ids(cache, bucket)
        skip_signature = cache.nfa_skip_signature(capture_req_ids)
        key = self._key(bucket, visible_bucket, skip_signature)
        if key in self.graphs:
            return self.graphs[key]
        # metadata 仍使用全 padding，避免 warmup 改写真实请求长度；selector
        # 通过临时 override 看到实际冻结签名，从而 capture 正确的区域特化。
        padded_ids = [-1] * bucket
        cache.ensure_visible_buckets((visible_bucket,))
        cache.new_step(padded_ids)
        self._prepare_metadata(
            cache, padded_ids, visible_bucket=visible_bucket)
        self.input_ids_buf[bucket].zero_()
        self.position_ids_buf[bucket].zero_()

        attention_mask = self._attention_mask_arg(model)
        missing = object()
        previous = getattr(cache, '_nfa_capture_skip_signature', missing)
        cache._nfa_capture_skip_signature = skip_signature
        try:
            model(
                input_ids=self.input_ids_buf[bucket],
                position_ids=self.position_ids_buf[bucket],
                past_key_values=cache,
                use_cache=True,
                attention_mask=attention_mask,
            )
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_logits = model(
                    input_ids=self.input_ids_buf[bucket],
                    position_ids=self.position_ids_buf[bucket],
                    past_key_values=cache,
                    use_cache=True,
                    attention_mask=attention_mask,
                )
        finally:
            if previous is missing:
                del cache._nfa_capture_skip_signature
            else:
                cache._nfa_capture_skip_signature = previous
        self.graphs[key] = graph
        self.static_logits[key] = static_logits
        self.capture_count += 1
        return graph

    def capture_all(self, model, cache):
        if not all(state.frozen for state in cache.states.values()):
            raise RuntimeError(
                'all active requests must be frozen before CUDA Graph capture')
        valid_visible_buckets = tuple(
            width for width in self.visible_buckets
            if width <= cache.max_visible_len)
        cache.ensure_visible_buckets(valid_visible_buckets)
        active_count = len(getattr(self.scheduler, 'active_req_ids', ()) or ())
        for bucket in self.buckets:
            if bucket > cache.max_requests or bucket < active_count:
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
        padded_ids = (req_ids if len(req_ids) == bucket else
                      req_ids + [-1] * (bucket - len(req_ids)))

        # Scheduler.prepare_step 已在同一 padded list 上调用 new_step；
        # _prepare_metadata 复用其 layout revision，不再重复 canonical 比较。

        # selector 不能放在整步 forward 前：各层必须在 post-RoPE 当前 Q 上
        # 独立选择。这里只刷新固定元数据和本图使用的 visible bucket。
        visible_bucket = self.visible_bucket_for(cache, padded_ids)
        self._prepare_metadata(cache, padded_ids, visible_bucket)
        self._copy_inputs(bucket, input_ids, position_ids)

        skip_signature = cache.nfa_skip_signature(padded_ids)
        key = self._key(bucket, visible_bucket, skip_signature)
        graph = self.graphs.get(key)
        if graph is not None:
            graph.replay()
            self.replay_count += 1
            logits = self.static_logits[key]
        else:
            self.eager_fallback_count += 1
            logits = model(
                input_ids=self.input_ids_buf[bucket],
                position_ids=self.position_ids_buf[bucket],
                past_key_values=cache,
                use_cache=True,
                attention_mask=self._attention_mask_arg(model),
            )
        # 第 N 步 GPU 工作已提交；此时处理第 N-1 步已就绪的 CPU 回执，
        # 不让上一帧 D2H 等待阻塞本帧 replay，也不依赖其完成来组装设备输入。
        self.scheduler._drain_pending_next_tokens()
        # 回执只在 step 边界记录一次；具体设备数据延迟到请求结束时读取，
        # 不在各层 forward 中更新 Python 对象或触发 D2H。
        cache.nfa_decode_steps += 1
        cache.commit_step()
        next_tokens = self.scheduler.record_next_tokens(
            padded_ids, logits, generated_counts=generated_counts,
            min_new_tokens=min_new_tokens, forced_tokens=forced_tokens,
            stop_on_eos=stop_on_eos)
        if return_next_tokens:
            return logits, next_tokens
        return logits


class GraphSessionPool:
    """Session 级 GraphRunner 池；描述符之外的对象绝不共享 CUDA Graph。"""

    def __init__(self):
        self._entries = {}
        self.hit_count = 0
        self.miss_count = 0
        self.warmup_seconds = 0.0

    @staticmethod
    def _descriptor(model, cache, buckets, visible_buckets, device):
        buffer_ptrs = [
            cache.layer_kv_ptrs_buf.data_ptr(),
            cache.block_table_buf.data_ptr(),
            cache.slot_mapping_buf.data_ptr(),
            cache.query_to_req_buf.data_ptr(),
            cache.active_req_ids_buf.data_ptr(),
            cache.valid_req_ids_buf.data_ptr(),
            cache.planned_lengths_buf.data_ptr(),
            cache.skip_far_layer_buf.data_ptr(),
            cache.skip_near_layer_buf.data_ptr(),
            cache.gather_indices_buf.data_ptr(),
            cache.visible_lens_buf.data_ptr(),
        ]
        buffer_ptrs.extend(
            layer.kv_pool_k.data_ptr() for layer in cache.layers)
        buffer_ptrs.extend(
            layer.kv_pool_v.data_ptr() for layer in cache.layers)
        buffer_ptrs.extend(
            tensor.data_ptr()
            for width, tensor in sorted(cache._gather_indices_by_vis.items()))
        buffer_ptrs.extend(
            tensor.data_ptr()
            for width, tensor in sorted(cache._visible_lens_by_vis.items()))
        return (
            id(model), id(cache), str(torch.device(device)),
            tuple(buckets), tuple(visible_buckets),
            cache.max_requests, cache.max_visible_len, cache.max_blocks,
            cache.n_layers, cache.num_kv_heads, cache.num_query_heads,
            tuple(buffer_ptrs),
        )

    def get_runner(self, model, cache, scheduler, *, buckets=(1,),
                   visible_buckets=(512, 1024, 2048, 4096),
                   device='cuda', capture=False):
        # 在计算描述符前物化所有将参与 capture 的 bucket，避免 warmup 后
        # 地址集合变化造成下一次请求误判为另一执行配置。
        cache.ensure_visible_buckets(
            width for width in visible_buckets if width <= cache.max_visible_len)
        descriptor = self._descriptor(
            model, cache, buckets, visible_buckets, device)
        entry = self._entries.get(descriptor)
        runner = None
        if entry is not None:
            cache_ref, candidate = entry
            if cache_ref() is cache:
                runner = candidate
            else:
                self._entries.pop(descriptor, None)
        if runner is None:
            self.miss_count += 1
            runner = GraphRunner(
                scheduler=scheduler, buckets=buckets,
                visible_buckets=visible_buckets, device=device)
            self._entries[descriptor] = (weakref.ref(cache), runner)
        else:
            self.hit_count += 1
            runner.scheduler = scheduler

        if capture:
            import time
            started = time.perf_counter()
            try:
                runner.capture_all(model, cache)
            except Exception:
                # capture 失败可能留下半初始化私有池；该描述符必须重新 warmup，
                # 不能让后续请求误用异常现场。
                self._entries.pop(descriptor, None)
                runner.scheduler = None
                self.warmup_seconds += time.perf_counter() - started
                raise
            self.warmup_seconds += time.perf_counter() - started
        return runner

    @staticmethod
    def release(runner):
        # 不让空闲 runner 通过旧 scheduler 强引用整个 Cache arena。
        runner.scheduler = None


def session_graph_runner(model, cache, scheduler, *, buckets=(1,),
                         visible_buckets=(512, 1024, 2048, 4096),
                         device='cuda', capture=False):
    pool = getattr(model, '_nfa_graph_session_pool', None)
    if pool is None:
        pool = GraphSessionPool()
        model._nfa_graph_session_pool = pool
    return pool.get_runner(
        model, cache, scheduler, buckets=buckets,
        visible_buckets=visible_buckets, device=device, capture=capture)
