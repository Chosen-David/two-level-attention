"""真实 Qwen RULER decode：当前 Q NFA / FlashAttention-2，支持 B1/2/4 与 Graph。

性能口径固定复制同一条 RULER prompt 到 B 行，并执行固定长度 decode；它不是
质量评测。输入来源由 --input-provenance 显式记录，实际 token 长度必须落在
声明档位的 95%--100%；64K/128K 循环体扩展数据不得冒充原生 RULER。
每个进程只跑一个配置，避免 CUDA Graph 私有池跨配置残留。
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager, nullcontext

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from benchmark.rope_policy import configure_rope, verify_runtime_rope
from sparse_attn.engine.frozen_runner import build_frozen_cache, prefill_frozen_request
from sparse_attn.engine.graph_runner import GraphRunner
from sparse_attn.info import resolve_treatment_snapshot
from sparse_attn.ops.paged_region_kernels_nfa import nfa_args
from sparse_attn.patches import register_patch
from sparse_attn.policies.frozen_gate import GateConfig
from sparse_attn.policies.region_geometry import RegionGeometry
from sparse_attn.scheduler.scheduler import ContinuousBatchScheduler, greedy_next_tokens


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--task', default='niah_single_1')
    parser.add_argument('--context-length', type=int, required=True,
                        choices=(32768, 65536, 131072))
    parser.add_argument('--batch-size', type=int, required=True, choices=(1, 2, 4))
    parser.add_argument('--method', required=True, choices=('nfa', 'fa2'))
    parser.add_argument('--execution', required=True, choices=('eager', 'graph'))
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--input-provenance', required=True,
                        choices=('native-ruler', 'repeated-needle-body-extension'),
                        help='输入数据来源必须显式声明，禁止按长度猜测')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--warmup-steps', type=int, default=3)
    parser.add_argument('--visible-len', type=int, default=8192)
    parser.add_argument('--fa2-prefill-cpu-offload', action='store_true',
                        help='仅在 prefill 逐层卸载 KV 以降低峰值；decode 前恢复固定 GPU Cache')
    return parser.parse_args()


def _load_input(args, tokenizer):
    path = os.path.join(args.data_root, str(args.context_length), f'{args.task}.jsonl')
    with open(path, 'rb') as handle:
        data_sha256 = hashlib.sha256(handle.read()).hexdigest()
    with open(path, encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle]
    row = rows[args.sample_index]
    ids = tokenizer(row['input'], return_tensors='pt', truncation=False).input_ids
    prompt_tokens = ids.shape[-1]
    if not 0.95 * args.context_length <= prompt_tokens <= args.context_length:
        raise ValueError(
            f'RULER prompt length mismatch: requested {args.context_length}, '
            f'got {prompt_tokens} tokens from {path} sample {args.sample_index}')
    return ids[0].cuda(), {
        'path': os.path.abspath(path), 'sha256': data_sha256,
        'sample_index': args.sample_index, 'sample_id': row.get('index'),
        'actual_prompt_tokens': prompt_tokens,
        'provenance': args.input_provenance,
        'performance_input_mode': 'replicate one prompt across all batch rows',
    }


def _load_model(args, max_new_tokens):
    config = AutoConfig.from_pretrained(args.model_path)
    if config.model_type != 'qwen3':
        raise ValueError('this production benchmark currently requires Qwen3')
    configure_rope(config, args.context_length, max_new_tokens,
                   yarn=args.context_length > 32768,
                   factor={65536: 2.0, 131072: 4.0}.get(args.context_length))
    config._attn_implementation = 'sdpa' if args.method == 'nfa' else 'flash_attention_2'
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=config, torch_dtype=torch.bfloat16).cuda().eval()
    verify_runtime_rope(model)
    if args.method == 'nfa':
        patch_args = nfa_args()
        patched = register_patch(model, patch_args, resolve_treatment_snapshot(patch_args))
        if patched != config.num_hidden_layers:
            raise RuntimeError(f'NFA patched {patched}/{config.num_hidden_layers} layers')
    return model


def _events():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


@contextmanager
def _fa2_no_packed_position_ids():
    """固定无 padding 输入避免 packed D2H；可选在 prefill 注意力后逐层卸载 KV。"""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original = ALL_ATTENTION_FUNCTIONS['flash_attention_2']
    state = {'prefill_cache': None}

    def forward(module, query, key, value, *args, **kwargs):
        kwargs.pop('position_ids', None)
        result = original(module, query, key, value, *args, **kwargs)
        cache = state['prefill_cache']
        if cache is not None:
            layer = cache.layers[module.layer_idx]
            layer.keys = layer.keys.to('cpu')
            layer.values = layer.values.to('cpu')
        return result

    ALL_ATTENTION_FUNCTIONS['flash_attention_2'] = forward
    try:
        yield state
    finally:
        ALL_ATTENTION_FUNCTIONS['flash_attention_2'] = original


def _summary(step_ms, warmup):
    steady = step_ms[warmup:] or step_ms
    return {
        'decode_steps': len(step_ms),
        'tpot_ms_median': statistics.median(step_ms),
        'steady_tpot_ms_median': statistics.median(steady),
        'tpot_ms_all': step_ms,
        'warmup_steps_excluded_from_steady': min(warmup, len(step_ms)),
    }


def _run_nfa(args, model, prompt, max_new_tokens, eos_ids):
    batch = args.batch_size
    patch_args = nfa_args()
    geometry = RegionGeometry(
        sink=patch_args.nfa_sink_tokens, near=patch_args.nfa_near_len,
        window=patch_args.nfa_swa_size, page=patch_args.tia_block_size,
        max_new_tokens=max_new_tokens)
    cache = build_frozen_cache(
        model, geometry, prompt.numel(), max_requests=batch,
        max_visible_len=args.visible_len)
    gate = GateConfig()
    generated = {}
    req_ids = []
    torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    for request_id in range(batch):
        output, _ = prefill_frozen_request(
            model, prompt.unsqueeze(0), gate, geometry,
            cache=cache, request_id=request_id)
        first = greedy_next_tokens(
            output.logits, eos_ids, generated_counts=0, min_new_tokens=1)
        generated[request_id] = [int(first[0])]
        req_ids.append(request_id)
        cache.transition_request(request_id, 'GENERATE')
        del output
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - prefill_start

    scheduler = ContinuousBatchScheduler(
        cache=cache, max_batch=batch, eos_ids=eos_ids, pad_token_id=0)
    for request_id in req_ids:
        scheduler.add_request(request_id)
        scheduler.next_tokens[request_id] = generated[request_id][0]
    runner = GraphRunner(
        scheduler=scheduler, buckets=(batch,), visible_buckets=(args.visible_len,),
        device=prompt.device)
    if args.execution == 'graph':
        runner.capture_all(model, cache)

    step_ms = []
    for _ in range(max_new_tokens - 1):
        counts = [len(generated[request_id]) for request_id in req_ids]
        start, end = _events()
        start.record()
        _, tokens = runner.step(
            model, cache, generated_counts=counts, min_new_tokens=1,
            return_next_tokens=True, stop_on_eos=False)
        end.record()
        rows = tokens.detach().cpu().tolist()
        for row, request_id in enumerate(req_ids):
            generated[request_id].append(int(rows[row]))
        step_ms.append(start.elapsed_time(end))
    scheduler.prepare_step()
    result = _summary(step_ms, args.warmup_steps)
    result.update(prefill_seconds=prefill_seconds,
                  generated_token_ids=[generated[request_id] for request_id in req_ids],
                  gate_mode='dynamic', graph_replays=(len(step_ms) if args.execution == 'graph' else 0))
    return result


def _run_fa2(args, model, prompt, max_new_tokens, eos_ids, fa2_state):
    batch = args.batch_size
    length = prompt.numel()
    device = prompt.device
    cache = StaticCache(config=model.config, max_cache_len=length + max_new_tokens)
    if not args.fa2_prefill_cpu_offload:
        cache.early_initialization(
            batch, model.config.num_key_value_heads, model.config.head_dim,
            model.dtype, device)
    input_ids = prompt.unsqueeze(0).expand(batch, -1).contiguous()
    cache_position = torch.arange(length, device=device)
    position_ids = cache_position.unsqueeze(0).expand(batch, -1)
    torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    fa2_state['prefill_cache'] = cache if args.fa2_prefill_cpu_offload else None
    try:
        output = model(
            input_ids=input_ids, position_ids=position_ids,
            cache_position=cache_position, past_key_values=cache,
            use_cache=True, logits_to_keep=1)
    finally:
        fa2_state['prefill_cache'] = None
    if args.fa2_prefill_cpu_offload:
        for layer in cache.layers:
            layer.keys = layer.keys.to(device)
            layer.values = layer.values.to(device)
    next_tokens = greedy_next_tokens(
        output.logits, eos_ids, generated_counts=0, min_new_tokens=1)
    generated = [[int(token)] for token in next_tokens.cpu()]
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - prefill_start
    del output

    input_buf = torch.empty((batch, 1), dtype=torch.long, device=device)
    position_buf = torch.empty((batch, 1), dtype=torch.long, device=device)
    cache_position_buf = torch.empty((1,), dtype=torch.long, device=device)
    graph = None
    static_logits = None

    def forward():
        return model(
            input_ids=input_buf, position_ids=position_buf,
            cache_position=cache_position_buf, past_key_values=cache,
            use_cache=True, logits_to_keep=1)

    if args.execution == 'graph':
        input_buf.copy_(next_tokens[:, None])
        position_buf.fill_(length)
        cache_position_buf.fill_(length)
        forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_logits = forward()

    step_ms = []
    for step in range(max_new_tokens - 1):
        start, end = _events()
        input_buf.copy_(next_tokens[:, None])
        position_buf.fill_(length + step)
        cache_position_buf.fill_(length + step)
        start.record()
        if graph is None:
            logits = forward().logits
        else:
            graph.replay()
            logits = static_logits.logits
        next_tokens = greedy_next_tokens(
            logits, eos_ids, generated_counts=step + 1, min_new_tokens=1)
        end.record()
        rows = next_tokens.cpu().tolist()
        for row in range(batch):
            generated[row].append(int(rows[row]))
        step_ms.append(start.elapsed_time(end))
    result = _summary(step_ms, args.warmup_steps)
    result.update(prefill_seconds=prefill_seconds, generated_token_ids=generated,
                  prefill_cpu_offload=args.fa2_prefill_cpu_offload,
                  graph_replays=(len(step_ms) if graph is not None else 0))
    return result


def main():
    args = parse_args()
    if args.max_new_tokens < 2:
        raise ValueError('max-new-tokens must be at least 2 for decode timing')
    if args.fa2_prefill_cpu_offload and args.method != 'fa2':
        raise ValueError('fa2-prefill-cpu-offload is FA2-only')
    torch.manual_seed(20261102)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompt, input_receipt = _load_input(args, tokenizer)
    model = _load_model(args, args.max_new_tokens)
    from benchmark.RULER.pred_ruler import _ruler_eos_ids
    eos_ids = _ruler_eos_ids(model, tokenizer)
    fa2_scope = _fa2_no_packed_position_ids() if args.method == 'fa2' else nullcontext()
    with torch.inference_mode(), fa2_scope as fa2_state:
        if args.method == 'nfa':
            result = _run_nfa(args, model, prompt, args.max_new_tokens, eos_ids)
        else:
            result = _run_fa2(args, model, prompt, args.max_new_tokens, eos_ids, fa2_state)
    result.update(
        method=args.method, execution=args.execution, batch_size=args.batch_size,
        context_length=args.context_length, max_new_tokens=args.max_new_tokens,
        model_path=os.path.abspath(args.model_path), input=input_receipt,
        decode_protocol='fixed length; EOS recorded but not used for early stop',
        timing_scope='CUDA events around model decode, selector where applicable, and argmax',
        gpu_hygiene={'status': 'requires external process monitor'},
        transformers_version=__import__('transformers').__version__,
        torch_version=torch.__version__)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('generated_token_ids', 'tpot_ms_all')}, indent=2))
    print(f"saved -> {args.output}")


if __name__ == '__main__':
    main()
