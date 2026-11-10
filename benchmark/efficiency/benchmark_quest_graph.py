"""Qwen3-8B Quest 固定缓存 PyTorch 语义复现的 Graph 测量。

仅 Graph；profiler 与 clean E2E 独立进程。结果不得称为官方 FP16/B1 CUDA
runtime 性能。计时口径与 NFA/JSSA/AsyncTLS/FlashInfer 完整 decode TPOT 对齐。
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from benchmark.efficiency import benchmark_nfa_end_to_end as common
from benchmark.RULER.pred_ruler import _ruler_eos_ids
from exp.vendor_baselines.quest_vendor.qwen3_quest_graph import (
    QuestQwen3GraphRuntime,
    bind_quest_graph_runtime,
    register_quest_graph_patch,
)
from sparse_attn.scheduler.scheduler import greedy_next_tokens


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--task', default='niah_multikey_2')
    parser.add_argument('--context-length', type=int, required=True,
                        choices=(32768, 65536, 131072, 262144, 524288))
    parser.add_argument('--batch-size', type=int, required=True, choices=(1, 2, 4))
    parser.add_argument('--execution', required=True, choices=('graph',))
    parser.add_argument('--sample-index', type=int, default=2)
    parser.add_argument('--input-provenance', required=True,
                        choices=('native-ruler', 'repeated-needle-body-extension'))
    parser.add_argument('--max-new-tokens', type=int, default=33)
    parser.add_argument('--warmup-steps', type=int, default=3)
    parser.add_argument('--visible-len', type=int, default=8192)
    parser.add_argument('--measurement-mode', required=True,
                        choices=('clean-e2e', 'operator-profile'))
    parser.add_argument('--quest-budget', type=int, default=2048)
    parser.add_argument('--quest-page-size', type=int, default=16)
    parser.add_argument('--quest-dense-layers', type=int, default=2)
    return parser.parse_args()


def _sha256(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _source_receipt():
    receipt = common._source_receipt()
    files = [
        Path(__file__),
        REPO / 'exp/vendor_baselines/quest_vendor/qwen3_quest_graph.py',
    ]
    receipt['quest_vendor_files_sha256'] = {
        str(path.relative_to(REPO)): _sha256(path) for path in files
    }
    return receipt


def _run(args, model, prompt, eos_ids):
    batch = args.batch_size
    length = prompt.numel()
    device = prompt.device
    decode_steps = args.max_new_tokens - 1
    runtime = QuestQwen3GraphRuntime(
        model, batch, length, args.max_new_tokens, device=device,
        token_budget=args.quest_budget, page_size=args.quest_page_size,
        dense_layers=args.quest_dense_layers)
    bind_quest_graph_runtime(model, runtime)

    input_ids = prompt.unsqueeze(0)
    cache_position = torch.arange(length, device=device)
    position_ids = cache_position.unsqueeze(0)
    torch.cuda.synchronize()
    prefill_start = common.time.perf_counter()
    next_token_rows = []
    for _ in range(batch):
        output = model(
            input_ids=input_ids, position_ids=position_ids,
            cache_position=cache_position, past_key_values=runtime.mask_metadata,
            use_cache=True, logits_to_keep=1)
        next_token_rows.append(greedy_next_tokens(
            output.logits, eos_ids, generated_counts=0, min_new_tokens=1))
        del output
    next_tokens = torch.cat(next_token_rows)
    generated = [[int(token)] for token in next_tokens.cpu()]
    torch.cuda.synchronize()
    prefill_seconds = common.time.perf_counter() - prefill_start
    del input_ids, position_ids, cache_position
    prefill_memory = common._memory_snapshot()

    runtime.prepare_decode(args.execution, decode_steps)
    input_buf = torch.empty((batch, 1), dtype=torch.long, device=device)
    position_buf = torch.empty((batch, 1), dtype=torch.long, device=device)
    cache_position_buf = torch.empty((1,), dtype=torch.long, device=device)

    def forward():
        return model(
            input_ids=input_buf, position_ids=position_buf,
            cache_position=cache_position_buf,
            past_key_values=runtime.mask_metadata, use_cache=True,
            logits_to_keep=1)

    input_buf.copy_(next_tokens[:, None])
    position_buf.fill_(length)
    cache_position_buf.fill_(length)
    runtime.prepare_decode_step(0)
    forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    runtime.prepare_decode_step(0)
    with torch.cuda.graph(graph):
        static_logits = forward()
    decode_ready_memory = common._memory_snapshot()

    event_pairs = []
    token_history = torch.empty(
        (decode_steps, batch), dtype=torch.long, device=device)
    steady_decode_start = None
    profiler_started = False
    for step in range(decode_steps):
        if args.measurement_mode == 'operator-profile' and step == args.warmup_steps:
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            profiler_started = True
        if args.measurement_mode == 'clean-e2e' and step >= args.warmup_steps:
            start, end = common._events()
            event_pairs.append((start, end))
            start.record()
        input_buf.copy_(next_tokens[:, None])
        position_buf.fill_(length + step)
        cache_position_buf.fill_(length + step)
        runtime.prepare_decode_step(step)
        graph.replay()
        logits = static_logits.logits
        next_tokens = greedy_next_tokens(
            logits, eos_ids, generated_counts=step + 1, min_new_tokens=1)
        if args.measurement_mode == 'clean-e2e' and step >= args.warmup_steps:
            end.record()
        token_history[step].copy_(next_tokens)
        if args.measurement_mode == 'clean-e2e' and step + 1 == args.warmup_steps:
            torch.cuda.synchronize()
            steady_decode_start = common._memory_snapshot()
            torch.cuda.reset_peak_memory_stats()
    if profiler_started:
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
    if steady_decode_start is None:
        steady_decode_start = common._memory_snapshot()
    torch.cuda.synchronize()
    for rows in token_history.cpu().tolist():
        for row in range(batch):
            generated[row].append(int(rows[row]))

    expected_decode_calls = runtime.num_layers * 2
    if runtime.prefill_calls != batch * runtime.num_layers:
        raise RuntimeError('Quest prefill call count mismatch')
    if runtime.decode_calls != expected_decode_calls:
        raise RuntimeError(
            f'Quest decode calls={runtime.decode_calls}, '
            f'expected={expected_decode_calls}')

    result = {
        'prefill_seconds': prefill_seconds,
        'generated_token_ids': generated,
        'graph_capture_count': 1,
        'graph_replays': decode_steps,
        'graph_eager_fallbacks': 0,
        'quest_runtime': runtime.receipt(),
        'memory': {
            'prefill_end': prefill_memory,
            'decode_ready': decode_ready_memory,
            'steady_decode_start': steady_decode_start,
            'decode_end': common._memory_snapshot(),
        },
    }
    if args.measurement_mode == 'clean-e2e':
        step_ms = [start.elapsed_time(end) for start, end in event_pairs]
        result.update(common._summary(step_ms, warmup=0))
    return result


def main():
    args = parse_args()
    if args.max_new_tokens < 2 or args.warmup_steps < 1:
        raise ValueError('need max-new-tokens>=2 and warmup-steps>=1')
    if args.max_new_tokens - 1 <= args.warmup_steps:
        raise ValueError('need at least one measured decode step after warmup')
    torch.manual_seed(20261102)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompt, input_receipt = common._load_input(args, tokenizer)
    args.method = 'quest-semantic-pytorch'
    model = common._load_model(args, args.max_new_tokens)
    patched = register_quest_graph_patch(model)
    if patched != model.config.num_hidden_layers:
        raise RuntimeError(
            f'Quest patched {patched}/{model.config.num_hidden_layers} layers')
    eos_ids = _ruler_eos_ids(model, tokenizer)
    model_loaded_memory = common._memory_snapshot()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        result = _run(args, model, prompt, eos_ids)
    memory = result['memory']
    memory['model_loaded'] = model_loaded_memory
    result.update(
        schema='qwen-quest-semantic-graph-e2e-v1',
        source=_source_receipt(),
        method='quest-semantic-pytorch',
        execution=args.execution,
        measurement_mode=args.measurement_mode,
        batch_size=args.batch_size,
        context_length=args.context_length,
        max_new_tokens=args.max_new_tokens,
        warmup_steps=args.warmup_steps,
        profiled_decode_steps=(
            args.max_new_tokens - 1 - args.warmup_steps
            if args.measurement_mode == 'operator-profile' else 0),
        visible_len=args.visible_len,
        task=args.task,
        sample_index=args.sample_index,
        input_provenance=args.input_provenance,
        model_path=str(Path(args.model_path).resolve()),
        input=input_receipt,
        decode_protocol='fixed length; EOS recorded but not used for early stop',
        prefill_batch_execution='serial one-request prefills',
        timing_scope=(
            'CUDA-event elapsed time around each complete steady decode step: '
            'input/position/page metadata updates, model and argmax'),
        timing_metric=(
            'end-to-end decode TPOT, not pure attention/operator latency'
            if args.measurement_mode == 'clean-e2e' else None),
        host_readback_in_decode_loop=False,
        token_history_copy_inside_timed_events=False,
        final_token_readback='one D2H after all decode steps',
        profiler_clean_e2e_isolated=True,
        profile_backend=(
            'nsys-cuda-graph-node-external'
            if args.measurement_mode == 'operator-profile' else None),
        nsys_profile_finalized=(
            False if args.measurement_mode == 'operator-profile' else None),
        host_overhead_derivable_by_subtraction=False,
        official_cuda=False,
        full_qk_mask_semantics=False,
        gpu_hygiene={'status': 'requires external process monitor'},
        torch_version=torch.__version__,
        transformers_version=__import__('transformers').__version__,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({
        'context_length': args.context_length,
        'batch_size': args.batch_size,
        'measurement_mode': args.measurement_mode,
        'tpot_ms_median': result.get('tpot_ms_median'),
        'quest_runtime': result['quest_runtime'],
        'output': str(output),
    }, indent=2))


if __name__ == '__main__':
    main()
