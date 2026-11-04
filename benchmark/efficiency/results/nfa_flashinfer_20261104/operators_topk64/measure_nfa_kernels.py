"""真实 Qwen decode 的 L1/L2/main-Attention 纯 CUDA kernel 测量。"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from types import SimpleNamespace

import torch
from torch.profiler import DeviceType, ProfilerActivity, profile

REPO = os.environ.get(
    'NFA_REPO', '/home/wangyuanshuo02/wt-nfa-hisa-20261103')
sys.path.insert(0, REPO)
from benchmark.efficiency import benchmark_nfa_end_to_end as bench
from sparse_attn.engine.frozen_runner import build_frozen_cache, prefill_frozen_request
from sparse_attn.engine.graph_runner import GraphRunner
from sparse_attn.ops.paged_region_kernels_nfa import nfa_args
from sparse_attn.policies.frozen_gate import GateConfig
from sparse_attn.policies.region_geometry import RegionGeometry
from sparse_attn.scheduler.scheduler import ContinuousBatchScheduler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, StaticCache

from kernel_stage_accounting import classify_nfa, corrected_arm_summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=('nfa', 'flashinfer'))
    parser.add_argument('--execution', required=True, choices=('eager', 'graph'))
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--context-length', required=True, type=int)
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--input-provenance', default='native-ruler')
    parser.add_argument('--gate-epsilon', type=float, default=0.05)
    parser.add_argument('--warmup-steps', type=int, default=3)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--visible-len', type=int, default=8192)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def load_model(args):
    if args.method == 'nfa' and args.context_length not in (98304, 262144):
        return bench._load_model(SimpleNamespace(
            model_path=args.model_path, context_length=args.context_length,
            method='nfa', execution=args.execution), 16)
    # 96K/256K 仅扩展位置容量；FlashInfer 臂的 prefill 使用 SDPA，
    # 被测 decode closure 仍只包含 FlashInfer Dense Attention。
    from benchmark.rope_policy import verify_runtime_rope
    config = AutoConfig.from_pretrained(args.model_path)
    if args.context_length > 32768:
        config.rope_scaling = dict(
            rope_type='yarn', type='yarn',
            factor=args.context_length / 32768.0,
            original_max_position_embeddings=32768, beta_fast=32, beta_slow=1)
        config.max_position_embeddings = args.context_length + 16
        config.rope_parameters = dict(
            config.rope_scaling, rope_theta=config.rope_theta)
    config._attn_implementation = 'sdpa'
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=config, dtype=torch.bfloat16).cuda().eval()
    verify_runtime_rope(model)
    if args.method == 'nfa':
        from sparse_attn.patches import register_patch
        register_patch(model, nfa_args())
    return model


def kernel_events(prof):
    rows = defaultdict(lambda: [0, 0.0])
    for event in prof.events():
        if event.device_type != DeviceType.CUDA:
            continue
        micros = event.self_device_time_total
        if micros <= 0:
            micros = event.device_time_total
        if micros <= 0:
            continue
        rows[event.name][0] += 1
        rows[event.name][1] += micros / 1000.0
    return {
        name: {'calls': calls, 'device_ms': total}
        for name, (calls, total) in sorted(rows.items())
    }


def profile_step(fn):
    with profile(activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA)) as prof:
        fn()
        torch.cuda.synchronize()
    return kernel_events(prof)


def skip_receipt(cache, request_id=0):
    state = cache.states[request_id]
    return {
        'near_layers': [i for i, flag in enumerate(state.skip_near_by_layer) if flag],
        'far_layers': [i for i, flag in enumerate(state.skip_far_by_layer) if flag],
        'both_layers': [i for i, (near, far) in enumerate(zip(
            state.skip_near_by_layer, state.skip_far_by_layer)) if near and far],
    }


def run_nfa(args, model, prompt):
    patch_args = nfa_args()
    geometry = RegionGeometry(
        sink=patch_args.nfa_sink_tokens, near=patch_args.nfa_near_len,
        window=patch_args.nfa_swa_size, page=patch_args.tia_block_size,
        max_new_tokens=args.warmup_steps + args.repeat + 2)
    cache = build_frozen_cache(
        model, geometry, prompt.numel(), max_requests=1,
        max_visible_len=args.visible_len)
    with torch.inference_mode():
        output, prefill_audit = prefill_frozen_request(
            model, prompt.unsqueeze(0), GateConfig(epsilon=args.gate_epsilon),
            geometry, cache=cache, request_id=0)
        first = output.logits[:, -1].argmax(-1)
        cache.transition_request(0, 'GENERATE')
    scheduler = ContinuousBatchScheduler(
        cache=cache, max_batch=1, eos_ids=set(), pad_token_id=0)
    scheduler.add_request(0)
    scheduler.next_tokens[0] = int(first[0])
    runner = GraphRunner(
        scheduler=scheduler, buckets=(1,), visible_buckets=(args.visible_len,),
        device=prompt.device)
    if args.execution == 'graph':
        runner.capture_all(model, cache)
    with torch.inference_mode():
        for step in range(args.warmup_steps):
            runner.step(model, cache, generated_counts=step + 1,
                        min_new_tokens=1, stop_on_eos=False)
        repeats = []
        for step in range(args.repeat):
            kernels = profile_step(lambda: runner.step(
                model, cache, generated_counts=args.warmup_steps + step + 1,
                min_new_tokens=1, stop_on_eos=False))
            stages, memberships = classify_nfa(kernels)
            repeats.append({'stages': stages, 'kernels': kernels,
                            'stage_memberships': memberships})
    return repeats, {
        'skip': skip_receipt(cache),
        'prefill_decisions': prefill_audit['decisions'],
        'profile_start_kv_tokens': prompt.numel() + args.warmup_steps,
    }


def run_flashinfer(args, model, prompt):
    import flashinfer
    length = prompt.numel()
    # 容量恰好覆盖 prefill + warmup，使去除 batch 维后的 HND K/V
    # 不需要 token 截断或复制，满足 FlashInfer contiguous 输入契约。
    cache = StaticCache(
        config=model.config, max_cache_len=length + args.warmup_steps)
    cache.early_initialization(
        1, model.config.num_key_value_heads, model.config.head_dim,
        model.dtype, prompt.device)
    position = torch.arange(length, device=prompt.device)
    with torch.inference_mode():
        output = model(
            input_ids=prompt.unsqueeze(0), position_ids=position.unsqueeze(0),
            cache_position=position, past_key_values=cache, use_cache=True,
            logits_to_keep=1)
        next_tokens = output.logits[:, -1].argmax(-1)
        del output
        # 与 NFA 相同，在正式 kernel 测量前推进 warmup 个真实 decode token。
        for step in range(args.warmup_steps):
            pos = torch.tensor([length + step], device=prompt.device)
            output = model(
                input_ids=next_tokens[:, None], position_ids=pos.unsqueeze(0),
                cache_position=pos, past_key_values=cache, use_cache=True,
                logits_to_keep=1)
            next_tokens = output.logits[:, -1].argmax(-1)
            del output
    kv_len = length + args.warmup_steps
    keys = [layer.keys[0, :, :kv_len] for layer in cache.layers]
    values = [layer.values[0, :, :kv_len] for layer in cache.layers]
    torch.manual_seed(20261104)
    queries = [torch.randn(
        model.config.num_attention_heads, model.config.head_dim,
        device=prompt.device, dtype=model.dtype) for _ in cache.layers]

    def dense_attention():
        return [flashinfer.single_decode_with_kv_cache(
            queries[i], keys[i], values[i], kv_layout='HND')
            for i in range(len(cache.layers))]

    with torch.inference_mode():
        outputs = dense_attention()
        torch.cuda.synchronize()
        graph = None
        if args.execution == 'graph':
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = graph_outputs = dense_attention()
        for _ in range(2):
            if graph is None:
                outputs = dense_attention()
            else:
                graph.replay()
        repeats = []
        for _ in range(args.repeat):
            fn = dense_attention if graph is None else graph.replay
            kernels = profile_step(fn)
            # 此 closure 只含 FlashInfer Dense Attention；所有设备 kernel 都计入。
            total = sum(row['device_ms'] for row in kernels.values())
            repeats.append({
                'stages': {'flashinfer_dense_ms': total},
                'kernels': kernels,
            })
        del outputs
        if graph is not None:
            del graph_outputs
    return repeats, {
        'profile_start_kv_tokens': kv_len,
        'flashinfer_version': flashinfer.__version__,
        'dense_layers': len(cache.layers),
        'kv_layout': 'HND',
        'prefill_attention': 'sdpa',
    }


def main():
    args = parse_args()
    torch.manual_seed(20261102)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    load_args = SimpleNamespace(
        data_root=args.data_root, task=args.task,
        context_length=args.context_length, sample_index=args.sample_index,
        input_provenance=args.input_provenance)
    prompt, input_receipt = bench._load_input(load_args, tokenizer)
    model = load_model(args)
    if args.method == 'nfa':
        repeats, metadata = run_nfa(args, model, prompt)
    else:
        repeats, metadata = run_flashinfer(args, model, prompt)
    result = {
        'schema': 'nfa-pure-kernel-stages-v2',
        'repo_head': subprocess.check_output(
            ['git', '-C', REPO, 'rev-parse', 'HEAD'], text=True).strip(),
        'repo_diff_sha256': hashlib.sha256(subprocess.check_output(
            ['git', '-C', REPO, 'diff'])).hexdigest(),
        'method': args.method, 'execution': args.execution,
        'batch_size': 1, 'model_path': args.model_path,
        'input': input_receipt, 'input_provenance': args.input_provenance,
        'warmup_steps': args.warmup_steps, 'repeat': args.repeat,
        'gate_epsilon': args.gate_epsilon if args.method == 'nfa' else None,
        'stage_definitions': {
            'l1': 'page stats/Q prepare + L1 scores + L1 Top-K/candidate pack',
            'l2': 'fine Q/scores + filtered Top-K + final union/physical mapping',
            'main_attention': 'gather_attend kernels only',
            'total': 'L1 + L2 + main Attention; no host/other model kernels',
        },
        'metadata': metadata,
        'summary': None,
        'repeats': repeats,
    }
    result['summary'] = corrected_arm_summary(result)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({key: value for key, value in result.items()
                      if key != 'repeats'}, indent=2))
    print('saved ->', args.output)


if __name__ == '__main__':
    main()
