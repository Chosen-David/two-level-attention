"""单进程真实 Qwen3 prefill gate 基线/候选采集器。

每次只跑一个请求，避免缓存与 allocator 跨配置残留。动态臂保存原生输出签名、
逐层 gate、region profile 与 core-token IDs；native 臂用于分离统计开销。
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from benchmark.efficiency.benchmark_nfa_end_to_end import _load_input
from benchmark.rope_policy import configure_rope, verify_runtime_rope
from sparse_attn.engine.frozen_runner import build_frozen_cache, prefill_frozen_request
from sparse_attn.info import resolve_treatment_snapshot
from sparse_attn.ops.paged_region_kernels_nfa import nfa_args
from sparse_attn.patches import register_patch
from sparse_attn.policies.frozen_gate import GateConfig
from sparse_attn.policies.region_geometry import RegionGeometry


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--task', default='niah_single_1')
    parser.add_argument('--context-length', type=int, required=True,
                        choices=(32768, 65536, 131072))
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--input-provenance', required=True,
                        choices=('native-ruler', 'repeated-needle-body-extension'))
    parser.add_argument('--mode', required=True, choices=('native', 'dynamic'))
    parser.add_argument('--profile-implementation', choices=('shared', 'legacy'),
                        default='shared')
    parser.add_argument('--label', required=True)
    return parser.parse_args()


def _sha256_file(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _source_receipt():
    diff = subprocess.check_output(
        ['git', '-C', REPO, 'diff', '--binary', 'HEAD'])
    paths = (
        'sparse_attn/policies/frozen_gate.py',
        'sparse_attn/engine/frozen_runner.py',
        'sparse_attn/patches/qwen3_attn_patch.py',
        'sparse_attn/patches/glm4_moe_lite_attn_patch.py',
        'exp/bench/prefill_gate_benchmark.py',
    )
    return {
        'head': subprocess.check_output(
            ['git', '-C', REPO, 'rev-parse', 'HEAD'], text=True).strip(),
        'diff_sha256': hashlib.sha256(diff).hexdigest(),
        'files': {path: _sha256_file(os.path.join(REPO, path)) for path in paths},
    }


def _logits_signature(logits):
    dense = logits.detach().contiguous()
    raw = dense.view(torch.int16).cpu().numpy().tobytes()
    return {
        'sha256': hashlib.sha256(raw).hexdigest(),
        'argmax_token_id': int(dense.argmax(dim=-1).item()),
        'shape': list(dense.shape),
        'dtype': str(dense.dtype),
    }


def _legacy_profile(q, k, geometry, rows=256, chunk=16, topk=64, scale=None):
    """修前 A 臂：region/core 独立统计，仅用于同进程配对性能基线。"""
    length = q.shape[2]
    n = min(rows, length)
    heads = q.shape[1]
    kk = k.detach().float().repeat_interleave(heads // k.shape[1], 1)
    pos = torch.arange(length, device=q.device)[None, :]
    signals = {'near': [], 'far': []}
    for start in range(length - n, length, chunk):
        end = min(start + chunk, length)
        qe = q.detach()[:, :, start:end].float()
        scores = qe @ kk.transpose(-1, -2) * scale
        qlen = torch.arange(start + 1, end + 1, device=q.device)[:, None]
        scores.masked_fill_(pos >= qlen, -torch.inf)
        probs = scores.softmax(-1)
        sink_hi = torch.minimum(qlen, torch.tensor(geometry.sink, device=q.device))
        near_hi = torch.maximum(sink_hi, qlen - geometry.window)
        far_hi = torch.minimum(near_hi, torch.maximum(
            sink_hi, ((qlen - geometry.window - geometry.near) // geometry.page) * geometry.page))
        masks = {'far': (pos >= sink_hi) & (pos < far_hi),
                 'near': (pos >= far_hi) & (pos < near_hi)}
        for name, mask in masks.items():
            signals[name].append((probs * mask).sum(-1).mean((0, 1)).cpu())
    profile = {name: {'mean': float(torch.cat(values).mean()),
                      'p95': float(torch.quantile(torch.cat(values), .95)),
                      'max': float(torch.cat(values).max())}
               for name, values in signals.items()}

    qs = q.detach()[:, :, length - n:].float()
    scores = qs @ kk.transpose(-1, -2) * scale
    pos = torch.arange(length, device=q.device)[None, None, :]
    qlen = torch.arange(length - n + 1, length + 1, device=q.device)[None, :, None]
    scores.masked_fill_(pos >= qlen, -torch.inf)
    token_scores = scores.softmax(-1).mean(dim=(0, 1)).mean(dim=0)
    sink_hi, far_hi, near_hi = geometry.bounds(length)
    ids = torch.arange(length, device=q.device)
    core = {}
    for name, lo, hi in (('far', sink_hi, far_hi), ('near', far_hi, near_hi)):
        positions = ids[(ids >= lo) & (ids < hi)]
        count = min(topk, positions.numel())
        core[name] = positions[torch.topk(token_scores[positions], count).indices].tolist()
    profile['core'] = core
    return profile


def main():
    args = parse_args()
    if args.mode == 'native' and args.profile_implementation != 'shared':
        raise ValueError('profile-implementation is only meaningful for dynamic mode')
    torch.manual_seed(20261102)
    config = AutoConfig.from_pretrained(args.model_path)
    if config.model_type != 'qwen3':
        raise ValueError('this benchmark currently requires Qwen3')
    configure_rope(config, args.context_length, 128,
                   yarn=args.context_length > 32768,
                   factor={65536: 2.0, 131072: 4.0}.get(args.context_length))
    config._attn_implementation = 'sdpa'
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompt, input_receipt = _load_input(args, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=config, dtype=torch.bfloat16).cuda().eval()
    verify_runtime_rope(model)
    patch_args = nfa_args()
    patched = register_patch(
        model, patch_args, resolve_treatment_snapshot(patch_args))
    if patched != config.num_hidden_layers:
        raise RuntimeError(f'NFA patched {patched}/{config.num_hidden_layers} layers')
    geometry = RegionGeometry(
        sink=patch_args.nfa_sink_tokens, near=patch_args.nfa_near_len,
        window=patch_args.nfa_swa_size, page=patch_args.tia_block_size,
        max_new_tokens=128)
    gate = GateConfig()
    if args.profile_implementation == 'legacy':
        import sparse_attn.engine.frozen_runner as frozen_runner
        frozen_runner.profile_regions_and_core_tokens = _legacy_profile
    cache = build_frozen_cache(model, geometry, prompt.numel())

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start = time.perf_counter()
    start_event.record()
    with torch.inference_mode():
        if args.mode == 'dynamic':
            output, audit = prefill_frozen_request(
                model, prompt.unsqueeze(0), gate, geometry, cache=cache, request_id=0)
        else:
            request_id = cache.add_request(0)
            cache.set_active_request(request_id)
            output = model(prompt.unsqueeze(0), past_key_values=cache,
                           use_cache=True, logits_to_keep=1)
            n_layers = config.num_hidden_layers
            cache.freeze(request_id, [False] * n_layers, [False] * n_layers,
                         {layer_idx: [] for layer_idx in range(n_layers)})
            audit = None
    end_event.record()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    result = {
        'label': args.label,
        'mode': args.mode,
        'profile_implementation': args.profile_implementation,
        'prefill_seconds': elapsed,
        'prefill_cuda_seconds': start_event.elapsed_time(end_event) / 1000,
        'peak_memory_gib': torch.cuda.max_memory_allocated() / 2**30,
        'context_length': args.context_length,
        'input': input_receipt,
        'model_path': os.path.abspath(args.model_path),
        'gate_config': {'epsilon': gate.epsilon, 'rows': gate.rows,
                        'statistic': gate.statistic, 'mode': gate.mode},
        'geometry': dict(sink=geometry.sink, near=geometry.near,
                         window=geometry.window, page=geometry.page,
                         max_new_tokens=geometry.max_new_tokens),
        'attention': {
            'implementation': config._attn_implementation,
            'torch_version': torch.__version__,
            'transformers_version': transformers.__version__,
            'flash_sdp_enabled': torch.backends.cuda.flash_sdp_enabled(),
            'mem_efficient_sdp_enabled': torch.backends.cuda.mem_efficient_sdp_enabled(),
            'math_sdp_enabled': torch.backends.cuda.math_sdp_enabled(),
        },
        'output': _logits_signature(output.logits),
        'source': _source_receipt(),
    }
    if audit is not None:
        result['gate'] = audit['decisions']
        result['region_profiles'] = {
            layer_idx: {name: profile[name] for name in ('near', 'far')}
            for layer_idx, profile in audit['profiles'].items()}
        result['core_token_ids'] = {
            layer_idx: profile['core']
            for layer_idx, profile in audit['profiles'].items()}
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({key: result[key] for key in (
        'label', 'mode', 'prefill_seconds', 'prefill_cuda_seconds',
        'peak_memory_gib', 'output')}, indent=2))
    print(f'saved -> {args.output}')


if __name__ == '__main__':
    main()
