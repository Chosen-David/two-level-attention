"""Safe baseline planning and guarded native FFD integration."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FFD_SHA = 'ca09458ab1536328e5f46502b27891412733cd6d'


def ffd_settings(args):
    delta = getattr(args, 'ffd_delta', 5.0)
    capacity = getattr(args, 'ffd_cache_length', 131200)
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta <= 0:
        raise ValueError('FFD delta must be finite and positive (upstream log2 units)')
    if type(capacity) is not int or capacity < 128 or capacity % 128:
        raise ValueError('FFD cache length must be a positive multiple of 128')
    return dict(use_sparse_decode=True, delta=float(delta), k_bits=2,
                use_fp8_residual=True, BS=128, use_full_graph=False), capacity


def verified_vendor(name):
    root = ROOT / 'exp' / 'vendor_baselines' / name
    manifest = json.loads((root / 'UPSTREAM.json').read_text())
    for rel, expected in manifest['files'].items():
        path = root / rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'vendor integrity failed: {name}/{rel}')
    return root


def ffd_identity(args):
    settings, capacity = ffd_settings(args)
    return dict(method='ffd_vendor', upstream=FFD_SHA, settings=settings,
                adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                cache_max_seq_len=capacity, model_support=['llama'],
                budget='variable top-delta; actual unique K unobserved',
                delta_units='upstream log2 (natural-log offset = delta * ln(2))',
                comparison='native quantized decode; not controlled same-K',
                gpu_quality_validated=False)


def baseline_preflight(args):
    method = getattr(args, 'method', None)
    if method in ('ffd_vendor', 'sparq_vendor'):
        if not getattr(args, 'baseline_execute', False):
            raise ValueError('candidate execution disabled; inspect python -m sparse_attn.baseline_runtime; pass --baseline-execute explicitly')
        verified_vendor(method)
    if method == 'ffd_vendor':
        ffd_settings(args)
        if getattr(args, 'context_length', 0) > 32768 or getattr(args, 'yarn', False):
            raise ValueError('FFD RULER candidate supports <=32K without Qwen YaRN only')
        if 'llama' not in str(getattr(args, 'model', '')).lower():
            raise ValueError('FFD fixed source requires Llama; Qwen3/MLA unsupported')


def require_unmasked_decode(hidden_states, attention_mask, cache, position):
    if hidden_states.shape[0] != 1 or cache is None or position is None:
        raise ValueError('vendor sparse decode requires batch=1, cache and explicit position')
    if attention_mask is not None:
        import torch
        if attention_mask.ndim != 4 or attention_mask.shape[-2] != 1:
            raise ValueError('vendor sparse decode requires a 4D trivial attention mask')
        ok = attention_mask.all() if attention_mask.dtype == torch.bool else (attention_mask == 0).all()
        if not bool(ok):
            raise ValueError('vendor sparse decode padding/custom masks unsupported')


def validate_minference_pattern(pattern, layers, heads, starting_layer):
    if type(starting_layer) is not int or not -1 <= starting_layer < layers:
        raise ValueError('MInference starting_layer would disable sparse prefill')
    if not isinstance(pattern, list) or len(pattern) != layers:
        raise ValueError('MInference requires exact layer pattern coverage')
    for layer in range(max(0, starting_layer), layers):
        row = pattern[layer]
        if not isinstance(row, dict) or set(row) != {str(h) for h in range(heads)}:
            raise ValueError('MInference missing/extra head patterns; default fallback forbidden')
        for entry in row.values():
            if (not isinstance(entry, (list, tuple)) or len(entry) != 4 or
                entry[0] not in ('vertical_and_slash', 'stream_llm', 'block_sparse') or
                any(type(x) is not int or x <= 0 for x in entry[1:3])):
                raise ValueError('invalid MInference head pattern')


def load_ffd(path, args, config=None):
    baseline_preflight(args)
    from transformers import AutoConfig
    settings, capacity = ffd_settings(args)
    if config is None:
        config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if config.model_type != 'llama':
        raise ValueError('FFD fixed upstream supports Llama only; Qwen/MLA adapter not validated')
    root = verified_vendor('ffd_vendor')
    sys.path.insert(0, str(root))
    import transformers
    if transformers.__version__ != '4.57.3':
        raise RuntimeError('FFD fixed source requires transformers==4.57.3')
    from ffd_core import LlamaForCausalLM, QuantizedKVCache
    import torch
    if not str(Path(sys.modules['ffd_core'].__file__).resolve()).startswith(str(root.resolve()) + '/'):
        raise RuntimeError('FFD import shadowed by another installation')
    from ffd_core.modeling import llama
    if (llama.flash_attn_func.__module__ == llama.__name__ and
            config.num_attention_heads != config.num_key_value_heads):
        raise RuntimeError('FFD native GQA requires FlashAttention; upstream SDPA fallback lacks enable_gqa')
    stats = {}
    config.attn_settings = dict(settings, debug_stats=stats)
    model = LlamaForCausalLM.from_pretrained(path, config=config, local_files_only=True,
                                           trust_remote_code=False, torch_dtype=torch.bfloat16,
                                           device_map='auto').eval()
    install_ffd_guard(model, QuantizedKVCache, stats, settings, capacity)
    model._baseline_identity = ffd_identity(args)
    from ffd_core.modeling import llama
    model._baseline_identity['prefill_backend'] = llama.flash_attn_func.__module__
    model._baseline_identity['adapter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    model._baseline_identity['torch_version'] = torch.__version__
    model._baseline_identity['transformers_version'] = transformers.__version__
    return model


def install_ffd_guard(model, cache_type, stats, settings, capacity):
    """Supply the real cache and check the route of each successful model call."""
    layers = model.config.num_hidden_layers
    pending = {}
    def before(module, positional, kw):
        ids = kw.get('input_ids')
        if positional or ids is None or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
            raise ValueError('FFD requires keyword input_ids [1, positive length]')
        if kw.get('use_cache', True) is not True:
            raise ValueError('FFD requires use_cache=True')
        cache = kw.get('past_key_values')
        if cache is None or (not isinstance(cache, cache_type) and hasattr(cache, 'get_seq_length') and cache.get_seq_length() == 0):
            if ids.shape[1] < settings['BS']:
                raise ValueError('FFD integration requires initial prefill >= BS=128')
            cache = cache_type(BS=128, use_fp8_residual=True, k_bits=2, max_seq_len=capacity)
            kw['past_key_values'] = cache
        if not isinstance(cache, cache_type):
            raise ValueError('FFD requires pinned QuantizedKVCache; dense cache rejected')
        if cache.get_seq_length() > 0 and ids.shape[1] > 1:
            raise ValueError('FFD chunked continuation unsupported; use initial prefill then single tokens')
        if cache.get_seq_length() + ids.shape[1] > capacity:
            raise ValueError('FFD cache capacity exceeded')
        mask = kw.get('attention_mask')
        if mask is not None and (mask.ndim != 2 or not bool((mask == 1).all())):
            raise ValueError('FFD candidate only supports unpadded full prompts')
        if set(module.config.attn_settings) != set(settings) | {'debug_stats'} or any(module.config.attn_settings.get(k) != v for k, v in settings.items()):
            raise ValueError('FFD runtime settings changed after identity freeze')
        pending.update(before=dict(stats), q_len=ids.shape[1])
        return positional, kw
    def after(module, positional, kw, output):
        old = pending['before']
        expected = 'sparse_path_calls' if pending['q_len'] == 1 else 'prefill_calls'
        if stats.get(expected, 0) - old.get(expected, 0) != layers:
            raise RuntimeError('FFD actual route mismatch; refusing mislabeled dense output')
        if stats.get('flash_path_decode_calls', 0) != old.get('flash_path_decode_calls', 0):
            raise RuntimeError('FFD unexpected dense decode')
    model.register_forward_pre_hook(before, with_kwargs=True)
    model.register_forward_hook(after, with_kwargs=True)
    model._baseline_stats = stats


@contextmanager
def baseline_request(model):
    """Host wall includes cache/index work and completion sync, not kernel-only time."""
    identity = getattr(model, '_baseline_identity', None)
    result = {}
    if identity is None:
        yield result
        return
    import torch
    devices = {p.device for p in model.parameters() if p.device.type == 'cuda'}
    def sync():
        for device in devices:
            torch.cuda.synchronize(device)
    stats = model._baseline_stats
    before = dict(stats)
    sync()
    start = time.perf_counter()
    if not math.isfinite(start):
        raise ValueError('nonfinite baseline start clock')
    yield result
    sync()
    end = time.perf_counter()
    elapsed = (end - start) * 1000
    if not math.isfinite(end) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('nonfinite/nonpositive baseline duration')
    diff = {k: v - before.get(k, 0) for k, v in stats.items()}
    if set(before) - set(stats) or any(v < 0 for v in diff.values()):
        raise RuntimeError('baseline counters reset within request')
    sparse_key = {'ffd_vendor': 'sparse_path_calls', 'sparq_vendor': 'sparq_decode_calls'}.get(identity['method'])
    if sparse_key and diff.get(sparse_key, 0) == 0:
        raise RuntimeError('baseline request never exercised sparse decode; no accepted receipt')
    result.update(identity=identity, routes=diff, request_wall_ms=elapsed,
                  timing_scope='generation context including cache/index and boundary sync',
                  actual_unique_k=None, quality_accepted=False, status='candidate')


def plan():
    return dict(execution='disabled', gpu_started=False, methods={
        'ffd_vendor':'native Llama candidate; Qwen3/MLA unsupported; variable top-delta Q2+FP8',
        'sparq_vendor':'Qwen3 per-query-head gpt-fast adapter candidate; k=1024, context-derived r',
        'quest_vendor':'Qwen3 quality adapter; full QK mask, not efficient sparse-read kernel',
        'fasa_vendor':'Qwen3 calibration-dependent adapter; GPU/quality unverified in this change',
        'minference_vendor':'Qwen3 sparse-prefill/dense-decode adapter; frozen pattern required',
        'ClusterKV':'blocked: no local official efficiency dependencies/model adapter',
        'HISA':'blocked: exact identity not established; no guessed implementation',
        'HiP/MoBA':'conditional candidates; no unconditional model support',
    })

if __name__ == '__main__':
    print(json.dumps(plan(), indent=2))
