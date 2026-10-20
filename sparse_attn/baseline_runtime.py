"""Safe baseline planning and guarded native FFD integration.

【2026-10-20 batch decode 基础设施（deepseek 轨，Phase 0）】
本模块同时承载 vendor 稀疏 decode 的 batch>1 支持：模块级全局
`_BATCH_DECODE_CTX`（BatchCtx dataclass）与 E127 `_ERR_FLAG` 同模式
（模块级全局单例、set/clear 显式管理）。`require_unmasked_decode`
因此改为双路径：ctx is None → 旧单序列语义逐字保留（bsz=1 拒绝 +
原报错文案，既有 smoke/E127 零扰动）；ctx 非 None → 新 batch 契约
（校验 bsz==ctx.batch、4D mask 且 q_len==1、mask 派生有效长 ==
ctx.seq_lens 交叉校验）。各 vendor patch 在 batch 模式用
`batch_valid_mask(key_states, ctx)` 排除 left-pad 的 pad 列。
"""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import sys
import time
import weakref

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


@dataclass
class BatchCtx:
    """batch>1 稀疏 decode 的行状态上下文（left-pad 协议）。

    - batch: 固定 batch 大小（graph/缓存形状稳定的契约）
    - seq_lens: 每行当前真实上下文长度（decode 前，含已生成 token）
    - gen_lens: 每行已生成 token 数（不含首个预填充 token 自身计数口径，
      与 pred_ruler_batch 主循环的 gen 列表对齐）
    - alive: 每行是否存活（未触发 EOS 停止，False 后该行只吃 filler 计算）
    - pad_id: pad token id（left-pad 填充用）
    """
    batch: int
    seq_lens: list
    gen_lens: list
    alive: list
    pad_id: int


# 模块级全局 batch 上下文：None = legacy 单序列模式（各 vendor 既有行为
# 逐字保留）；非 None = batch 模式（由 pred_ruler_batch/pred_batch 在
# decode 循环外层 set/clear，与 E127 _ERR_FLAG 的显式生命周期管理同模式）。
_BATCH_DECODE_CTX = None


def set_batch_decode_ctx(ctx):
    """进入 batch decode 模式（ctx: BatchCtx）。必须与 clear 成对使用。"""
    global _BATCH_DECODE_CTX
    _BATCH_DECODE_CTX = ctx


def clear_batch_decode_ctx():
    """退出 batch decode 模式（恢复 legacy 单序列语义）。"""
    global _BATCH_DECODE_CTX
    _BATCH_DECODE_CTX = None


def batch_decode_ctx():
    """读取当前 batch 上下文（None = legacy 模式）。"""
    return _BATCH_DECODE_CTX


def batch_valid_mask(key_states, ctx=None):
    """生成 [B, 1, S] 有效列 mask（True=真实 token，False=left-pad 列）。

    left-pad 判据：行 b 的 pad 前缀长 = S - seq_lens[b]，因此有效列 =
    [S - seq_lens[b], S)。key_states: [B, kv_heads, S, d]。返回 broadcast
    到 [B, 1, S]（各 vendor 按需 expand 到 kv_heads 维或直接高级索引）。
    """
    import torch
    ctx = ctx if ctx is not None else _BATCH_DECODE_CTX
    if ctx is None:
        raise ValueError('batch_valid_mask requires an active batch decode ctx')
    S = key_states.shape[-2]
    if key_states.shape[0] != ctx.batch:
        raise ValueError(
            f'batch_valid_mask key batch {key_states.shape[0]} != ctx.batch '
            f'{ctx.batch}')
    col = torch.arange(S, device=key_states.device).view(1, 1, S)
    lens = torch.tensor(ctx.seq_lens, device=key_states.device,
                        dtype=torch.long).view(ctx.batch, 1, 1)
    return col >= (S - lens)  # [B, 1, S]


def batch_valid_len_from_mask(mask):
    """从 4D attention mask 派生每行有效长（防御性交叉校验用）。

    mask: [B, 1, q_len, S]。取末行（q_len==1）的非 pad 列数即有效长。
    """
    import torch
    if mask.ndim != 4:
        raise ValueError('batch_valid_len_from_mask expects a 4D mask')
    if mask.shape[1] != 1 or mask.shape[2] != 1:
        raise ValueError('batch_valid_len_from_mask expects q_len==1 mask')
    row = mask[..., 0, :]  # [B, S]
    if mask.dtype == torch.bool:
        return row.sum(dim=-1).tolist()
    return (row != 0).sum(dim=-1).tolist()


def require_unmasked_decode(hidden_states, attention_mask, cache, position):
    ctx = _BATCH_DECODE_CTX
    if ctx is None:
        # ---- legacy 单序列语义（逐字保留，bsz=1 拒绝 + 原报错文案）----
        if hidden_states.shape[0] != 1 or cache is None or position is None:
            raise ValueError('vendor sparse decode requires batch=1, cache and explicit position')
        if attention_mask is not None:
            import torch
            if attention_mask.ndim != 4 or attention_mask.shape[-2] != 1:
                raise ValueError('vendor sparse decode requires a 4D trivial attention mask')
            ok = attention_mask.all() if attention_mask.dtype == torch.bool else (attention_mask == 0).all()
            if not bool(ok):
                raise ValueError('vendor sparse decode padding/custom masks unsupported')
        return
    # ---- batch 契约（ctx 非 None）----
    import torch
    bsz = hidden_states.shape[0]
    if bsz != ctx.batch or cache is None or position is None:
        raise ValueError(
            f'batch vendor sparse decode requires bsz==ctx.batch '
            f'({ctx.batch}), cache and explicit position')
    if attention_mask is None:
        raise ValueError('batch vendor sparse decode requires a 4D attention mask')
    if attention_mask.ndim != 4 or attention_mask.shape[-2] != 1:
        raise ValueError('batch vendor sparse decode requires a 4D attention mask with q_len=1')
    # 交叉校验：mask 派生有效长 == ctx.seq_lens（防 left-pad 协议漂移）
    derived = batch_valid_len_from_mask(attention_mask)
    if derived != ctx.seq_lens:
        raise ValueError(
            f'batch vendor sparse decode mask derived lengths {derived} '
            f'!= ctx.seq_lens {ctx.seq_lens}')
    _ = torch  # 保持 import（与 legacy 分支同构）


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
    owned_caches = weakref.WeakKeyDictionary()
    cache_config = dict(BS=settings['BS'], k_bits=settings['k_bits'],
                        use_fp8_residual=settings['use_fp8_residual'], max_seq_len=capacity)
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
            cache = cache_type(**cache_config)
            owned_caches[cache] = dict(valid=True, length=0, layers=())
            kw['past_key_values'] = cache
        if not isinstance(cache, cache_type) or cache not in owned_caches:
            raise ValueError('FFD accepts only caches created by this loader; external caches rejected')
        state = owned_caches[cache]
        if not state['valid']:
            raise ValueError('FFD cache from an incomplete/failed call cannot be reused')
        current_layers = tuple(getattr(cache, 'layers', []))
        if len(current_layers) != len(state['layers']) or any(a is not b for a, b in zip(current_layers, state['layers'])):
            raise ValueError('FFD cache layer identity/count changed')
        for component in [cache, *current_layers]:
            if any(type(getattr(component, k, None)) is not type(v) or getattr(component, k) != v
                   for k, v in cache_config.items()):
                raise ValueError('FFD cache configuration differs from frozen settings')
        if getattr(cache, 'offloading', False):
            raise ValueError('FFD offloading cache is not supported')
        if cache.get_seq_length() == 0 and ids.shape[1] < settings['BS']:
            raise ValueError('FFD empty cache requires initial prefill >= BS=128')
        if cache.get_seq_length() != state['length'] or any(x.get_seq_length() != state['length'] for x in current_layers):
            raise ValueError('FFD cache layer lengths changed or diverged')
        if cache.get_seq_length() > 0 and ids.shape[1] > 1:
            raise ValueError('FFD chunked continuation unsupported; use initial prefill then single tokens')
        if cache.get_seq_length() + ids.shape[1] > capacity:
            raise ValueError('FFD cache capacity exceeded')
        mask = kw.get('attention_mask')
        if mask is not None and (mask.ndim != 2 or not bool((mask == 1).all())):
            raise ValueError('FFD candidate only supports unpadded full prompts')
        if set(module.config.attn_settings) != set(settings) | {'debug_stats'} or any(module.config.attn_settings.get(k) != v for k, v in settings.items()):
            raise ValueError('FFD runtime settings changed after identity freeze')
        pending.update(before=dict(stats), q_len=ids.shape[1], cache=cache, state=state)
        state['valid'] = False
        return positional, kw
    def after(module, positional, kw, output):
        old = pending['before']
        expected = 'sparse_path_calls' if pending['q_len'] == 1 else 'prefill_calls'
        if stats.get(expected, 0) - old.get(expected, 0) != layers:
            raise RuntimeError('FFD actual route mismatch; refusing mislabeled dense output')
        if stats.get('flash_path_decode_calls', 0) != old.get('flash_path_decode_calls', 0):
            raise RuntimeError('FFD unexpected dense decode')
        cache, state = pending['cache'], pending['state']
        expected_length = state['length'] + pending['q_len']
        if getattr(output, 'past_key_values', None) is not cache:
            raise RuntimeError('FFD output cache replaced')
        actual_layers = tuple(cache.layers)
        if state['layers'] and (len(actual_layers) != len(state['layers']) or any(a is not b for a, b in zip(actual_layers, state['layers']))):
            raise RuntimeError('FFD cache layers replaced during forward')
        if len(actual_layers) != layers or any(x.get_seq_length() != expected_length for x in actual_layers):
            raise RuntimeError('FFD incomplete cache layer update')
        if cache.get_seq_length() != expected_length:
            raise RuntimeError('FFD cache length update mismatch')
        state.update(valid=True, length=expected_length, layers=actual_layers)
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
        'ClusterKV':'official source/gitlinks fetched; project license unresolved; CUDA build/wheel/model adapter blocked',
        'HISA':'Xu et al., arXiv2603.28458v4; official TileLang prefill indexer fetched; no Qwen3/HF integration',
        'HiP/MoBA':'fixed sources fetched; HiP Qwen3 documented SGLang dense-decode route; MoBA native dense decode, continued-training condition',
    })

if __name__ == '__main__':
    print(json.dumps(plan(), indent=2))
