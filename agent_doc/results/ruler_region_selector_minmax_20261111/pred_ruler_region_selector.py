"""本轮四臂对照的 GraphReady 入口适配；不改变选择器或生成语义。

NFA 臂仍走 pred_ruler 原生 champion 校验。两个单池臂仅绕过该校验中
champion-only 的 near/far 几何，复用同一 dense prefill、full gate、decode、
receipt 和提交路径。单池没有 near 候选页；cache 的 64-token 几何边界与
selector 的 POOL_NEAR_LEN=max(page, near_len) 一致。
"""
import os

from benchmark.RULER import pred_ruler
from sparse_attn.policies.frozen_gate import GateConfig
from sparse_attn.policies.region_geometry import RegionGeometry


_ORIGINAL_CONFIGURE = pred_ruler._configure_frozen_regions
_ORIGINAL_COMPLETE = pred_ruler.complete_frozen_request


def _complete_and_release_graph_session(model, cache, generated_count, audit):
    """独立 cache 请求结束后释放旧 Graph session，避免 128K KV arena 累积。"""
    try:
        return _ORIGINAL_COMPLETE(model, cache, generated_count, audit)
    finally:
        pool = getattr(model, '_nfa_graph_session_pool', None)
        if pool is not None:
            for _cache_ref, runner in pool._entries.values():
                runner.scheduler = None
            pool._entries.clear()
            model._nfa_graph_session_pool = None


pred_ruler.complete_frozen_request = _complete_and_release_graph_session


def _configure_single_pool(args, config):
    if getattr(config, 'model_type', None) != 'qwen3':
        raise ValueError('single-pool GraphReady adapter requires Qwen3')
    if getattr(config, 'head_dim', None) != 128:
        raise ValueError('single-pool GraphReady adapter requires head_dim=128')
    if args.method != 'nfa':
        raise ValueError('single-pool GraphReady adapter requires method=nfa')
    if args.nfa_frozen_region_mode != 'full':
        raise ValueError('this experiment requires full/no-skip gate')
    if (args.nfa_static_gate
            or getattr(args, 'nfa_enable_static_gate', False)):
        raise ValueError('this experiment forbids static gate artifacts')
    if args.nfa_far_method not in ('avg', 'minmax'):
        raise ValueError('single-pool L1 statistic must be avg or minmax')
    if (args.nfa_near_method, args.nfa_near_pages,
            args.nfa_near_tokens, args.nfa_near_len) != ('avg', 0, 'off', 0):
        raise ValueError('single-pool requires avg/0/off/0 near geometry')

    expected = {
        'tia_block_size': 64, 'tia_level1_topk': 128,
        'tia_level2_topk': 1024, 'tia_level2_cmp_ratio': 4,
        'nfa_subspace': 'tail', 'nfa_l2_subspace': 'same',
        'nfa_far_select': '4bit', 'nfa_near_select': '4bit',
        'nfa_sink_tokens': 4, 'nfa_swa_size': 128,
        'nfa_enable_subspace': True, 'nfa_enable_layer_skip': False,
        'nfa_dyn_dgate': False, 'nfa_far_skip_backend': 'legacy',
        'nfa_far_skip_cache': 'retain', 'nfa_sparse_prefill': False,
        'nfa_decode_dense': False, 'nfa_per_q_head': False,
        'tia_enable_async_topk': False, 'nfa_l2_quant': False,
        'nfa_sinkswa_only': False, 'nfa_boundary_bypass': True,
        'nfa_static_pair': False, 'nfa_proj_basis': None,
        'nfa_alpha': None, 'nfa_beta': None, 'nfa_gamma': None,
    }
    mismatches = {
        name: (getattr(args, name, None), value)
        for name, value in expected.items()
        if getattr(args, name, None) != value
    }
    for name in ('nfa_sparse_prefill', 'nfa_decode_dense'):
        if getattr(args, name, None) is None:
            mismatches.pop(name, None)
    if mismatches:
        raise ValueError(f'single-pool common geometry mismatch: {mismatches}')

    environment = {
        'NFA_OPERATOR_MODE': 'off', 'NFA_FINE_GATHER': '0',
        'NFA_SPARSE_PREFILL': '0', 'NFA_DECODE_DENSE': '0',
        'NFA_FROZEN_EAGER_REFERENCE': '0',
    }
    for name, value in environment.items():
        current = os.environ.get(name)
        if current is not None and current.lower() != value:
            raise ValueError(f'{name}={current!r} conflicts with frozen execution')
        os.environ[name] = value
    os.environ['NFA_CUDA_GRAPH'] = 'auto'
    config._attn_implementation = 'sdpa'

    max_new_tokens = pred_ruler._ruler_max_new_tokens(args.task)
    gate = GateConfig(
        mode='full', epsilon=args.nfa_frozen_region_epsilon,
        rows=args.nfa_frozen_region_rows,
        statistic=args.nfa_frozen_region_statistic)
    geometry = RegionGeometry(
        sink=args.nfa_sink_tokens, near=args.tia_block_size,
        window=args.nfa_swa_size, page=args.tia_block_size,
        max_new_tokens=max_new_tokens)
    args.nfa_frozen_region_max_new_tokens = max_new_tokens
    args._frozen_region_gate = gate
    args._frozen_region_geometry = geometry
    return gate, geometry


def _configure_frozen_regions(args, config):
    is_single_pool = (
        args.method == 'nfa'
        and args.nfa_prefill_frozen_regions
        and getattr(config, 'model_type', None) == 'qwen3'
        and getattr(args, 'nfa_near_pages', None) == 0
        and getattr(args, 'nfa_near_len', None) == 0
    )
    if is_single_pool:
        return _configure_single_pool(args, config)
    return _ORIGINAL_CONFIGURE(args, config)


pred_ruler._configure_frozen_regions = _configure_frozen_regions


if __name__ == '__main__':
    pred_ruler.main()
