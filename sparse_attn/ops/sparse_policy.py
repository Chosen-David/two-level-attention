"""Static sparse-selector eligibility; no CUDA initialization or allocations.

Quality requests opt into the validated selector defaults within Graph state. These reasons
are capture-time decisions, not evidence that a captured graph was replayed.
"""
import os
import torch


def quality_defaults():
    from .cuda_graph_state import graph_state
    return graph_state().quality_defaults


def live_normalization_enabled(*, quality_request=False):
    default = '1' if quality_request or quality_defaults() or os.environ.get('TLI_BATCH_INDEXER') == 'exact' else '0'
    return os.environ.get('TLI_LIVE_NORMALIZATION', default) == '1'


def stats_policy():
    return os.environ.get('TLI_SPARSE_STATS', 'owner' if quality_defaults() or os.environ.get('TLI_BATCH_INDEXER') == 'exact' else 'sorted')


def geometry_fallback(dtype, device, capacity, group, heads):
    if torch.__version__.split('+')[0] != '2.8.0':
        return 'torch_version'
    if dtype != torch.bfloat16:
        return 'dtype'
    if device.type != 'cuda':
        return 'device'
    if not 9216 < capacity < 2**31 - 1:
        return 'capacity'
    stats = stats_policy()
    if stats not in ('sorted', 'owner'):
        return 'stats_policy'
    if capacity % 8 and stats != 'owner':
        return 'unaligned_requires_owner'
    if stats == 'owner' and heads * capacity >= 2**31:
        return 'owner_index_overflow'
    if group < 1 or group & (group - 1):
        return 'gqa_group'
    if not 1 <= heads * group <= 65535:
        return 'head_grid'
    return None


def selection_fallback(dtype, device, capacity, group, heads, *,
                       l2_bypass, protected_end, topk):
    policy = os.environ.get('TLI_SPARSE_SELECTION', '1' if quality_defaults() else '0')
    if policy != '1':
        return 'disabled' if policy == '0' else 'selection_policy'
    if l2_bypass:
        return 'l2_bypass'
    if protected_end > capacity:
        return 'protected_domain_exceeds_capacity'
    if min(capacity, topk) != 1024:
        return 'topk_requires_1024'
    return geometry_fallback(dtype, device, capacity, group, heads)
