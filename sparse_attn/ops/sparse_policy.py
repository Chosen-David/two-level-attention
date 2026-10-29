"""Static sparse-selector eligibility; no CUDA initialization or allocations.

Keep the opt-in default until the model quality gate is complete. These reasons
are capture-time decisions, not evidence that a captured graph was replayed.
"""
import os
import torch


def geometry_fallback(dtype, device, capacity, group, heads):
    if torch.__version__.split('+')[0] != '2.8.0':
        return 'torch_version'
    if dtype != torch.bfloat16:
        return 'dtype'
    if device.type != 'cuda':
        return 'device'
    if not 9216 < capacity < 2**31 - 1:
        return 'capacity'
    stats = os.environ.get('TLI_SPARSE_STATS', 'sorted')
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
    policy = os.environ.get('TLI_SPARSE_SELECTION', '0')
    if policy != '1':
        return 'disabled' if policy == '0' else 'selection_policy'
    if l2_bypass:
        return 'l2_bypass'
    if protected_end > capacity:
        return 'protected_domain_exceeds_capacity'
    if min(capacity, topk) != 1024:
        return 'topk_requires_1024'
    return geometry_fallback(dtype, device, capacity, group, heads)
