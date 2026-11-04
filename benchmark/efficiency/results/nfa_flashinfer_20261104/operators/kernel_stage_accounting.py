"""NFA 纯 kernel 阶段的 exactly-once 分类与覆盖校验。"""

STAGE_TOKENS = {
    'l1_ms': (
        '_nfa_page_stats_kernel', '_nfa_page_stats_prepare_q_kernel',
        '_nfa_l1_scores_kernel', '_prepare_q_kernel',
        '_topk_candidate_prepare_kernel', '_topk_candidate_batched_kernel',
        '_candidate_expand_batched_kernel', '_candidate_pack_batched_kernel',
    ),
    'l2_ms': (
        '_prepare_q_grouped_kernel', '_nfa_fine_scores_paged_kernel',
        '_nfa_filtered_topk_kernel', '_nfa_filtered_topk_radix_kernel',
        '_nfa_filtered_topk_partial_kernel', '_nfa_filtered_topk_merge_kernel',
        '_nfa_finish_topk_kernel', '_final_union_physical_kernel',
    ),
    'main_attention_ms': (
        '_gather_attend_kernel', '_gather_attend_grouped_kernel',
        '_gather_attend_grouped_split_kernel', '_gather_attend_split_combine_kernel',
    ),
}

# 这些词命中但未被上面的阶段表接收时，必须失败，不能静默漏记新 selector kernel。
NFA_AUDIT_TOKENS = (
    'nfa', 'topk', 'candidate', 'union', 'gather_attend',
    'prepare_q', 'fine_scores', 'page_stats', 'l1_scores',
)


def classify_nfa(kernels):
    """按单次 repeat 分类；unknown、重复归属或空阶段均报错。"""
    stages = {name: 0.0 for name in STAGE_TOKENS}
    memberships = {}
    for name, row in kernels.items():
        matches = [
            stage for stage, tokens in STAGE_TOKENS.items()
            if any(token in name for token in tokens)
        ]
        if len(matches) > 1:
            raise ValueError(f'kernel 重复归属: {name} -> {matches}')
        if matches:
            stage = matches[0]
            stages[stage] += row['device_ms']
            memberships[name] = stage
        elif any(token in name.lower() for token in NFA_AUDIT_TOKENS):
            raise ValueError(f'未分类的 NFA 相关 kernel: {name}')
    missing = [stage for stage in STAGE_TOKENS if stage not in memberships.values()]
    if missing:
        raise ValueError(f'NFA 算子链缺少阶段: {missing}')
    stages['nfa_operator_total_ms'] = sum(stages.values())
    return stages, memberships


def summarize_values(values):
    import statistics
    return {
        'values_ms': values,
        'median_ms': statistics.median(values),
        'min_ms': min(values),
        'max_ms': max(values),
    }


def corrected_arm_summary(row):
    """忽略旧 summary，从每次 repeat 的原始 kernel 重新计算。"""
    if row['method'] == 'nfa':
        repeats = [classify_nfa(rep['kernels'])[0] for rep in row['repeats']]
    else:
        repeats = [
            {'flashinfer_dense_ms': sum(
                kernel['device_ms'] for kernel in rep['kernels'].values())}
            for rep in row['repeats']
        ]
    names = sorted(repeats[0])
    return {
        name: summarize_values([rep[name] for rep in repeats])
        for name in names
    }
