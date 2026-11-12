#!/usr/bin/env python3
"""从已审计四态回执提取纯 L1/L2 打分 kernel 的单层等效耗时。"""
import json
from pathlib import Path

import build_nfa_region_four_state_per_layer_20261112 as base

REPO = Path('/home/wangyuanshuo02/two-level-attention')
OUT = REPO / 'agent_doc/results/nfa_region_pure_score_per_layer_20261112.json'
PARENT = REPO / 'agent_doc/results/nfa_region_four_state_per_layer_20261112.json'
L1_KERNELS = {
    'no-skip': '_nfa_l1_scores_kernel',
    'near-only': '_nfa_l1_near_scores_kernel',
    'far-only': '_nfa_l1_far_scores_kernel',
}
L2_KERNEL = '_nfa_fine_scores_paged_kernel'


def kernel_record(row, name):
    kernel = row['kernel_summary'][name]
    assert kernel['mean_calls_per_step'] == 36.0
    return {
        'whole_model_ms_per_decode_step': kernel['mean_ms_per_step'],
        'single_layer_equivalent_us_per_decode_step': (
            kernel['mean_ms_per_step'] * 1000.0 / 36.0),
        'mean_calls_per_step': kernel['mean_calls_per_step'],
        'grid_x_values': kernel['grid_x_values'],
        'grid_y_values': kernel['grid_y_values'],
        'grid_z_values': kernel['grid_z_values'],
    }


def ratio(baseline, current):
    if current == 0:
        assert baseline > 0
        return None
    return baseline / current


def main():
    records = {}
    for length in base.LENGTHS:
        for execution in base.EXECUTIONS:
            for mode in base.MODES:
                path = base.receipt_path(length, execution, mode)
                row = json.loads(path.read_text(encoding='utf-8'))
                base.check_receipt(row, length, execution, mode)
                nsys = row['nsys_receipt']
                for key in ('report', 'sqlite'):
                    artifact_path = Path(nsys[f'{key}_path'])
                    assert artifact_path.is_file()
                    assert nsys[f'{key}_sha256'] == base.sha256(artifact_path)
                record = {
                    'length': length,
                    'batch': 1,
                    'execution': execution,
                    'mode': mode,
                    'uniform_layers': 36,
                    'source': row['source'],
                    'input_sha256': row['input']['sha256'],
                    'raw_receipt': base.artifact(path),
                }
                if mode == 'both-skip':
                    assert all(name not in row['kernel_summary'] for name in (
                        *L1_KERNELS.values(), L2_KERNEL))
                    record.update({
                        'l1_kernel': None,
                        'l2_kernel': None,
                        'l1_score': None,
                        'l2_score': None,
                        'single_layer_equivalent_us_per_decode_step': {
                            'l1_score_kernel': 0.0,
                            'l2_score_kernel': 0.0,
                        },
                    })
                else:
                    l1_name = L1_KERNELS[mode]
                    l1 = kernel_record(row, l1_name)
                    l2 = kernel_record(row, L2_KERNEL)
                    assert (l1['whole_model_ms_per_decode_step']
                            <= row['operator_summary']['l1_score_ms']['mean_ms_per_step'])
                    assert (l2['whole_model_ms_per_decode_step']
                            <= row['operator_summary']['l2_score_ms']['mean_ms_per_step'])
                    record.update({
                        'l1_kernel': l1_name,
                        'l2_kernel': L2_KERNEL,
                        'l1_score': l1,
                        'l2_score': l2,
                        'single_layer_equivalent_us_per_decode_step': {
                            'l1_score_kernel': l1[
                                'single_layer_equivalent_us_per_decode_step'],
                            'l2_score_kernel': l2[
                                'single_layer_equivalent_us_per_decode_step'],
                        },
                    })
                records[length, execution, mode] = record

    for length in base.LENGTHS:
        for execution in base.EXECUTIONS:
            baseline = records[length, execution, 'no-skip'][
                'single_layer_equivalent_us_per_decode_step']
            for mode in base.MODES:
                current = records[length, execution, mode][
                    'single_layer_equivalent_us_per_decode_step']
                records[length, execution, mode]['speedup_vs_no_skip'] = {
                    name: ratio(baseline[name], current[name])
                    for name in ('l1_score_kernel', 'l2_score_kernel')
                }
            for mode in ('near-only', 'far-only'):
                no_skip = records[length, execution, 'no-skip']
                current = records[length, execution, mode]
                assert (max(current['l1_score']['grid_x_values'])
                        < max(no_skip['l1_score']['grid_x_values']))
                assert (max(current['l2_score']['grid_y_values'])
                        < max(no_skip['l2_score']['grid_y_values']))

    result = {
        'schema': 'nfa-region-pure-score-per-layer-v1',
        'validation_status': 'pending-independent-verification',
        'internal_audit': {
            'status': 'PASS',
            'raw_receipts_checked': 24,
            'nsys_report_sqlite_pairs_checked': 24,
            'unique_cells': 24,
            'source_identity': 'PASS',
            'input_sha256': 'PASS',
            'gpu_hygiene': 'PASS',
            'region_kernel_instances': 'PASS',
            'pure_score_grid_shrinkage': 'PASS',
            'independent_verification': 'PENDING',
        },
        'protocol': {
            'model': 'Qwen3-8B',
            'task': 'niah_multikey_2',
            'sample_index': 2,
            'lengths': base.LENGTHS,
            'batch': 1,
            'subspace': 'tail=32/32',
            'gate_epsilon': 0.1,
            'force_modes': base.MODES,
            'executions': base.EXECUTIONS,
            'warmup_steps': 3,
            'profiled_decode_steps': 5,
            'units': 'microseconds per equivalent layer per decode step',
            'single_layer_equivalent': (
                'All 36 layers are forced to the same mode. Divide each named '
                'kernel whole-model mean_ms_per_step by 36 and multiply by 1000.'),
            'included_kernels': {
                'l1': L1_KERNELS,
                'l2': L2_KERNEL,
            },
            'excluded_work': [
                'Q preparation', 'page statistics', 'L1/L2 Top-K',
                'candidate preparation/expansion', 'selection union',
                'selected attention', 'all model non-selector work',
            ],
            'stage_bucket_boundary': (
                'operator_summary.l1_score_ms/l2_score_ms are broader stage '
                'buckets and are not used as pure score-kernel timings.'),
            'both_skip_boundary': (
                'No L1/L2 score kernel is launched; both timings are exactly zero.'),
            'source_scope': (
                '32K reuses audited pc31 receipts; 64K/128K use clean pc33 '
                'receipts. Use within-length mode ratios; do not interpret '
                'cross-host absolute values as controlled length scaling.'),
            'operator_measurement': (
                'nsys/CUPTI kernel observations over five profiled decode steps; '
                'observer effect remains; no clean TPOT or host-overhead claim.'),
            'zero_stage_speedup': (
                'null when the measured kernel is fully eliminated; no Infinity '
                'or 1x is fabricated.'),
        },
        'records': [records[length, execution, mode]
                    for length in base.LENGTHS
                    for execution in base.EXECUTIONS
                    for mode in base.MODES],
        'parent_stage_result': base.artifact(PARENT),
        'raw_receipt_manifest': base.artifact(base.MANIFEST),
    }
    assert len(result['records']) == 24
    base.write_json(OUT, result)
    print(json.dumps({
        'status': 'PASS',
        'records': len(result['records']),
        'result': str(OUT),
        'result_sha256': base.sha256(OUT),
        'raw_manifest_sha256': base.sha256(base.MANIFEST),
    }, indent=2))


if __name__ == '__main__':
    main()
