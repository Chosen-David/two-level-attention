#!/usr/bin/env python3
"""审计并汇总 32K/64K/128K 全层统一四态的单层等效 NFA 算子数据。"""
import hashlib
import json
from pathlib import Path

REPO = Path('/home/wangyuanshuo02/two-level-attention')
OLD_RESULT = REPO / 'agent_doc/results/nfa_region_skip_performance_20261111.json'
NEW_ROOT = Path.home() / 'ak_verify_results/nfa_region_skip_forced_64_128_20261112'
OUT = REPO / 'agent_doc/results/nfa_region_four_state_per_layer_20261112.json'
MANIFEST = REPO / 'agent_doc/results/nfa_region_four_state_per_layer_raw_manifest_20261112.json'
MODES = ('no-skip', 'near-only', 'far-only', 'both-skip')
EXECUTIONS = ('graph', 'eager')
LENGTHS = (32768, 65536, 131072)
STAGES = (
    'l1_score_ms', 'l1_topk_ms', 'candidate_ms', 'l1_ms',
    'l2_score_ms', 'l2_topk_ms', 'selection_union_ms', 'l2_ms',
    'topk_total_ms', 'main_attention_ms', 'nfa_operator_total_ms',
)
OLD_SOURCE = {
    'head': '84bd19ee947c2498db7c988a9fa2ca766caf1267',
    'diff_sha256': '0ab60bfca394f1c2f1c52cbf9ba31f06cf3aa56bbb05b3ac8da5dffd826ba53f',
    'harness_sha256': '8bfa80e38d88995fa8902fd00dd34ec86acdb726a757eda6ff7a1fe9b0299aef',
}
NEW_SOURCE = {
    'head': 'b97af7ef909957e69b194e10ce4c97f5b1089be2',
    'diff_sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'harness_sha256': '8bfa80e38d88995fa8902fd00dd34ec86acdb726a757eda6ff7a1fe9b0299aef',
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def receipt_path(length, execution, mode):
    if length == 32768:
        root = Path.home() / 'ak_verify_results/nfa_region_skip_full_20261111'
    else:
        root = NEW_ROOT
    return (root / f'acceptance_{mode}_{execution}' / 'nfa-operator-profile' /
            f'nfa-operator-profile_l{length}_b1_{execution}.json')


def check_receipt(row, length, execution, mode):
    expected_source = OLD_SOURCE if length == 32768 else NEW_SOURCE
    assert row['schema'] == 'nfa-selector-and-e2e-v2'
    assert row['source'] == expected_source, (row['source'], expected_source)
    assert row['gpu_hygiene'] == 'clean'
    assert row['task'] == 'niah_multikey_2' and row['sample_index'] == 2
    assert row['context_length'] == length and row['batch_size'] == 1
    assert row['execution'] == execution and row['measurement_mode'] == 'operator-profile'
    assert row['force_skip_mode'] == mode and row['gate_epsilon'] == 0.1
    assert row['subspace'] == 'tail' and row['l2_subspace'] == 'same'
    assert row['score_dimensions'] == {
        'head_dim': 128, 'l1_score_dims': 32, 'l2_score_dims': 32}
    assert row['profile_repeat'] == 5 and row['profiled_decode_steps'] == 5
    assert row['nsys_profile_finalized'] is True
    assert row['region_kernel_audit']['passed'] is True
    assert row['graph_capture_count'] == (1 if execution == 'graph' else 0)
    if execution == 'graph':
        assert row['graph_replays'] == 8 and row['graph_eager_fallbacks'] == 0
    skip = row['skip']['0']
    assert skip['n_layers'] == 36
    expected = {
        'no-skip': (0, 0, 0), 'near-only': (0, 36, 0),
        'far-only': (36, 0, 0), 'both-skip': (36, 36, 36),
    }[mode]
    observed = tuple(len(skip[name]) for name in ('near_layers', 'far_layers', 'both_layers'))
    assert observed == expected, (length, execution, mode, observed, expected)
    calls = row['region_kernel_audit']['observed_calls_per_step']
    if mode == 'no-skip':
        assert calls['generic_l1'] == 36 and calls['near_only_l1'] == 0 and calls['far_only_l1'] == 0
    elif mode == 'near-only':
        assert calls['generic_l1'] == 0 and calls['near_only_l1'] == 36 and calls['far_only_l1'] == 0
    elif mode == 'far-only':
        assert calls['generic_l1'] == 0 and calls['near_only_l1'] == 0 and calls['far_only_l1'] == 36
    else:
        assert calls['generic_l1'] == calls['near_only_l1'] == calls['far_only_l1'] == 0
        assert calls['active_page_stats'] == calls['active_candidate'] == calls['active_fine_score'] == 0
    input_path = Path('/tmp/nfa_kernel_inputs') / str(length) / 'niah_multikey_2.jsonl'
    assert row['input']['sha256'] == sha256(input_path)
    nsys = row['nsys_receipt']
    for key in ('report', 'sqlite'):
        path = Path(nsys[f'{key}_path'])
        assert path.is_file() and nsys[f'{key}_sha256'] == sha256(path)
    stages = {name: row['operator_summary'][name]['mean_ms_per_step'] for name in STAGES}
    if mode == 'both-skip':
        for name in ('l1_score_ms', 'l1_topk_ms', 'candidate_ms', 'l1_ms',
                     'l2_score_ms', 'l2_topk_ms', 'topk_total_ms'):
            assert stages[name] == 0.0, (length, execution, name, stages[name])
        assert stages['selection_union_ms'] > 0 and stages['main_attention_ms'] > 0
    else:
        assert all(stages[name] > 0 for name in ('l1_ms', 'l2_ms', 'nfa_operator_total_ms'))
    return stages


def grid_evidence(row):
    wanted = (
        '_nfa_l1_scores_kernel', '_nfa_l1_near_scores_kernel', '_nfa_l1_far_scores_kernel',
        '_l1_topk_partial_kernel', '_topk_candidate_batched_kernel',
        '_candidate_expand_batched_kernel', '_nfa_filtered_topk_partial_kernel',
        '_nfa_fine_scores_paged_kernel',
    )
    return {
        name: {
            'mean_calls_per_step': row['kernel_summary'][name]['mean_calls_per_step'],
            'grid_x_values': row['kernel_summary'][name]['grid_x_values'],
            'grid_y_values': row['kernel_summary'][name]['grid_y_values'],
            'grid_z_values': row['kernel_summary'][name]['grid_z_values'],
        } for name in wanted if name in row['kernel_summary']
    }


def ratios(baseline, stages):
    result = {}
    for name in ('l1_ms', 'l2_ms', 'topk_total_ms', 'main_attention_ms',
                 'nfa_operator_total_ms'):
        denominator = stages[name]
        if denominator == 0:
            assert baseline[name] > 0
            result[name] = None
        else:
            result[name] = baseline[name] / denominator
    return result


def main():
    rows = {}
    raw_files = []
    for length in LENGTHS:
        for execution in EXECUTIONS:
            for mode in MODES:
                path = receipt_path(length, execution, mode)
                assert path.is_file(), path
                row = json.loads(path.read_text(encoding='utf-8'))
                stages = check_receipt(row, length, execution, mode)
                raw_artifact = artifact(path)
                nsys = row['nsys_receipt']
                nsys_artifacts = {
                    'report': artifact(nsys['report_path']),
                    'sqlite': artifact(nsys['sqlite_path']),
                }
                raw_files.extend((raw_artifact, nsys_artifacts['report'], nsys_artifacts['sqlite']))
                rows[length, execution, mode] = {
                    'length': length, 'batch': 1, 'execution': execution, 'mode': mode,
                    'mode_semantics': {
                        'no-skip': 'compute near and far',
                        'near-only': 'skip far; compute near only',
                        'far-only': 'skip near; compute far only',
                        'both-skip': 'skip near and far; protected-only union/attention remains',
                    }[mode],
                    'uniform_layers': 36,
                    'whole_model_ms_per_decode_step': stages,
                    'single_layer_equivalent_us_per_decode_step': {
                        name: value * 1000.0 / 36.0 for name, value in stages.items()},
                    'region_kernel_audit': row['region_kernel_audit'],
                    'grid_evidence': grid_evidence(row),
                    'source': row['source'],
                    'input_sha256': row['input']['sha256'],
                    'raw_receipt': raw_artifact,
                    'nsys': nsys_artifacts,
                }
    for length in LENGTHS:
        for execution in EXECUTIONS:
            baseline = rows[length, execution, 'no-skip']['whole_model_ms_per_decode_step']
            for mode in MODES:
                stages = rows[length, execution, mode]['whole_model_ms_per_decode_step']
                rows[length, execution, mode]['speedup_vs_no_skip'] = ratios(baseline, stages)
    # Graph 的分区 tile/grid 缩容必须与四态一起成立；eager 单 CTA 快路不伪造 grid 缩容。
    for length in LENGTHS:
        base = rows[length, 'graph', 'no-skip']['grid_evidence']
        for mode in ('near-only', 'far-only'):
            current = rows[length, 'graph', mode]['grid_evidence']
            for name in ('_l1_topk_partial_kernel', '_candidate_expand_batched_kernel',
                         '_nfa_filtered_topk_partial_kernel', '_nfa_fine_scores_paged_kernel'):
                assert max(current[name]['grid_y_values']) < max(base[name]['grid_y_values']), (
                    length, mode, name, current[name], base[name])
    manifest = {
        'schema': 'nfa-region-four-state-per-layer-raw-manifest-v1',
        'raw_receipt_count': 24,
        'nsys_report_sqlite_pair_count': 24,
        'file_count': len(raw_files),
        'files': raw_files,
    }
    assert len(raw_files) == 72
    write_json(MANIFEST, manifest)
    result = {
        'schema': 'nfa-region-four-state-per-layer-v1',
        'validation_status': 'pending-independent-verification',
        'internal_audit': {
            'status': 'PASS', 'raw_receipts_checked': 24,
            'nsys_report_sqlite_pairs_checked': 24,
            'input_sha256': 'PASS', 'source_identity': 'PASS',
            'gpu_hygiene': 'PASS', 'graph_capture_replay_fallback': 'PASS',
            'region_kernel_instances': 'PASS', 'graph_grid_shrinkage': 'PASS',
            'independent_verification': 'PENDING',
        },
        'protocol': {
            'model': 'Qwen3-8B', 'task': 'niah_multikey_2', 'sample_index': 2,
            'lengths': LENGTHS, 'batch': 1, 'subspace': 'tail=32/32',
            'gate_epsilon': 0.1, 'force_modes': MODES, 'executions': EXECUTIONS,
            'warmup_steps': 3, 'profiled_decode_steps': 5,
            'single_layer_equivalent': (
                'Every one of 36 layers is forced to the same mode; divide the '
                'whole-model per-decode-step stage total by 36. This is an arithmetic '
                'mean equivalent layer, not a measurement attributed to one layer index.'),
            'both_skip_boundary': (
                'L1/L2 score, candidate and Top-K are truly zero; protected-only '
                'selection union and selected attention are non-zero and reported separately.'),
            'operator_measurement': (
                'nsys/CUPTI observed kernel time over five profiled decode steps; '
                'observer effect remains; no clean TPOT or host-overhead claim.'),
            'source_scope': (
                '32K reuses audited pc31 receipts from 84bd19ee+0ab60bf source diff; '
                '64K/128K are new clean pc33 receipts at b97af7ef. Harness SHA256 is identical. '
                'Use within-length mode ratios; do not treat cross-host absolute values as a controlled length scaling study.'),
            'zero_stage_speedup': (
                'null when the measured stage is fully eliminated (0 ms); no Infinity or 1x is fabricated.'),
        },
        'records': [rows[length, execution, mode]
                    for length in LENGTHS for execution in EXECUTIONS for mode in MODES],
        'raw_receipt_manifest': artifact(MANIFEST),
    }
    write_json(OUT, result)
    print(json.dumps({
        'status': 'PASS', 'records': len(result['records']),
        'raw_receipts': 24, 'nsys_pairs': 24,
        'result': str(OUT), 'result_sha256': sha256(OUT),
        'manifest': str(MANIFEST), 'manifest_sha256': sha256(MANIFEST),
    }, indent=2))


if __name__ == '__main__':
    main()
