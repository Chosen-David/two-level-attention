"""Independently validate the three fused-LSE prefill increment receipts."""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import subprocess
from pathlib import Path

RESULTS = Path(__file__).resolve().parent
REPO = RESULTS.parents[2]
LENGTHS = (32768, 65536, 131072)
EMPTY_DIFF_SHA256 = hashlib.sha256(b'').hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise AssertionError(f'{actual} != {expected}')


def validate_summary(actual, measurements):
    dense = [row['dense']['wall_seconds'] for row in measurements]
    fused = [row['fused']['wall_seconds'] for row in measurements]
    delta = [row['delta_seconds'] for row in measurements]
    dense_median = statistics.median(dense)
    fused_median = statistics.median(fused)
    paired_median = statistics.median(delta)
    device = statistics.median([
        row['fused']['calibrator']['device_seconds_total']
        for row in measurements])
    host_wait = statistics.median([
        row['fused']['calibrator']['host_wait_seconds_total']
        for row in measurements])
    expected = {
        'repeat': len(measurements),
        'dense_prefill_median_seconds': dense_median,
        'fused_prefill_median_seconds': fused_median,
        'increment_over_dense_median_seconds': fused_median - dense_median,
        'increment_over_dense_ratio': (fused_median - dense_median) / dense_median,
        'paired_increment_median_seconds': paired_median,
        'paired_increment_over_dense_ratio': statistics.median(
            value / base for value, base in zip(delta, dense)),
        'paired_increment_median_over_dense_median_ratio': (
            paired_median / dense_median),
        'calibrator_device_median_seconds': device,
        'calibrator_device_over_dense_prefill': device / dense_median,
        'calibrator_host_wait_median_seconds': host_wait,
    }
    assert actual['dense_prefill_seconds'] == dense
    assert actual['fused_prefill_seconds'] == fused
    assert actual['paired_increment_seconds'] == delta
    assert actual['all_next_token_identity'] is True
    assert 'calibrator_host_over_dense_prefill' not in actual
    assert 'diagnostic only' in actual['host_wait_scope']
    for key, value in expected.items():
        if key == 'repeat':
            assert actual[key] == value
        else:
            close(actual[key], value)


def validate_log(length):
    text = (RESULTS / 'logs' / f'prefill_{length}.log').read_text()
    observed = [(int(repeat), arm) for repeat, arm in re.findall(
        rf'PREFILL {length} repeat=(\d+) arm=(dense|fused)', text)]
    expected = []
    for repeat in range(3):
        order = ('dense', 'fused') if repeat % 2 == 0 else ('fused', 'dense')
        expected.extend((repeat, arm) for arm in order)
    assert observed == expected, (length, observed)
    assert f'saved -> {RESULTS}/prefill_{length}.json' in text


def validate_hygiene(receipt):
    hygiene = receipt['gpu_hygiene']
    assert hygiene['status'] == 'clean'
    assert hygiene['pollution_samples'] == 0
    path = Path(hygiene['log'])
    assert path.resolve() == (RESULTS / f"prefill_{receipt['arguments']['context_length']}.json.hygiene.jsonl").resolve()
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert lines
    for sample in lines:
        assert sample['unknown_apps'] == []
        assert sample['active_pid'] in sample['allowed_pids']
        assert sample['owned_apps']
    return len(lines)


def validate_receipt(length):
    path = RESULTS / f'prefill_{length}.json'
    receipt = json.loads(path.read_text())
    assert receipt['schema'] == 'nfa-fused-lse-prefill-increment-v2'
    assert receipt['arguments']['context_length'] == length
    assert receipt['arguments']['repeat'] == 3
    assert receipt['source']['diff_sha256'] == EMPTY_DIFF_SHA256
    subprocess.run(['git', '-C', str(REPO), 'cat-file', '-e',
                    receipt['source']['head'] + '^{commit}'], check=True)
    assert 'does not claim reuse of SDPA internal LSE' in receipt['fusion_boundary']

    input_receipt = receipt['input']
    assert sha256(Path(input_receipt['path'])) == input_receipt['sha256']
    assert 0 < input_receipt['actual_prompt_tokens'] <= length
    model = receipt['model']
    for name, expected in model['metadata_sha256'].items():
        assert sha256(Path(model['path']) / name) == expected

    warmup = receipt['correctness']['warmup_fused']
    legacy = warmup['legacy_validation']
    assert receipt['correctness']['warmup_next_token_identity'] is True
    assert warmup['calls'] == warmup['fused_kernel_calls'] == 36
    assert legacy['status'] == 'PASS'
    assert legacy['layers'] == 36
    assert legacy['gate_decisions_identity'] is True
    assert legacy['max_abs_error'] <= legacy['abs_tolerance'] == 5e-6

    measurements = receipt['measurements']
    assert len(measurements) == 3
    for row in measurements:
        dense = row['dense']
        fused = row['fused']
        assert row['token_identity'] is True
        assert dense['next_token'] == fused['next_token']
        close(row['delta_seconds'], fused['wall_seconds'] - dense['wall_seconds'])
        assert dense['calibrator']['calls'] == 0
        assert dense['calibrator']['fused_kernel_calls'] == 0
        assert fused['calibrator']['calls'] == 36
        assert fused['calibrator']['fused_kernel_calls'] == 36
        assert len(fused['calibrator']['device_ms_per_layer']) == 36
        assert all(value >= 0 for value in fused['calibrator']['device_ms_per_layer'])
        close(fused['calibrator']['device_seconds_total'],
              sum(fused['calibrator']['device_ms_per_layer']) / 1000.0)
        assert 'host_seconds_total' not in fused['calibrator']
        assert fused['calibrator']['host_wait_seconds_total'] >= 0
    validate_summary(receipt['summary'], measurements)
    validate_log(length)
    hygiene_samples = validate_hygiene(receipt)
    summary = receipt['summary']
    return receipt, {
        'context_length': length,
        'actual_prompt_tokens': input_receipt['actual_prompt_tokens'],
        'dense_prefill_median_seconds': summary['dense_prefill_median_seconds'],
        'fused_prefill_median_seconds': summary['fused_prefill_median_seconds'],
        'paired_increment_median_seconds': summary['paired_increment_median_seconds'],
        'paired_increment_over_dense_ratio': summary[
            'paired_increment_median_over_dense_median_ratio'],
        'calibrator_device_median_seconds': summary[
            'calibrator_device_median_seconds'],
        'calibrator_device_over_dense_prefill': summary[
            'calibrator_device_over_dense_prefill'],
        'next_token_identity': True,
        'legacy_gate_identity_layers': 36,
        'legacy_max_abs_error': legacy['max_abs_error'],
        'fused_calls_per_prefill': 36,
        'hygiene_samples': hygiene_samples,
    }


def main():
    rows = []
    receipts = []
    for length in LENGTHS:
        receipt, row = validate_receipt(length)
        receipts.append(receipt)
        rows.append(row)
    heads = {receipt['source']['head'] for receipt in receipts}
    assert len(heads) == 1
    source_head = heads.pop()
    production_paths = [
        'sparse_attn/ops/frozen_gate_triton.py',
        'sparse_attn/policies/frozen_gate.py',
        'benchmark/efficiency/measure_prefill_lse_calibrator.py',
    ]
    delta = subprocess.check_output([
        'git', '-C', str(REPO), 'diff', '--name-only', source_head + '..HEAD',
        '--', *production_paths], text=True).splitlines()
    assert delta == []

    regression_log = RESULTS / 'regression.log'
    regression_text = regression_log.read_text()
    assert '57 passed' in regression_text
    assert ' failed' not in regression_text

    evidence_files = [regression_log]
    for length in LENGTHS:
        evidence_files.extend([
            RESULTS / f'prefill_{length}.json',
            RESULTS / f'prefill_{length}.json.hygiene.jsonl',
            RESULTS / 'logs' / f'prefill_{length}.log',
        ])
    evidence_files.append(Path(__file__).resolve())
    manifest = {
        'schema': 'nfa-fused-lse-prefill-manifest-v1',
        'files': [{
            'path': str(path.relative_to(RESULTS)),
            'bytes': path.stat().st_size,
            'sha256': sha256(path),
        } for path in evidence_files],
    }
    manifest_path = RESULTS / 'data_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    validation = {
        'schema': 'nfa-fused-lse-prefill-validation-v1',
        'status': 'PASS',
        'scope': ('Three real Qwen3-8B RULER prefill receipt sets; independent '
                  'summary recomputation, ABBA logs, model/input hashes, fused '
                  'call counts, legacy gate identity, and GPU hygiene'),
        'fusion_boundary': ('Single-pass fused-LSE calibrator followed by unchanged '
                            'native dense SDPA; no SDPA-internal LSE reuse claimed'),
        'source_head_at_measurement': source_head,
        'implementation_head_at_validation': subprocess.check_output([
            'git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        'production_delta_files_since_measurement': delta,
        'regression': {
            'status': 'PASS',
            'tests_passed': 57,
            'test_files': [
                'tests/test_frozen_gate_profile.py',
                'tests/test_frozen_region_runtime.py',
                'tests/test_frozen_skip_request_guard.py',
            ],
            'log': 'regression.log',
        },
        'results': rows,
        'data_manifest_sha256': sha256(manifest_path),
    }
    (RESULTS / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
