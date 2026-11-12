#!/usr/bin/env python3
"""从收割后的原始请求重建逐文件清单和精简质量汇总。"""
import hashlib
import json
from pathlib import Path

RESULT = Path(__file__).resolve().parent
RAW = Path('/home/wangyuanshuo02/ak_verify_results/glm_longbench_16en_nope_last32_20261109_merged')
HOSTS = {
    0: 'pc40/10.238.139.251', 1: 'pc40/10.238.139.251',
    2: 'pc31/10.245.58.57', 3: 'pc31/10.245.58.57',
    4: 'pc32/10.245.84.251', 5: 'pc32/10.245.84.251',
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    audit = json.loads((RESULT / 'audit.json').read_text())
    manifest = json.loads((RESULT / 'manifest_rank0.json').read_text())
    protocol = manifest['protocol']
    positions = {
        (dataset, row): index
        for index, (dataset, row) in enumerate(
            (dataset, row)
            for dataset in protocol['datasets']
            for row in protocol['selected_rows'][dataset]
        )
    }
    files = []
    for path in sorted((RAW / 'requests').glob('*/*.json')):
        dataset = path.parent.name
        row = int(path.stem)
        rank = positions[(dataset, row)] % 6
        record = json.loads(path.read_text())
        assert record['status'] == 'complete'
        assert set(record['arms']) == {'full', 'native_full'}
        files.append({
            'path': str(path.relative_to(RAW)),
            'dataset': dataset,
            'row': row,
            'rank': rank,
            'source_host': HOSTS[rank],
            'bytes': path.stat().st_size,
            'sha256': sha256(path),
        })
    assert len(files) == 3750
    request_manifest = {
        'schema': 'glm-last32-lbv1-request-sha256-v1',
        'raw_root': str(RAW),
        'request_count': len(files),
        'total_bytes': sum(item['bytes'] for item in files),
        'source_hosts': HOSTS,
        'files': files,
    }
    atomic_json(RESULT / 'request_manifest.json', request_manifest)

    harvest_files = []
    for pattern in ('manifest_rank*.json', 'worker_done_rank*.json',
                    'heartbeat_rank*.json', 'final_audit.json'):
        for path in sorted(RAW.glob(pattern)):
            harvest_files.append({
                'path': path.name,
                'bytes': path.stat().st_size,
                'sha256': sha256(path),
            })
    assert len(harvest_files) == 19
    atomic_json(RESULT / 'harvest_manifest.json', {
        'schema': 'glm-last32-lbv1-harvest-sha256-v1',
        'raw_root': str(RAW),
        'file_count': len(harvest_files),
        'files': harvest_files,
    })

    scores = {
        dataset: {
            'nfa_full_region': round(values['full'] * 100, 12),
            'native_full': round(values['native_full'] * 100, 12),
        }
        for dataset, values in audit['tasks'].items()
    }
    runner = RESULT / 'run_glm_longbench_16en_last32_20261109.sh'
    wrapper = RESULT / 'glm_longbench_16en_frozen_20261108.py'
    auditor = RESULT / 'audit_glm_longbench_16en_20261108.py'
    summary = {
        'schema': 'glm-last32-lbv1-full-region-quality-summary-v1',
        'status': 'pending-independent-verification',
        'date': '2026-11-12',
        'model': 'GLM-4.7-Flash',
        'dataset': 'LongBench v1 English 16 tasks',
        'run_identity': audit['identity'],
        'code_revision': '6760cc495f554ffa85e8eb6d6e552ce02911f457',
        'code_snapshot': '/tmp/tli-glm-jssa-prod-20261106',
        'method_scope': {
            'nfa_full_region': (
                'glm_nope_last32 seed=0; thresholds=[]; 47 layers keep; '
                'near/far/both skip=0; CUDA Graph decode; BS=1'),
            'native_full': 'HF SDPA prefill/decode with DynamicCache; eager; BS=1',
            'not_in_scope': (
                'This is not the threshold=0.1 near/far-skip NFA arm. '
                'Request wall_seconds includes quality-audit and per-request graph setup; '
                'native_full wall_seconds is NaN. No TPOT or speedup claim.'),
        },
        'population': {
            'requests': 3750,
            'requests_per_rank': 625,
            'rows': protocol['dataset_counts'],
            'arms_per_request': ['full', 'native_full'],
        },
        'score_protocol': (
            'Official LongBench sample_score per row and arm; arithmetic mean per task, '
            'then unweighted macro average over 16 tasks; seed=42, context_cap=31500'),
        'execution': {
            'rank0': 'pc40 GPU0', 'rank1': 'pc40 GPU1',
            'rank2': 'pc31 GPU0', 'rank3': 'pc31 GPU1',
            'rank4': 'pc32 GPU0', 'rank5': 'pc32 GPU1',
            'hardware': 'H20-3e',
        },
        'scores': scores,
        'macro_avg16': {
            'nfa_full_region': audit['macro_full'] * 100,
            'native_full': audit['macro_native_full'] * 100,
            'delta_nfa_minus_native': (audit['macro_full'] - audit['macro_native_full']) * 100,
        },
        'micro_avg3750': {
            'nfa_full_region': audit['micro_full'] * 100,
            'native_full': audit['micro_native_full'] * 100,
            'delta_nfa_minus_native': (audit['micro_full'] - audit['micro_native_full']) * 100,
        },
        'producer_side_audit': {
            'status': audit['status'],
            'closed_set_requests': 3750,
            'datasets': 16,
            'rank_request_counts': audit['rank_request_counts'],
            'official_arm_scores_recomputed': 7500,
            'tokenizer_inputs_recomputed': 3750,
            'nfa_47_layer_no_skip_receipts_checked': 3750,
            'failures': audit['failures'],
            'scope': 'Independent read-only recomputation by the producer session; different-owner acceptance remains pending',
            'audit_sha256': sha256(RESULT / 'audit.json'),
            'audit_script_sha256': sha256(auditor),
            'request_manifest_sha256': sha256(RESULT / 'request_manifest.json'),
            'harvest_manifest_sha256': sha256(RESULT / 'harvest_manifest.json'),
            'launcher_sha256': sha256(runner),
            'worker_wrapper_sha256': sha256(wrapper),
            'build_script_sha256': sha256(Path(__file__).resolve()),
        },
        'raw_evidence_root': str(RAW),
        'delivery_scope': (
            'Versioned artifacts contain the compact summary, complete audit, six run manifests, '
            'completion/heartbeat receipts, per-file SHA256 manifests and reproduction scripts. '
            'Raw per-request JSON remains at raw_evidence_root.'),
    }
    atomic_json(RESULT / 'summary.json', summary)
    print(json.dumps({
        'requests': len(files),
        'harvest_files': len(harvest_files),
        'macro_avg16': summary['macro_avg16'],
        'request_manifest_sha256': summary['producer_side_audit']['request_manifest_sha256'],
        'harvest_manifest_sha256': summary['producer_side_audit']['harvest_manifest_sha256'],
        'audit_sha256': summary['producer_side_audit']['audit_sha256'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
