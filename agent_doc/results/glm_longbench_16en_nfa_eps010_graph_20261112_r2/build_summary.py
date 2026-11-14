#!/usr/bin/env python3
"""从独立merged原始请求重建SHA256清单和精简质量汇总。"""
import hashlib
import json
from pathlib import Path

RESULT = Path(__file__).resolve().parent
RAW = Path('/home/wangyuanshuo02/ak_verify_results/glm_longbench_16en_nfa_eps010_graph_20261112_r2_merged')
HOSTS = {
    0: 'pc40/10.238.139.251', 1: 'pc40/10.238.139.251',
    2: 'pc31/10.245.58.57', 3: 'pc31/10.245.58.57',
    4: 'pc32/10.245.84.251', 5: 'pc32/10.245.84.251',
    6: 'pc02/10.238.122.187', 7: 'pc02/10.238.122.187',
}
EXPECTED_RANK_COUNTS = {str(rank): 469 if rank < 6 else 468 for rank in range(8)}


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
    assert audit['status'] == 'PASS' and not audit['failures']
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
        rank = positions[(dataset, row)] % 8
        record = json.loads(path.read_text())
        assert record['status'] == 'complete'
        assert record['identity'] == audit['identity']
        assert set(record['arms']) == {'dual_0.1'}
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
        'schema': 'glm-nfa-eps010-graph-lbv1-request-sha256-v1',
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
    assert len(harvest_files) == 25
    atomic_json(RESULT / 'harvest_manifest.json', {
        'schema': 'glm-nfa-eps010-graph-lbv1-harvest-sha256-v1',
        'raw_root': str(RAW),
        'file_count': len(harvest_files),
        'files': harvest_files,
    })

    scores = {
        dataset: {
            'dual_0.1': round(values['dual_0.1'] * 100, 12),
            'gate_counts': values['gate_counts'],
            'cuda_graph': values['cuda_graph'],
        }
        for dataset, values in audit['tasks'].items()
    }
    runner = RESULT / 'run_glm_nfa_eps010_graph_20261112.sh'
    wrapper = RESULT / 'glm_nfa_eps010_graph_frozen_20261112.py'
    auditor = RESULT / 'audit_glm_nfa_eps010_graph_20261112.py'
    patch = RESULT / 'glm_nfa_eps010_graph_20261112_r2.patch'
    summary = {
        'schema': 'glm-nfa-eps010-graph-lbv1-quality-summary-v1',
        'status': 'pending-independent-verification',
        'date': '2026-11-14',
        'model': 'GLM-4.7-Flash',
        'dataset': 'LongBench v1 English 16 tasks',
        'run_identity': audit['identity'],
        'code_revision': audit['git_head'],
        'worktree_diff_sha256': audit['worktree_diff_sha256'],
        'code_snapshot': {
            'local_producer_audit': '/tmp/tli-glm-nfa-eps010-graph-20261112',
            'remote_workers': '/tmp/tli-glm-nfa-eps010-graph-20261112-r2',
        },
        'method_scope': {
            'dual_0.1': (
                'glm_nope_last32 seed=0; threshold=0.1; dual near/far gate; '
                'real region skip; CUDA Graph decode; BS=1; eager fallback=0'),
            'not_in_scope': (
                'Single NFA arm only; no native FullKV or full-region arm in this run. '
                'No TPOT or speedup claim. Producer-side recomputation is not '
                'different-owner independent acceptance.'),
        },
        'population': {
            'requests': 3750,
            'requests_per_rank': EXPECTED_RANK_COUNTS,
            'rows': protocol['dataset_counts'],
            'arms_per_request': ['dual_0.1'],
        },
        'score_protocol': (
            'Official LongBench sample_score per row; arithmetic mean per task, '
            'then unweighted macro average over 16 tasks; seed=42, context_cap=31500'),
        'execution': {
            'rank0': 'pc40 GPU0', 'rank1': 'pc40 GPU1',
            'rank2': 'pc31 GPU0', 'rank3': 'pc31 GPU1',
            'rank4': 'pc32 GPU0', 'rank5': 'pc32 GPU1',
            'rank6': 'pc02 GPU0', 'rank7': 'pc02 GPU1',
            'hardware': 'H20-3e',
        },
        'scores': scores,
        'macro_avg16': {'dual_0.1': audit['macro_dual_0.1'] * 100},
        'micro_avg3750': {'dual_0.1': audit['micro_dual_0.1'] * 100},
        'gate_totals': audit['gate_totals'],
        'graph_totals': audit['graph_totals'],
        'producer_side_audit': {
            'status': audit['status'],
            'closed_set_requests': 3750,
            'datasets': 16,
            'rank_request_counts': audit['rank_request_counts'],
            'official_scores_recomputed': audit['official_scores_recomputed'],
            'tokenizer_inputs_recomputed': audit['tokenizer_inputs_recomputed'],
            'nfa_47_layer_receipts_checked': 3750,
            'gate_decisions_checked': 3750 * 47,
            'failures': audit['failures'],
            'scope': audit['validation_scope'],
            'audit_sha256': sha256(RESULT / 'audit.json'),
            'audit_script_sha256': sha256(auditor),
            'request_manifest_sha256': sha256(RESULT / 'request_manifest.json'),
            'harvest_manifest_sha256': sha256(RESULT / 'harvest_manifest.json'),
            'launcher_sha256': sha256(runner),
            'worker_wrapper_sha256': sha256(wrapper),
            'patch_sha256': sha256(patch),
            'build_script_sha256': sha256(Path(__file__).resolve()),
        },
        'raw_evidence_root': str(RAW),
        'delivery_scope': (
            'Versioned artifacts contain the compact summary, complete audit, eight run '
            'manifests, completion/heartbeat receipts, per-file SHA256 manifests and '
            'reproduction scripts. Raw per-request JSON remains at raw_evidence_root.'),
    }
    atomic_json(RESULT / 'summary.json', summary)
    print(json.dumps({
        'requests': len(files),
        'harvest_files': len(harvest_files),
        'macro_avg16': summary['macro_avg16'],
        'micro_avg3750': summary['micro_avg3750'],
        'gate_totals': summary['gate_totals'],
        'graph_totals': summary['graph_totals'],
        'request_manifest_sha256': summary['producer_side_audit']['request_manifest_sha256'],
        'harvest_manifest_sha256': summary['producer_side_audit']['harvest_manifest_sha256'],
        'audit_sha256': summary['producer_side_audit']['audit_sha256'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
