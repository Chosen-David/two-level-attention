"""Independent verification for aligned glm_nope_last32 full-task raw records."""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

from benchmark.LongBench.prefill_frozen_skip import prepare_ids, sha_file, ROOT, SCREEN_TASKS
from benchmark.LongBench.eval import sample_score
from benchmark.RULER.score_ruler import string_match_all

ARMS = ('dense', 'selected')
RULER_TASK = 'niah_multikey_2'
SOURCE_COMMIT = '42821f81a4739d05d8bc81e1d4cc58307e34bb7e'


def expected_counts():
    return {RULER_TASK: 100, 'qasper': 200, 'hotpotqa': 200,
            'musique': 200, 'gov_report': 200, 'repobench-p': 500}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--ruler', required=True)
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--runner-script', default='')
    args = ap.parse_args()

    run = Path(args.run_dir)
    source = Path(args.source)
    model = Path(args.model)
    data = Path(args.data)
    ruler = Path(args.ruler)
    checks = {}

    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True).strip()
    checks['source_commit'] = commit
    checks['source_dirty'] = dirty
    assert commit == SOURCE_COMMIT
    assert not dirty

    shared = json.loads((run / 'protocol.json').read_text())
    identity = shared['identity']
    protocol = shared['protocol']
    assert hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() == identity
    assert protocol.get('source_commit') == SOURCE_COMMIT
    assert set(protocol['arms']) == set(ARMS)
    checks['protocol_identity_recomputed'] = True

    source_hash_checks = 0
    for rel, expected_hash in protocol['source'].items():
        actual = sha_file(source / rel)
        assert actual == expected_hash, rel
        source_hash_checks += 1
    checks['source_hash_checks'] = source_hash_checks

    script_local = Path(args.runner_script or 'agent_doc/results/glm_nope_last32_ruler_aligned_20261101/scripts/glm_nope_last32_full_aligned.py')
    if script_local.exists():
        checks['runner_script_sha256'] = sha_file(script_local)
        assert sha_file(script_local) == protocol['script_sha256']

    model_checks = 0
    for name, metadata in protocol['model_manifest'].items():
        p = model / name
        assert p.exists(), name
        assert p.stat().st_size == metadata['size'], name
        if not name.endswith('.safetensors'):
            assert sha_file(p) == metadata['sha256'], name
        model_checks += 1
    checks['model_file_checks'] = model_checks

    for task, expected_hash in protocol['data'].items():
        if task == RULER_TASK:
            actual = sha_file(ruler)
        else:
            actual = sha_file(data / f'{task}.jsonl')
        assert actual == expected_hash, task
    checks['dataset_hash_checks'] = len(protocol['data'])

    tokenizer = AutoTokenizer.from_pretrained(str(model))
    prompts = json.loads((ROOT / 'benchmark/LongBench/config/dataset2prompt.json').read_text())
    lengths = json.loads((ROOT / 'benchmark/LongBench/config/dataset2maxlen.json').read_text())
    dataset_rows = {}
    ruler_rows = [json.loads(x) for x in ruler.read_text().splitlines()]
    dataset_rows[RULER_TASK] = ruler_rows
    for task in SCREEN_TASKS:
        dataset_rows[task] = [json.loads(x) for x in (data / f'{task}.jsonl').read_text().splitlines()]

    total = 0
    score_checks = 0
    for task, expected_count in expected_counts().items():
        files = sorted((run / 'requests' / task).glob('*.json'))
        assert len(files) == expected_count, (task, len(files))
        rows = dataset_rows[task]
        for path in files:
            rec = json.loads(path.read_text())
            total += 1
            row_id = rec['row']
            assert rec['identity'] == identity
            assert rec['status'] == 'complete'
            assert set(rec['arms']) == set(ARMS)
            row = rows[row_id]
            assert rec['source_index'] == row.get('index', row_id)
            assert rec['answers'] == (row['outputs'] if task == RULER_TASK else row['answers'])
            if task == RULER_TASK:
                ids = tokenizer(row['input'], add_special_tokens=True).input_ids
            else:
                ids, meta = prepare_ids(tokenizer, row, task, prompts[task], 31500)
                assert rec['input'] == meta, (task, row_id)
            input_sha = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            assert rec['input']['input_sha256'] == input_sha, (task, row_id)

            for arm in ARMS:
                value = rec['arms'][arm]
                assert tokenizer.decode(value['generated_ids'], skip_special_tokens=True) == value['prediction']
                assert tokenizer.decode(value['generated_ids']) == value['raw_text']
                assert len(value['generated_ids']) == value['generated_tokens']
                if task == RULER_TASK:
                    expected_score = string_match_all([value['prediction']], [rec['answers']])
                else:
                    metric_ds = 'repobench' if task == 'repobench-p' else task
                    expected_score = 100 * sample_score(metric_ds, value['prediction'], rec['answers'],
                                                        row.get('all_classes', []))
                if abs(float(value['score_pct']) - float(expected_score)) > 1e-6:
                    raise AssertionError((task, row_id, arm, value['score_pct'], expected_score))
                score_checks += 1
                if arm == 'selected':
                    summary = value.get('tli_audit_summary', {})
                    assert summary.get('status') == 'complete'
                    gates = summary.get('gate_counts', {})
                    assert gates.get('keep') == 47 and gates.get('both') == 0

    done_files = sorted(run.glob('worker_done_rank*.json'))
    assert len(done_files) == 7
    for p in done_files:
        d = json.loads(p.read_text())
        assert d['identity'] == identity
        assert d['failed_seen'] == []
    checks.update(total_request_files=total, arm_score_checks=score_checks,
                  worker_done_files=len(done_files), failed_seen=0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    review = {
        'schema': 'glm-nope-last32-full-aligned-independent-verification-v1',
        'status': 'raw_complete_verified_pending_scientific_review',
        'identity': identity,
        'source_commit': SOURCE_COMMIT,
        'checks': checks,
        'scope': 'Protocol identity, source/model/dataset hashes, input tokenization, answers, generated text, and scores independently verified. Raw remains pending for any benchmark/paper acceptance review.',
    }
    out.write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps(review, indent=2))


if __name__ == '__main__':
    main()
