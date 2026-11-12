#!/usr/bin/env python3
"""GLM-4.7-Flash FullKV FlashInfer LongBench v1 英文 16 任务闭集审计。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

DATASETS = [
    'narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa',
    '2wikimqa', 'musique', 'gov_report', 'qmsum', 'multi_news',
    'trec', 'triviaqa', 'samsum', 'passage_count',
    'passage_retrieval_en', 'lcc', 'repobench-p',
]
EXPECTED_COUNTS = {
    'narrativeqa': 200, 'qasper': 200, 'multifieldqa_en': 150,
    'hotpotqa': 200, '2wikimqa': 200, 'musique': 200,
    'gov_report': 200, 'qmsum': 200, 'multi_news': 200,
    'trec': 200, 'triviaqa': 200, 'samsum': 200,
    'passage_count': 200, 'passage_retrieval_en': 200,
    'lcc': 500, 'repobench-p': 500,
}
EXPECTED_COMMIT = 'cfedb3d598ccafb11db7e3d410216c88495dc4ef'
LAYERS = 47


def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sha_payload(value):
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def require(condition, message, failures):
    if not condition:
        failures.append(message)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--request-manifest-output', required=True)
    parser.add_argument('--world-size', type=int, default=5)
    parser.add_argument('--expected-commit', default=EXPECTED_COMMIT)
    parser.add_argument('--recompute-inputs', action='store_true')
    args = parser.parse_args()

    root = Path(args.run_dir)
    repo = Path(args.repo)
    data_root = Path(args.data_root)
    model_root = Path(args.model)
    failures = []
    sys.path.insert(0, str(repo))
    from benchmark.LongBench.eval import sample_score

    tokenizer = None
    prepare_ids = None
    prompts = None
    if args.recompute_inputs:
        from transformers import AutoTokenizer
        from benchmark.LongBench.pred_glm_fullkv import prepare_ids
        tokenizer = AutoTokenizer.from_pretrained(model_root)
        prompts = json.loads((repo / 'benchmark/LongBench/config/dataset2prompt.json').read_text())

    actual_head = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    dirty = subprocess.check_output(
        ['git', 'status', '--porcelain'], cwd=repo, text=True).strip()
    require(actual_head == args.expected_commit,
            f'repo HEAD mismatch: {actual_head}', failures)
    require(not dirty, 'audit repo has uncommitted changes', failures)

    manifests = []
    for rank in range(args.world_size):
        path = root / f'manifest_rank{rank}.json'
        require(path.is_file(), f'missing {path.name}', failures)
        if path.is_file():
            try:
                manifest = json.loads(path.read_text())
                require(manifest.get('rank') == rank,
                        f'manifest rank mismatch {rank}', failures)
                manifests.append(manifest)
            except Exception as exc:
                failures.append(f'manifest {rank}: {type(exc).__name__}: {exc}')
        done = root / f'worker_done_rank{rank}.json'
        require(done.is_file(), f'missing {done.name}', failures)
    require(len(manifests) == args.world_size, 'not all manifests readable', failures)
    if not manifests:
        result = {'status': 'FAIL', 'failures': failures}
        write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False))
        return 1

    identity = manifests[0].get('identity')
    protocol = manifests[0].get('protocol', {})
    recomputed_identity = hashlib.sha256(json.dumps(
        protocol, sort_keys=True).encode()).hexdigest()
    require(identity == recomputed_identity, 'identity does not bind protocol', failures)
    require(all(m.get('identity') == identity for m in manifests),
            'rank identity mismatch', failures)
    require(all(m.get('protocol') == protocol for m in manifests),
            'rank protocol mismatch', failures)
    for rank in range(args.world_size):
        done = root / f'worker_done_rank{rank}.json'
        if done.is_file():
            try:
                require(json.loads(done.read_text()).get('identity') == identity,
                        f'worker done identity mismatch rank={rank}', failures)
            except Exception as exc:
                failures.append(f'worker done {rank}: {type(exc).__name__}: {exc}')

    require(protocol.get('schema') == 'glm-fullkv-longbench-v1',
            'protocol schema mismatch', failures)
    require(protocol.get('backend') == 'flashinfer', 'backend is not FlashInfer', failures)
    require(protocol.get('attention') == 'fullkv', 'attention is not FullKV', failures)
    require(protocol.get('page_size') == 64, 'page size != 64', failures)
    require(protocol.get('seed') == 42, 'seed != 42', failures)
    require(protocol.get('samples') == 0, 'samples != full dataset', failures)
    require(protocol.get('full_dataset') is True, 'full_dataset flag absent', failures)
    require(protocol.get('context_cap') == 31500, 'context cap != 31500', failures)
    require(protocol.get('datasets') == DATASETS, 'dataset protocol mismatch', failures)
    require(protocol.get('dataset_counts') == EXPECTED_COUNTS,
            'dataset counts mismatch', failures)
    require(protocol.get('world_size') == args.world_size, 'world size mismatch', failures)
    require(protocol.get('git_head') == args.expected_commit,
            'protocol git_head mismatch', failures)
    require(protocol.get('dtype') == 'bfloat16', 'dtype mismatch', failures)

    lengths = json.loads((repo / 'benchmark/LongBench/config/dataset2maxlen.json').read_text())
    require(protocol.get('max_new_tokens') == {name: lengths[name] for name in DATASETS},
            'max_new_tokens mismatch', failures)
    for dataset in DATASETS:
        source = data_root / f'{dataset}.jsonl'
        require(source.is_file(), f'missing dataset {source}', failures)
        if source.is_file():
            require(protocol.get('dataset_sha256', {}).get(dataset) == sha_file(source),
                    f'dataset SHA mismatch {dataset}', failures)
        require(sorted(protocol.get('selected_rows', {}).get(dataset, []))
                == list(range(EXPECTED_COUNTS[dataset])),
                f'full rows mismatch {dataset}', failures)
    for relative, expected_sha in protocol.get('source_sha256', {}).items():
        path = repo / relative
        require(path.is_file(), f'missing source {relative}', failures)
        if path.is_file():
            require(sha_file(path) == expected_sha, f'source SHA mismatch {relative}', failures)
    for name, key in [('config.json', 'model_config_sha256'),
                      ('tokenizer_config.json', 'tokenizer_config_sha256')]:
        path = model_root / name
        require(path.is_file(), f'missing model metadata {name}', failures)
        if path.is_file():
            require(protocol.get(key) == sha_file(path), f'{key} mismatch', failures)

    raw_rows = {
        dataset: [json.loads(line) for line in (
            data_root / f'{dataset}.jsonl').read_text().splitlines()]
        for dataset in DATASETS
    }

    tasks = {}
    rank_request_counts = {str(rank): 0 for rank in range(args.world_size)}
    expected_rank_counts = {str(rank): 0 for rank in range(args.world_size)}
    request_manifest = []
    all_scores = []
    dataset_offset = 0
    for dataset in DATASETS:
        selected = protocol.get('selected_rows', {}).get(dataset, [])
        expected_files = {f'{row:04d}.json' for row in selected}
        request_dir = root / 'requests' / dataset
        actual_files = ({path.name for path in request_dir.glob('*.json')}
                        if request_dir.is_dir() else set())
        require(actual_files == expected_files,
                f'request file set mismatch {dataset}: '
                f'missing={len(expected_files-actual_files)} extra={len(actual_files-expected_files)}',
                failures)
        records = []
        for position, row_index in enumerate(selected):
            expected_rank = (dataset_offset + position) % args.world_size
            expected_rank_counts[str(expected_rank)] += 1
            path = request_dir / f'{row_index:04d}.json'
            if not path.is_file():
                failures.append(f'missing request {dataset}:{row_index}')
                continue
            request_manifest.append({
                'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
                'sha256': sha_file(path),
            })
            try:
                record = json.loads(path.read_text())
                prefix = f'{dataset}:{row_index}'
                require(record.get('identity') == identity, f'identity {prefix}', failures)
                require(record.get('status') == 'complete', f'status {prefix}', failures)
                require(record.get('dataset') == dataset, f'dataset {prefix}', failures)
                require(record.get('row_index') == row_index, f'row index {prefix}', failures)
                require(record.get('backend') == 'flashinfer', f'backend {prefix}', failures)
                require(record.get('attention') == 'fullkv', f'attention {prefix}', failures)
                require(record.get('wrapper') == 'BatchDecodeWithPagedKVCacheWrapper',
                        f'wrapper {prefix}', failures)
                require(record.get('page_size') == 64, f'page size {prefix}', failures)
                require(record.get('prefill_append_calls') == LAYERS,
                        f'prefill append calls {prefix}', failures)
                generated = record.get('generated_ids')
                require(isinstance(generated, list) and generated
                        and all(isinstance(token, int) for token in generated),
                        f'generated IDs {prefix}', failures)
                if isinstance(generated, list) and generated:
                    decode_steps = len(generated) - 1
                    require(record.get('generated_tokens') == len(generated),
                            f'generated token count {prefix}', failures)
                    require(len(generated) <= lengths[dataset],
                            f'generated length cap {prefix}', failures)
                    require(record.get('decode_append_calls') == LAYERS * decode_steps,
                            f'decode append calls {prefix}', failures)
                    require(record.get('decode_wrapper_calls') == LAYERS * decode_steps,
                            f'decode wrapper calls {prefix}', failures)
                info = record.get('input', {})
                require(len(info.get('input_sha256', '')) == 64,
                        f'input SHA {prefix}', failures)
                require(info.get('input_tokens') == info.get('material_tokens', 0)
                        + info.get('suffix_tokens', 0),
                        f'material/suffix token sum {prefix}', failures)
                require(isinstance(record.get('prediction'), str),
                        f'prediction {prefix}', failures)
                score = record.get('score')
                require(isinstance(score, (int, float)) and math.isfinite(score)
                        and 0.0 <= score <= 1.0, f'score domain {prefix}', failures)
                wall = record.get('wall_seconds')
                require(isinstance(wall, (int, float)) and math.isfinite(wall) and wall > 0,
                        f'wall seconds {prefix}', failures)

                source_row = raw_rows[dataset][row_index]
                answers = source_row['answers']
                all_classes = source_row.get('all_classes', [])
                require(record.get('answers') == answers, f'answers mismatch {prefix}', failures)
                metric_dataset = 'repobench' if dataset == 'repobench-p' else dataset
                recomputed_score = sample_score(
                    metric_dataset, record['prediction'], answers, all_classes)
                require(math.isclose(score, recomputed_score, rel_tol=0.0, abs_tol=1e-12),
                        f'score recompute {prefix}: stored={score} actual={recomputed_score}',
                        failures)
                if args.recompute_inputs:
                    recomputed_ids, recomputed_info = prepare_ids(
                        tokenizer, source_row, dataset, prompts[dataset], 31500)
                    require(info == recomputed_info, f'input metadata recompute {prefix}', failures)
                    require(record['prediction'] == tokenizer.decode(
                        generated, skip_special_tokens=True),
                        f'prediction decode mismatch {prefix}', failures)
                records.append(record)
                rank_request_counts[str(expected_rank)] += 1
            except Exception as exc:
                failures.append(f'request {dataset}:{row_index}: {type(exc).__name__}: {exc}')
        records.sort(key=lambda record: record['row_index'])
        require(len(records) == EXPECTED_COUNTS[dataset], f'incomplete {dataset}', failures)
        scores = [record['score'] for record in records]
        all_scores.extend(scores)
        tasks[dataset] = {
            'n': len(records),
            'score': 100 * sum(scores) / len(scores) if scores else None,
            'generated_tokens': sum(record['generated_tokens'] for record in records),
            'wall_seconds_sum': sum(record['wall_seconds'] for record in records),
            'prediction_sha256': sha_payload([
                [record['row_index'], record['prediction']] for record in records]),
            'generated_ids_sha256': sha_payload([
                [record['row_index'], record['generated_ids']] for record in records]),
        }
        dataset_offset += len(selected)

    require(rank_request_counts == expected_rank_counts,
            f'rank counts mismatch: actual={rank_request_counts} expected={expected_rank_counts}',
            failures)
    require(sum(rank_request_counts.values()) == 3750,
            'total requests != 3750', failures)
    request_manifest.sort(key=lambda item: item['path'])
    require(len(request_manifest) == 3750, 'request manifest size != 3750', failures)
    manifest_document = {
        'schema': 'glm-fullkv-fi-lb16-request-files-v1',
        'run_identity': identity, 'files': request_manifest,
    }
    write_json(args.request_manifest_output, manifest_document)
    complete = (len(tasks) == len(DATASETS)
                and all(task['n'] == EXPECTED_COUNTS[name] for name, task in tasks.items()))
    result = {
        'schema': 'glm-fullkv-fi-longbench16-audit-v1',
        'status': 'PASS' if not failures else 'FAIL',
        'validation_scope': 'producer-side closure and independent recomputation; not a different-owner acceptance',
        'run_dir': str(root), 'identity': identity,
        'code_revision': actual_head, 'repo_dirty': bool(dirty),
        'rank_request_counts': rank_request_counts,
        'expected_rank_counts': expected_rank_counts,
        'tasks': tasks,
        'macro_avg16': (sum(task['score'] for task in tasks.values()) / len(DATASETS)
                        if complete else None),
        'micro_avg3750': (100 * sum(all_scores) / len(all_scores)
                          if len(all_scores) == 3750 else None),
        'request_manifest': str(args.request_manifest_output),
        'request_manifest_sha256': sha_file(Path(args.request_manifest_output)),
        'request_manifest_ordered_sha256': sha_payload(request_manifest),
        'recomputed_official_scores': len(all_scores),
        'recomputed_tokenizer_inputs': len(all_scores) if args.recompute_inputs else 0,
        'failures': failures,
    }
    write_json(args.output, result)
    print(json.dumps({
        'status': result['status'], 'identity': identity,
        'macro_avg16': result['macro_avg16'], 'micro_avg3750': result['micro_avg3750'],
        'rank_request_counts': rank_request_counts,
        'request_manifest_sha256': result['request_manifest_sha256'],
        'failures': failures,
        'tasks': tasks,
    }, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
