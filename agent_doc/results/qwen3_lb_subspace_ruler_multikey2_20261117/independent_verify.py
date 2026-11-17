#!/usr/bin/env python3
import glob, hashlib, json, sys
from pathlib import Path

ROOT = Path('/home/wangyuanshuo02/qwen3_lb_subspace_20261116')
COLLECTED = ROOT / 'collected_ruler'
REPO = ROOT / 'two-level-attention'
sys.path.insert(0, str(REPO))
from benchmark.RULER.score_ruler import string_match_all

CELLS = {
    'L32768_dim16': ('ruler_multikey2_L32768_dim16', 1, 0),
    'L32768_dim32': ('ruler_multikey2_L32768_dim32', 1, 0),
    'L32768_dim64': ('ruler_multikey2_L32768_dim64', 1, 0),
    'L32768_dim128': ('ruler_multikey2_L32768_dim128', 1, 0),
    'L65536_dim16': ('ruler_multikey2_L65536_dim16', 1, 0),
    'L65536_dim32': ('ruler_multikey2_L65536_dim32', 1, 0),
    'L65536_dim64': ('ruler_multikey2_L65536_dim64', 1, 0),
    'L65536_dim128': ('ruler_multikey2_L65536_dim128', 1, 0),
    'L131072_dim16': ('ruler_multikey2_L131072_dim16', 1, 0),
    'L131072_dim32': ('ruler_multikey2_L131072_dim32', 1, 0),
    'L131072_dim64': ('ruler_multikey2_L131072_dim64', 1, 0),
    'L131072_dim128': ('ruler_multikey2_L131072_dim128', 2, 0),
}
SHARD_DIRS = {
    'L131072_dim128': ['ruler_multikey2_L131072_dim128_s0of2', 'ruler_multikey2_L131072_dim128_s1of2'],
}

def parse_manifest_name(path):
    name = path.name
    # quality_manifest_L<L>_s<idx>of<num>.json
    stem = name[:-5]
    parts = stem.split('_')
    L = int(parts[2][1:])
    shardpart = parts[3]
    idx, num = shardpart[1:].split('of')
    return L, int(idx), int(num)

def find_manifest(cell_dir):
    manifests = list(cell_dir.glob('quality_manifest_L*.json'))
    if len(manifests) != 1:
        raise RuntimeError(f'{cell_dir} manifests={len(manifests)}')
    return manifests[0]

def find_pred(cell_dir):
    preds = list(cell_dir.glob('L*/pred/*.jsonl'))
    if len(preds) != 1:
        raise RuntimeError(f'{cell_dir} pred files={len(preds)}')
    return preds[0]

def check_record(rec, expected_task='niah_multikey_2'):
    pred = rec.get('pred')
    answers = rec.get('answers')
    if not isinstance(pred, str) or not isinstance(answers, list):
        raise RuntimeError('pred/answers invalid')
    recompute = string_match_all([pred], [answers])
    if abs(recompute - rec.get('official_score', -999)) > 1e-9:
        raise RuntimeError(f"official_score mismatch {rec.get('_id')}: {rec.get('official_score')} vs {recompute}")
    answers_sha256 = hashlib.sha256(json.dumps(answers, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if rec.get('answers_sha256') != answers_sha256:
        raise RuntimeError(f"answers_sha256 mismatch {rec.get('_id')}")
    if rec.get('_answers_sha') != answers_sha256[:16]:
        raise RuntimeError(f"_answers_sha mismatch {rec.get('_id')}")
    if rec.get('_id') != f"{expected_task}:{rec.get('source_index')}":
        raise RuntimeError(f"_id/source_index mismatch {rec.get('_id')}")

def check_sidecars(dirname, cell_dir, pred_path, records, L, shard_idx, num_shards, task='niah_multikey_2'):
    receipts = list(cell_dir.glob('L*/pred/*.receipts.json'))
    scores = list(cell_dir.glob('L*/pred/*.scores.json'))
    source_manifests = list(cell_dir.glob('L*/pred/*.source_manifest.json'))
    runcaps = list(cell_dir.glob('L*/pred/*.runcap.json'))
    if len(receipts) != 1 or len(scores) != 1 or len(source_manifests) != 1 or len(runcaps) != 1:
        raise RuntimeError(f'{dirname}: sidecar count mismatch receipts={len(receipts)} scores={len(scores)} srcmanifest={len(source_manifests)} runcap={len(runcaps)}')
    receipts = json.loads(receipts[0].read_text())
    scores = json.loads(scores[0].read_text())
    srcm = json.loads(source_manifests[0].read_text())
    runcap = json.loads(runcaps[0].read_text())
    if len(receipts) != len(records) or len(scores) != len(records):
        raise RuntimeError(f'{dirname}: sidecar population mismatch')
    for i, (rec, rcp, sc) in enumerate(zip(records, receipts, scores)):
        expected_idx = shard_idx + i * num_shards
        if rec.get('source_index') != expected_idx:
            raise RuntimeError(f'{dirname}: row order index {rec.get("source_index")} != {expected_idx} at {i}')
        if rcp.get('task') != task or rcp.get('context_length') != L or rcp.get('source_index') != expected_idx:
            raise RuntimeError(f'{dirname}: receipt identity mismatch at {i}')
        if rcp.get('sample_id') != rec.get('_id') or rcp.get('input') != rec.get('input'):
            raise RuntimeError(f'{dirname}: receipt input/sample mismatch at {i}')
        if rcp.get('generated_tokens') != rec.get('generated_tokens'):
            raise RuntimeError(f'{dirname}: receipt generated_tokens mismatch at {i}')
        if rcp.get('stop_reason') not in {'eos', 'max_new_tokens'}:
            raise RuntimeError(f'{dirname}: invalid stop_reason at {i}')
        if rec.get('output_truncated') != (rcp.get('stop_reason') == 'max_new_tokens'):
            raise RuntimeError(f'{dirname}: output_truncated mismatch at {i}')
        if sc.get('_id') != rec.get('_id') or sc.get('source_index') != expected_idx or sc.get('answers_sha256') != rec.get('answers_sha256'):
            raise RuntimeError(f'{dirname}: scores identity mismatch at {i}')
        if abs(sc.get('official_score', -999) - rec.get('official_score')) > 1e-9:
            raise RuntimeError(f'{dirname}: scores official_score mismatch at {i}')
    task_block = srcm.get(task)
    if not isinstance(task_block, dict):
        raise RuntimeError(f'{dirname}: source_manifest missing task block')
    if len(task_block.get('ids', [])) != len(records):
        raise RuntimeError(f'{dirname}: source_manifest ids population mismatch')
    for i, (_id, rec) in enumerate(zip(task_block['ids'], records)):
        if _id != rec.get('_id'):
            raise RuntimeError(f'{dirname}: source_manifest id mismatch at {i}')
    if task_block.get('answers_sha', {}).get(records[0]['_id']) is None:
        raise RuntimeError(f'{dirname}: source_manifest answers_sha missing')
    if runcap.get('status') != 'complete' or runcap.get('result_lines') != len(records):
        raise RuntimeError(f'{dirname}: runcap status/lines mismatch')
    if runcap.get('producer', {}).get('git_commit') != '9f3e8057acd528a904aa8883b830d541b389e39f':
        raise RuntimeError(f'{dirname}: runcap git_commit mismatch')
    if runcap.get('sample_sharding', {}).get('rank') != shard_idx or runcap.get('sample_sharding', {}).get('world_size') != num_shards:
        raise RuntimeError(f'{dirname}: runcap sharding mismatch')
    actual_sha = hashlib.sha256(pred_path.read_bytes()).hexdigest()
    if runcap.get('result_sha256') != actual_sha:
        raise RuntimeError(f'{dirname}: runcap result_sha256 mismatch')

def verify_dir(dirname, expected_num_shards=None, expected_shard=None):
    cell_dir = COLLECTED / dirname
    manifest_path = find_manifest(cell_dir)
    L, shard_idx, num_shards = parse_manifest_name(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    pred_path = find_pred(cell_dir)
    records = [json.loads(line) for line in pred_path.read_text().splitlines()]
    if len(records) != manifest['rows']:
        raise RuntimeError(f'{dirname}: rows pred={len(records)} manifest={manifest["rows"]}')
    if manifest['shard_index'] != shard_idx or manifest['num_shards'] != num_shards:
        raise RuntimeError(f'{dirname}: manifest shard metadata inconsistent')
    if expected_num_shards is not None and num_shards != expected_num_shards:
        raise RuntimeError(f'{dirname}: expected shards {expected_num_shards}, got {num_shards}')
    if expected_shard is not None and shard_idx != expected_shard:
        raise RuntimeError(f'{dirname}: expected shard {expected_shard}, got {shard_idx}')
    if manifest['context_length'] != L:
        raise RuntimeError(f'{dirname}: context length mismatch {L} vs manifest {manifest["context_length"]}')
    if manifest['task'] != 'niah_multikey_2':
        raise RuntimeError(f'{dirname}: task mismatch {manifest["task"]}')
    if manifest['verification'] != 'pending-independent-verification':
        raise RuntimeError(f'{dirname}: unexpected verification status')
    src = manifest['runtime']['source']
    if src['head'] != '9f3e8057acd528a904aa8883b830d541b389e39f':
        raise RuntimeError(f'{dirname}: source head mismatch {src["head"]}')
    selector = manifest['runtime']['nfa_selector_config']
    if selector.get('level2_topk') != 1024:
        raise RuntimeError(f'{dirname}: level2_topk != 1024')
    if selector.get('level1_topk') != 128:
        raise RuntimeError(f'{dirname}: level1_topk != 128')
    seen = set()
    for rec in records:
        check_record(rec)
        idx = rec['source_index']
        if idx in seen:
            raise RuntimeError(f'{dirname}: duplicate source_index {idx}')
        seen.add(idx)
        if idx % num_shards != shard_idx:
            raise RuntimeError(f'{dirname}: index {idx} does not belong to shard {shard_idx}/{num_shards}')
    check_sidecars(dirname, cell_dir, pred_path, records, L, shard_idx, num_shards)
    return {
        'dirname': dirname,
        'L': L,
        'shard': shard_idx,
        'num_shards': num_shards,
        'rows': len(records),
        'indices': sorted(seen),
        'manifest_score': manifest['official_score'],
        'recomputed_score': sum(rec['official_score'] for rec in records) / len(records),
        'pred': str(pred_path),
        'manifest': str(manifest_path),
    }

def main():
    rows = []
    for cell, (base, num_shards, _) in CELLS.items():
        dirs = SHARD_DIRS.get(cell, [base])
        shard_infos = []
        for dirname in dirs:
            info = verify_dir(dirname, expected_num_shards=num_shards)
            shard_infos.append(info)
            rows.append(info)
        if num_shards > 1:
            all_idx = []
            for info in shard_infos:
                all_idx.extend(info['indices'])
            if sorted(all_idx) != list(range(100)):
                missing = [i for i in range(100) if i not in all_idx]
                dup = sorted({i for i in all_idx if all_idx.count(i) > 1})
                raise RuntimeError(f'{cell}: shard merge invalid missing={missing} dup={dup}')
    print(json.dumps(rows, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
