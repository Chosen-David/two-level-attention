"""CPU fixed-QK O3 scoring diagnostic; never a task-quality or production-parity test.

Consumes existing collector k[S,Hkv,128], q[Nq,Hq,128], qpos[Nq].
NPZ needs only numpy; existing PT files additionally need torch (CPU, weights_only).
Full causal support, global TopK, no protected/quota policy: explicitly a scoring
control, not replay of the production L1/L2 selector. See o3_retrieval.md.
"""
import argparse
import ctypes
import os
import re
import tempfile
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from figure_data_contract import label


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def coordinates(kind, seed=None):
    # Same split-half pair relation as E85d/e85e; no arbitrary scalar random.
    if kind == 'tail':
        pairs = np.arange(48, 64)
    elif kind == 'high':
        pairs = np.arange(16)
    elif kind == 'paired_random':
        pairs = np.sort(np.random.default_rng(seed).choice(64, 16, replace=False))
    else:
        raise ValueError('unknown coordinate policy')
    return pairs, np.concatenate([pairs, pairs + 64])


def qdq(k):
    """Ties-to-even on the float32-computed quotient, not exact-real half ties."""
    lo = k.min(axis=-1, keepdims=True)
    scale = np.maximum(k.max(axis=-1, keepdims=True) - lo, np.float32(1e-9)) / 15
    return np.clip(np.rint((k - lo) / scale), 0, 15) * scale + lo


def probabilities(q, k, coord=None):
    groups, hq = k.shape[1], q.shape[0]
    require(hq % groups == 0, 'Q heads must be divisible by KV heads')
    if coord is not None:
        q, k = q[:, coord], qdq(k[:, :, coord])
    scores = np.einsum('hgd,shd->hgs', q.reshape(groups, hq // groups, -1), k)
    scores *= np.float32(128 ** -.5)
    require(np.isfinite(scores).all(), 'nonfinite logits')
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=-1, keepdims=True)
    return p, p.mean(axis=1)  # softmax per Q-head FIRST, then GQA mean


def topk(scores, budget):
    # Stable sort: ties use lowest original token ID; caller supplies ascending IDs.
    return np.argsort(-scores, axis=-1, kind='stable')[:, :min(budget, scores.shape[-1])]


def measure(q, k, budget, coord, protected=None):
    ref_heads, ref = probabilities(q, k)
    _, approx = probabilities(q, k, coord)
    selected, oracle = topk(approx, budget), topk(ref, budget)
    rows = []
    for group, (ids, target) in enumerate(zip(selected, oracle)):
        overlap = len(set(ids) & set(target))
        rows.append({'kv_head': group, 'selected_ids': ids.tolist(),
                     'reference_ids': target.tolist(), 'actual_k': len(ids),
                     'recall': overlap / len(target),
                     'jaccard': overlap / (len(ids) + len(target) - overlap),
                     'retained_mass': float(ref[group, ids].sum()),
                     'worst_q_head_mass': float(ref_heads[group][:, ids].sum(axis=-1).min())})
    if protected is not None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from exp.paper.observations import mass_partition
        for group, row in enumerate(rows):
            row.update(mass_partition(ref[group], row['selected_ids'], protected))
    return rows


def load_trace(path, expected=None):
    raw = path.read_bytes()
    require(expected is None or hashlib.sha256(raw).hexdigest() == expected, 'trace hash mismatch')
    stream = io.BytesIO(raw)
    if path.suffix == '.npz':
        with np.load(stream, allow_pickle=False) as data:
            result = {key: data[key] for key in ('q', 'k', 'qpos')}
            result['_source_bytes'] = len(raw)
            return result
    require(path.suffix == '.pt', 'trace must be npz or pt')
    import torch
    data = torch.load(stream, map_location='cpu', weights_only=True)
    require(all(torch.is_floating_point(data[key]) for key in ('q', 'k')), 'Q/K must be real floating tensors')
    result = {key: data[key].float().numpy() if key != 'qpos' else data[key].numpy()
              for key in ('q', 'k', 'qpos')}
    result['_source_bytes'] = len(raw)
    result['_original_dtypes'] = {key: str(data[key].dtype) for key in ('q', 'k')}
    return result


def check_trace(data):
    q, k, pos = data['q'], data['k'], data['qpos']
    require(np.issubdtype(q.dtype, np.floating) and np.issubdtype(k.dtype, np.floating), 'Q/K must be real floating arrays')
    require(q.ndim == k.ndim == 3 and q.shape[-1] == k.shape[-1] == 128, 'expected D128 rank-3 Q/K')
    require(all(x > 0 for x in q.shape + k.shape), 'empty Q/K')
    require(q.shape[1] % k.shape[1] == 0, 'invalid GQA geometry')
    require(pos.shape == (q.shape[0],) and np.issubdtype(pos.dtype, np.integer), 'invalid qpos')
    require(len(set(pos.tolist())) == len(pos) and ((pos >= 0) & (pos < len(k))).all(), 'duplicate/out-of-range qpos')
    require(np.isfinite(q).all() and np.isfinite(k).all(), 'nonfinite Q/K')


def check_protocol(p):
    for key in ('run_id', 'model_revision', 'dataset_revision', 'producer_commit', 'split', 'rope_receipt_sha256'):
        label(p.get(key), key)
        require(re.fullmatch(r'[A-Za-z0-9_.:-]+', p[key]) is not None, 'identity must use a path-free public token')
    require(len(p['rope_receipt_sha256']) == 64 and all(c in '0123456789abcdef' for c in p['rope_receipt_sha256']), 'invalid RoPE receipt hash')
    require(isinstance(p.get('rope_receipt_path'), str) and p['rope_receipt_path'], 'missing RoPE receipt path')
    require(p.get('rope_layout') == 'split_half_128_descending_frequency', 'verify actual RoPE layout first')
    require(p.get('data_kind') in ('synthetic', 'observational'), 'declare synthetic or observational data_kind')
    require(p.get('query_phase') == 'prefill', 'current collector supports prompt-only trace')
    require(p.get('support') == 'all_causal_global_topk', 'unsupported support policy')
    require(p.get('quantization') == 'minmax4_k_float32', 'unsupported quantization')
    for key in ('budget', 'bootstrap_repeats', 'bootstrap_seed'):
        require(type(p.get(key)) is int and p[key] >= (0 if key == 'bootstrap_seed' else 1), 'invalid ' + key)
    seeds = p.get('pair_seeds')
    require(isinstance(seeds, list) and len(seeds) >= 2 and all(type(s) is int and s >= 0 for s in seeds) and len(set(seeds)) == len(seeds), 'freeze >=2 unique pair seeds')
    require(isinstance(p.get('traces'), list) and p['traces'], 'no traces')
    seen = set()
    sample_identity = {}
    trace_identity = {}
    input_documents = {}
    for t in p['traces']:
        for key in ('path', 'sha256', 'sample_id', 'document_id', 'input_ids_sha256', 'layer', 'query_indices'):
            require(key in t, 'trace missing ' + key)
        label(t['sample_id'], 'sample_id')
        label(t['document_id'], 'document_id')
        require(all(re.fullmatch(r'[A-Za-z0-9_.:-]+', t[key]) for key in ('sample_id', 'document_id')), 'use anonymous path-free IDs')
        require(type(t['layer']) is int and t['layer'] >= 0, 'invalid layer')
        identity_fields = (t['document_id'], t['input_ids_sha256'])
        require(t['sample_id'] not in sample_identity or sample_identity[t['sample_id']] == identity_fields, 'conflicting sample identity')
        sample_identity[t['sample_id']] = identity_fields
        identity = (t['sample_id'], t['layer'])
        require(identity not in seen, 'duplicate sample/layer')
        seen.add(identity)
        for field in ('sha256', 'input_ids_sha256'):
            require(isinstance(t[field], str) and len(t[field]) == 64 and all(c in '0123456789abcdef' for c in t[field]), 'invalid hash ' + field)
        binding = (t['sample_id'], t['document_id'], t['layer'], t['input_ids_sha256'])
        require(t['sha256'] not in trace_identity or trace_identity[t['sha256']] == binding, 'duplicate trace with conflicting identity')
        trace_identity[t['sha256']] = binding
        input_hash = t['input_ids_sha256']
        require(input_hash not in input_documents or input_documents[input_hash] == t['document_id'], 'same input assigned to different documents')
        input_documents[input_hash] = t['document_id']
        qi = t['query_indices']
        require(isinstance(qi, list) and qi and all(type(i) is int and i >= 0 for i in qi) and len(set(qi)) == len(qi), 'invalid query_indices')


def summarize(rows, p):
    """Equal-document mean; within document equal weight over recorded observations.

    Each seed stays a separate arm: bootstrap interval describes document sampling,
    NOT seed uncertainty. Same bootstrap seed pairs resamples across arms.
    """
    output = []
    for arm in sorted({r['arm_id'] for r in rows}):
        grouped = {}
        for r in rows:
            if r['arm_id'] == arm:
                grouped.setdefault(r['document_id'], []).append(r['retained_mass'])
        values = np.array([np.mean(grouped[d]) for d in sorted(grouped)])
        require(len(values) >= 2, 'need >=2 independent documents for interval')
        rng = np.random.default_rng(p['bootstrap_seed'])
        boot = [np.mean(values[rng.integers(len(values), size=len(values))]) for _ in range(p['bootstrap_repeats'])]
        low, high = np.quantile(boot, [.025, .975])
        output.append(dict(figure_id='O3', panel_id='l2_scoring_diagnostic', arm_id=arm,
            reference_id='raw_full128_per_q_softmax', run_mode='fixed-QK', metric_id='retained_mass',
            unit='probability', estimate=float(values.mean()), ci_low=float(low), ci_high=float(high),
            n_independent_seq=len(values), aggregation='equal_document_mean', transform='none',
            source_run_id=p['run_id'], source_field='selection_metrics.jsonl:retained_mass'))
    return output


def publish_directory(stage, output):
    # Linux exclusive atomic directory publication; never replace another writer.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    require(rename is not None, 'atomic output publication requires Linux renameat2')
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(stage), -100, os.fsencode(output), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def run(protocol, output, plot=False):
    protocol_raw = protocol.read_bytes()
    p = json.loads(protocol_raw)
    check_protocol(p)
    repo = Path(__file__).resolve().parents[2]
    require(not output.resolve().is_relative_to(repo), 'analysis output must be outside repository')
    require(not output.exists(), 'output exists; use a fresh run directory')
    require(digest(protocol.parent / p['rope_receipt_path']) == p['rope_receipt_sha256'], 'RoPE receipt hash mismatch')
    rows = []
    o1_rows = []
    sources = []
    arms = [('tail', None), ('high', None)] + [('paired_random', s) for s in p['pair_seeds']]
    for t in p['traces']:
        path = protocol.parent / t['path']
        data = load_trace(path, t['sha256'])
        check_trace(data)
        require(max(t['query_indices']) < len(data['q']), 'query index out of range')
        sources.append({'sha256': t['sha256'], 'bytes': data['_source_bytes'],
                        'sample_id': t['sample_id'], 'layer': t['layer'],
                        'loaded_q_dtype': str(data['q'].dtype), 'loaded_k_dtype': str(data['k'].dtype),
                        'compute_dtype': 'float32',
                        'original_dtypes': data.get('_original_dtypes', {key: str(data[key].dtype) for key in ('q', 'k')})})
        for qi in t['query_indices']:
            position = int(data['qpos'][qi])
            q = data['q'][qi].astype(np.float32)
            k = data['k'][:position + 1].astype(np.float32)
            require(np.isfinite(q).all() and np.isfinite(k).all(), 'nonfinite after float32 conversion')
            protection = None
            if 'protection' in p:
                import sys
                sys.path.insert(0, str(repo))
                from exp.paper.observations import protected_ids, o1_mass
                protection = protected_ids(position, p['protection']['sink'], p['protection']['swa'])
                if 'distance_bin_edges' in p:
                    _, reference = probabilities(q,k)
                    near_start = t['near_start_by_query'][str(qi)]
                    for head, probability in enumerate(reference):
                        for item in o1_mass(probability,position,near_start,protection,p['distance_bin_edges']):
                            o1_rows.append(dict(item,sample_id=t['sample_id'],document_id=t['document_id'],layer=t['layer'],query_position=position,kv_head=head))
            for kind, seed in arms:
                pairs, coord = coordinates(kind, seed)
                for r in measure(q, k, p['budget'], coord, protection):
                    rows.append(dict(r, arm_id=kind if seed is None else f'{kind}_{seed}',
                        sample_id=t['sample_id'], document_id=t['document_id'], layer=t['layer'],
                        query_position=position, query_phase='prefill', pair_ids=pairs.tolist(),
                        coordinate_ids=coord.tolist(), seed=seed, scale=128 ** -.5,
                        support='all_causal_global_topk', quantization='minmax4_k_float32'))
    table = summarize(rows, p)
    destination = output
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='.o3-pending-', dir=destination.parent))
    (output / 'o1_mass.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in o1_rows))
    (output / 'selection_metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    public_protocol = {key: p[key] for key in ('run_id', 'model_revision', 'dataset_revision',
        'producer_commit', 'split', 'rope_receipt_sha256', 'rope_layout', 'data_kind',
        'query_phase', 'support', 'quantization', 'budget', 'bootstrap_repeats',
        'bootstrap_seed', 'pair_seeds')}
    public_protocol['protection'] = p.get('protection')
    public_protocol['distance_bin_edges'] = p.get('distance_bin_edges')
    public_protocol['o1_status'] = 'diagnostic' if o1_rows else 'missing-frozen-bins-or-boundaries'
    public_protocol['traces'] = [{key: trace[key] for key in ('sha256', 'sample_id',
        'document_id', 'input_ids_sha256', 'layer', 'query_indices')} for trace in p['traces']]
    (output / 'protocol_summary.json').write_text(json.dumps(public_protocol, indent=2))
    # Reuse the thin figure contract instead of introducing another table schema.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from figure_data_contract import build_rows, tsv_bytes
    summary = output / 'summary.json'
    summary.write_text(json.dumps(table, indent=2))
    cells = []
    for i, row in enumerate(table):
        cell = {k: v for k, v in row.items() if k not in
                ('estimate', 'ci_low', 'ci_high', 'n_independent_seq', 'source_run_id', 'source_field')}
        cell['source_id'] = 'analysis'
        cell['fields'] = {key: f'/{i}/{key}' for key in
                          ('estimate', 'ci_low', 'ci_high', 'n_independent_seq')}
        cells.append(cell)
    bindings = output / 'figure_bindings.json'
    bindings.write_text(json.dumps({'schema': 'figure-bindings-v1', 'status': 'frozen',
        'sources': {'analysis': {'path': 'summary.json', 'sha256': digest(summary), 'run_id': p['run_id']}},
        'cells': cells}, indent=2))
    bound_rows, _ = build_rows(bindings)
    (output / 'figure_data.tsv').write_bytes(tsv_bytes(bound_rows))
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rcParams['svg.hashsalt'] = 'o3-fixed-qk-diagnostic'
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
        for i, row in enumerate(table):
            ax.vlines(i, row['ci_low'], row['ci_high'], color='#246080')
            ax.scatter(i, row['estimate'], color='#246080')
        ax.set_xticks(range(len(table)), [r['arm_id'] for r in table], rotation=20)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Raw full-reference retained mass')
        ax.set_title('O3 scoring diagnostic — pending independent review\n' + p['data_kind'] + ' / ' + p['run_id'])
        ax.spines[['right', 'top']].set_visible(False)
        fig.savefig(output / 'o3_diagnostic.svg', metadata={'Date': None})
        fig.savefig(output / 'o3_diagnostic.png', dpi=160)
        plt.close(fig)
    (output / 'validation.json').write_text(json.dumps({'status': 'pending-independent-review',
        'scope': 'CPU scoring diagnostic only; no production parity, L1, quotas, task quality or GPU cost',
        'data_kind': p['data_kind'],
        'identity_status': 'unverified-declarations',
        'identity': 'provided declarations and trace byte hashes; producer claims not independently verified'}, indent=2))
    assets = [{'path': f.name, 'bytes': f.stat().st_size, 'sha256': digest(f)} for f in sorted(output.iterdir())]
    (output / 'analysis_receipt.json').write_text(json.dumps({'run_id': p['run_id'], 'script_sha256': digest(Path(__file__)),
        'protocol_sha256': hashlib.sha256(protocol_raw).hexdigest(),
        'data_kind': p['data_kind'], 'identity_status': 'unverified-declarations',
        'numpy_version': np.__version__, 'sources': sources, 'artifacts': assets, 'role': 'analysis-only receipt; not producer run-cap manifest or independent verification'}, indent=2))
    publish_directory(output, destination)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()
    try:
        run(args.protocol, args.out, args.plot)
    except (ValueError, OSError, KeyError, TypeError, ImportError) as exc:
        ap.exit(2, f'O3 BLOCKED: {type(exc).__name__}; inspect private input/receipt locally\n')
