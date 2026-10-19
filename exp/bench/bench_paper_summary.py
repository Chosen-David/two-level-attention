"""Paired CPU aggregation benchmark; no model, CUDA or attention speed claim.

Run from repository root with --baseline set to the reviewed parent commit.
The baseline module is loaded from git without changing the working tree.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exp.paper import contract
from sparse_attn.ops.selection_contract import support_receipt


def fixture(count):
    routes = {phase: ['dense'] for phase in ('material', 'question', 'decode')}
    identity = dict(model_revision='fixture-v1', tokenizer_revision='fixture-v1',
        dataset_revision='fixture-v1', scorer_sha256='a'*64, template='plain',
        thinking=False, bos_eos='fixed', rope='none', dtype='bf16', generation='greedy',
        phase_route_by_layer=routes, first_dense_layers=[], batch=1, chunk=64,
        max_new_tokens=8, config_sha256='b'*64, producer_commit='c'*40,
        backend_id='synthetic-host-receipt', selection_semantics={'kind': 'dense'},
        comparison_track='native')
    overrides = {arm: {key: identity[key] for key in ('backend_id',
        'selection_semantics', 'config_sha256', 'phase_route_by_layer',
        'first_dense_layers')} for arm in ('dense', 'method')}
    samples = [dict(sample_id=f's{i}', document_id=f'd{i}', split='confirmation',
        input_ids_sha256=hashlib.sha256(str(i).encode()).hexdigest()) for i in range(count)]
    geometry = dict(query_position=63, length=64, sink=2, swa=0)
    p = dict(schema='paper-run-v1', status='frozen', identity=identity,
        arm_overrides=overrides, samples=samples,
        support_units={s['sample_id']: {a: {'q63:h0': geometry} for a in overrides}
                       for s in samples}, context_limit=128, repeats=2, warmup=0,
        timing_level='T3', cache_state='request-cold', order_policy='AB/BA',
        run_mode='fixed-workload', timer_boundary='model-entry-to-last-output-complete',
        sync_policy='before-after', arms=['dense', 'method'])
    sha = hashlib.sha256(contract.canonical(p)).hexdigest()
    support = support_receipt(list(range(64)), **geometry, k_cap=64)
    rows = []
    for sample in samples:
        for repetition in range(p['repeats']):
            for arm in p['arms']:
                rows.append(dict(sample, arm=arm, protocol_sha256=sha, identity=identity,
                    repetition=repetition, status='ok', measured_this_run=True,
                    **{k: p[k] for k in ('timing_level', 'cache_state', 'run_mode',
                                       'timer_boundary', 'sync_policy')},
                    aggregation='direct-wall', request_wall_ms=10 if arm=='dense' else 20,
                    actual_ctx_tokens=64, actual_generated_tokens=3,
                    lifecycle={k: dict(ms=0, included_in_wall=True, overlap=False)
                               for k in contract.LIFECYCLE}, fallback_count=0,
                    actual_routes=routes, order=(p['arms'].index(arm)+repetition)%2,
                    support={'q63:h0': support}))
    return p, rows, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--samples', nargs='+', type=int, default=[32, 128, 512])
    parser.add_argument('--pairs', type=int, default=5)
    args = parser.parse_args()
    if args.pairs < 1 or any(n < 1 for n in args.samples):
        parser.error('positive sample and pair counts required')
    commit = subprocess.check_output(['git', 'rev-parse', args.baseline+'^{commit}'],
                                     cwd=ROOT, text=True).strip()
    source = subprocess.check_output(['git', 'show', commit+':exp/paper/contract.py'], cwd=ROOT)
    before = types.ModuleType('baseline_contract')
    exec(compile(source, '<baseline_contract>', 'exec'), before.__dict__)
    report = dict(baseline_commit=commit, python=sys.version, platform=platform.platform(),
        processor=platform.processor(), timer='perf_counter_ns; full summarize only',
        warmup_per_implementation=1, includes='protocol, all row checks, pairing and summary',
        excludes='fixture creation, imports, hashing, JSON serialization; no GPU',
        baseline_source_sha256=hashlib.sha256(source).hexdigest(),
        candidate_source_sha256=hashlib.sha256(Path(contract.__file__).read_bytes()).hexdigest(), cases=[])
    for count in args.samples:
        p, rows, sha = fixture(count)
        snapshot = contract.canonical([p, rows])
        expected = before.summarize(rows, p, sha)
        if contract.summarize(rows, p, sha) != expected:
            raise RuntimeError('warmup summary differs from baseline')
        times = {name: [] for name in ('baseline', 'candidate')}
        functions = dict(baseline=before.summarize, candidate=contract.summarize)
        for pair in range(args.pairs):
            order = ('baseline', 'candidate') if pair % 2 == 0 else ('candidate', 'baseline')
            for name in order:
                start = time.perf_counter_ns()
                result = functions[name](rows, p, sha)
                elapsed = (time.perf_counter_ns()-start)/1e6
                if result != expected:
                    raise RuntimeError('summary differs from baseline')
                times[name].append(elapsed)
        if contract.canonical([p, rows]) != snapshot:
            raise RuntimeError('input mutated')
        medians = {name: statistics.median(values) for name, values in times.items()}
        report['cases'].append(dict(samples=count, rows=len(rows),
            input_sha256=hashlib.sha256(snapshot).hexdigest(),
            summary_sha256=hashlib.sha256(contract.canonical(expected)).hexdigest(),
            wall_ms=times, median_ms=medians,
            baseline_over_candidate=medians['baseline']/medians['candidate']))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
