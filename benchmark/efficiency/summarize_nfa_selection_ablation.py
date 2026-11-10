"""合并 NFA tail/full × skip/no-skip 的 12 个可行 Graph 算子格。"""
import argparse
import csv
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from benchmark.efficiency.run_nfa_perf_matrix import sha256, write_json


COMBOS = ('tail_skip', 'tail_noskip', 'full_skip', 'full_noskip')
STAGES = (
    'l1_score_ms', 'l1_topk_ms', 'candidate_ms', 'l1_ms',
    'l2_score_ms', 'l2_topk_ms', 'selection_union_ms', 'l2_ms',
    'topk_total_ms', 'selected_attention_ms', 'nfa_operator_total_ms',
)


def _load(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('validation_status') != 'pending-independent-verification':
        raise ValueError(f'unexpected validation status: {path}')
    rows = {}
    for record in value['records']:
        key = (record['length'], record['batch'])
        if key in rows:
            raise ValueError(f'duplicate cell {key}: {path}')
        rows[key] = record['arms']['nfa_operator_profile']
    if len(rows) != 12:
        raise ValueError(f'expected 12 feasible cells, got {len(rows)}: {path}')
    return value, rows


def _stages(operator):
    summary = operator['operator_summary']
    return {name: summary[name]['mean_ms_per_step'] for name in STAGES}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation-root', type=Path, required=True)
    parser.add_argument('--main-summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = {
        'tail_skip': args.main_summary,
        'tail_noskip': args.ablation_root / 'tail_noskip/summary.json',
        'full_skip': args.ablation_root / 'full_skip/summary.json',
        'full_noskip': args.ablation_root / 'full_noskip/summary.json',
    }
    documents, rows = {}, {}
    for combo, path in paths.items():
        documents[combo], rows[combo] = _load(path)
    cells = set(rows['tail_skip'])
    if any(set(rows[combo]) != cells for combo in COMBOS[1:]):
        raise ValueError('ablation cell sets differ')
    source = documents['tail_skip']['source_by_arm']['nfa-operator-profile']
    for combo in COMBOS[1:]:
        if documents[combo]['source_by_arm']['nfa-operator-profile'] != source:
            raise ValueError(f'NFA source identity differs for {combo}')

    records = []
    for length, batch in sorted(cells, key=lambda item: (-item[0], item[1])):
        operators = {combo: rows[combo][(length, batch)] for combo in COMBOS}
        stages = {combo: _stages(operators[combo]) for combo in COMBOS}
        record = {
            'length': length,
            'batch': batch,
            'execution': 'graph',
            'stages_ms_per_step': stages,
            'tail_vs_full_speedup': {
                mode: {
                    name: stages[f'full_{mode}'][name] / stages[f'tail_{mode}'][name]
                    for name in STAGES
                } for mode in ('skip', 'noskip')
            },
            'skip_vs_noskip_speedup': {
                subspace: {
                    name: stages[f'{subspace}_noskip'][name] / stages[f'{subspace}_skip'][name]
                    for name in STAGES
                } for subspace in ('tail', 'full')
            },
            'paths': {combo: operators[combo]['path'] for combo in COMBOS},
            'nsys_receipts': {
                combo: operators[combo]['nsys_receipt'] for combo in COMBOS
            },
        }
        records.append(record)

    result = {
        'schema': 'nfa-selection-tail-full-skip-ablation-v1',
        'validation_status': 'pending-independent-verification',
        'source': source,
        'combiner_sha256': sha256(Path(__file__)),
        'semantics': {
            'tail_skip': 'tail32 subspace + dynamic gate epsilon 0.05',
            'tail_noskip': 'tail32 subspace + gate epsilon 0',
            'full_skip': 'full-dimensional index + dynamic gate epsilon 0.05',
            'full_noskip': 'full-dimensional index + gate epsilon 0',
            'tail_vs_full_speedup': 'full time / tail time; >1 means tail is faster',
            'skip_vs_noskip_speedup': 'no-skip time / skip time; >1 means skip is faster',
            'negative_results_retained': True,
            'operator_observer_effect': documents['tail_skip']['protocol']['observer_effect'],
        },
        'source_summaries': {combo: str(path.resolve()) for combo, path in paths.items()},
        'infeasible_memory_cells': documents['tail_skip']['infeasible_memory_cells'],
        'records': records,
    }
    write_json(args.output, result)
    tsv_path = args.output.with_suffix('.tsv')
    with open(tsv_path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(('length', 'batch', 'combo', *STAGES))
        for record in records:
            for combo in COMBOS:
                writer.writerow((
                    record['length'], record['batch'], combo,
                    *(record['stages_ms_per_step'][combo][name] for name in STAGES)))
    print(f'saved {args.output} and {tsv_path}; cells={len(records)}')


if __name__ == '__main__':
    main()
