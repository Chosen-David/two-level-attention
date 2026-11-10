"""按同一 Qwen3/RULER 输入合并 NFA、FlashInfer 与 vendor Graph 性能。"""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from benchmark.efficiency.run_nfa_perf_matrix import sha256, write_json


def _load(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('validation_status') != 'pending-independent-verification':
        raise ValueError(f'unexpected validation status: {path}')
    return value


def _nfa_records(document):
    rows = {}
    for record in document['records']:
        key = (record['length'], record['batch'])
        arms = record['arms']
        rows[key] = {
            'input': record['input'],
            'nfa_tpot': arms['nfa_clean_e2e']['tpot_ms'],
            'flashinfer_tpot': arms['flashinfer_e2e']['tpot_ms'],
            'nfa_operator': arms['nfa_operator_profile']['operator_summary'],
            'flashinfer_operator': arms['flashinfer_operator_profile']['operator_summary'],
            'speedups': record['speedups'],
            'nfa_flashinfer_token_identity': record['nfa_flashinfer_token_identity'],
        }
    return rows


def _jssa_records(document):
    rows = {}
    for record in document['records']:
        rows[(record['length'], record['batch'])] = {
            'input': record['input'],
            'tpot': record['clean_e2e']['tpot_ms'],
            'operator': record['operator']['summary'],
        }
    return rows


def _semantic_records(document):
    rows = {}
    for record in document['records']:
        rows[(record['length'], record['batch'])] = {
            'input': record['input'],
            'tpot': record['clean_e2e']['tpot_ms'],
            'operator': record['operator']['summary'],
        }
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nfa-flashinfer', type=Path, required=True)
    parser.add_argument('--jssa', type=Path, required=True)
    parser.add_argument('--asynctls', type=Path, required=True)
    parser.add_argument('--quest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    documents = {
        'nfa_flashinfer': _load(args.nfa_flashinfer),
        'jssa': _load(args.jssa),
        'asynctls': _load(args.asynctls),
        'quest': _load(args.quest),
    }
    nfa = _nfa_records(documents['nfa_flashinfer'])
    jssa = _jssa_records(documents['jssa'])
    asynctls = _semantic_records(documents['asynctls'])
    quest = _semantic_records(documents['quest'])
    expected_cells = set(nfa)
    for name, rows in (('jssa', jssa), ('asynctls', asynctls), ('quest', quest)):
        if set(rows) != expected_cells:
            raise ValueError(f'{name} cells do not match NFA/FlashInfer matrix')
    if len(expected_cells) != 12:
        raise ValueError(f'expected 12 feasible cells, got {len(expected_cells)}')

    records = []
    for length, batch in sorted(expected_cells, key=lambda item: (-item[0], item[1])):
        key = (length, batch)
        input_receipt = nfa[key]['input']
        for name, rows in (('jssa', jssa), ('asynctls', asynctls), ('quest', quest)):
            other = rows[key]['input']
            if (other['sha256'] != input_receipt['sha256']
                    or other['sample_index'] != input_receipt['sample_index']
                    or other['actual_prompt_tokens'] != input_receipt['actual_prompt_tokens']):
                raise ValueError(f'input identity mismatch: {name} L{length} B{batch}')
        nfa_tpot = nfa[key]['nfa_tpot']
        record = {
            'length': length,
            'batch': batch,
            'execution': 'graph',
            'input': input_receipt,
            'clean_decode_tpot_ms': {
                'nfa_tail_skip': nfa_tpot,
                'flashinfer_dense': nfa[key]['flashinfer_tpot'],
                'jssa_official_triton': jssa[key]['tpot'],
                'asynctls_semantic_pytorch': asynctls[key]['tpot'],
                'quest_semantic_pytorch': quest[key]['tpot'],
            },
            'clean_tpot_speedup_over_nfa': {
                'flashinfer_dense': nfa[key]['flashinfer_tpot'] / nfa_tpot,
                'jssa_official_triton': jssa[key]['tpot'] / nfa_tpot,
                'asynctls_semantic_pytorch': asynctls[key]['tpot'] / nfa_tpot,
                'quest_semantic_pytorch': quest[key]['tpot'] / nfa_tpot,
            },
            'speedup_definition': 'other clean end-to-end decode TPOT / NFA clean TPOT; >1 means NFA is faster',
            'nfa_vs_flashinfer_operator_speedups': nfa[key]['speedups'],
            'nfa_flashinfer_token_identity': nfa[key]['nfa_flashinfer_token_identity'],
            'operator_observations': {
                'nfa': nfa[key]['nfa_operator'],
                'flashinfer': nfa[key]['flashinfer_operator'],
                'jssa': jssa[key]['operator'],
                'asynctls': asynctls[key]['operator'],
                'quest': quest[key]['operator'],
            },
            'operator_scope_notes': {
                'nfa_flashinfer': (
                    'pure attention and attention-path speedups use their explicit '
                    'main+merge/selected-attention definitions'),
                'jssa': (
                    'official named store+score+Top-K fallback+indexed-attention '
                    'total; not official radix Top-K performance'),
                'asynctls_quest': (
                    'only uniquely identifiable PyTorch index-family kernels; '
                    'index_family_ms is not a complete selector or attention total'),
            },
        }
        records.append(record)

    common_source = documents['nfa_flashinfer']['source_by_arm']['nfa-operator-profile']
    for name in ('jssa', 'asynctls', 'quest'):
        source = documents[name]['source']
        if source['head'] != common_source['head'] or source['diff_sha256'] != common_source['diff_sha256']:
            raise ValueError(f'common source identity mismatch: {name}')
    result = {
        'schema': 'qwen-nfa-vendor-graph-comparison-v1',
        'validation_status': 'pending-independent-verification',
        'combiner_sha256': sha256(Path(__file__)),
        'common_source': common_source,
        'source_summaries': {
            'nfa_flashinfer': str(args.nfa_flashinfer.resolve()),
            'jssa': str(args.jssa.resolve()),
            'asynctls': str(args.asynctls.resolve()),
            'quest': str(args.quest.resolve()),
        },
        'semantics': {
            'nfa': documents['nfa_flashinfer']['arguments'],
            'flashinfer': 'dense FlashInfer 0.5.2 Graph decode, page64',
            'jssa': documents['jssa']['semantics'],
            'asynctls': documents['asynctls']['semantics'],
            'quest': documents['quest']['semantics'],
        },
        'protocol': {
            'execution': 'CUDA Graph only',
            'clean_e2e': 'separate process without nsys or torch.profiler',
            'operator_profile': 'separate nsys CUDA Graph node trace process',
            'host_overhead_derivable_by_subtraction': False,
            'operator_totals_cross_method_comparable': False,
            'clean_tpot_cross_method_comparable': True,
            'extended_input_performance_only': True,
        },
        'records': records,
        'infeasible_memory_cells': {
            'nfa_flashinfer': documents['nfa_flashinfer']['infeasible_memory_cells'],
            'jssa': documents['jssa']['infeasible_memory_cells'],
            'asynctls': documents['asynctls']['infeasible_memory_cells'],
            'quest': documents['quest']['infeasible_memory_cells'],
        },
    }
    write_json(args.output, result)
    print(f'saved comparison -> {args.output}; cells={len(records)}')


if __name__ == '__main__':
    main()
