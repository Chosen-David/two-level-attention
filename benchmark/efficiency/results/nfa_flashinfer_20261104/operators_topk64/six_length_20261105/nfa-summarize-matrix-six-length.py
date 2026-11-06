"""校验并按 v2 exactly-once 口径汇总 NFA/FlashInfer 纯 kernel ABBA 矩阵。"""
import argparse
import json
import statistics
from pathlib import Path

from kernel_stage_accounting import classify_nfa, corrected_arm_summary

LENGTHS = (32768, 65536, 98304, 131072)
EXECUTIONS = ('graph', 'eager')
ABBAS = ('a1', 'b1', 'b2', 'a2')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--gate-epsilon', required=True, type=float)
    parser.add_argument('--subspace', choices=('tail', 'full'))
    parser.add_argument('--lengths', type=int, nargs='+', default=list(LENGTHS))
    parser.add_argument('--output', required=True)
    parser.add_argument('roots', nargs='+', type=Path)
    args = parser.parse_args()

    arms = {}
    source_schemas = set()
    for length in args.lengths:
        for execution in EXECUTIONS:
            for abba in ABBAS:
                name = f'{args.task}_s0_{length}_{execution}_{abba}.json'
                paths = [root / name for root in args.roots if (root / name).is_file()]
                if len(paths) != 1:
                    raise ValueError(f'{name} 数量应为 1，实际为 {len(paths)}')
                row = json.loads(paths[0].read_text(encoding='utf-8'))
                expected_method = 'nfa' if abba.startswith('a') else 'flashinfer'
                assert row['schema'] in ('nfa-pure-kernel-stages-v1',
                                         'nfa-pure-kernel-stages-v2')
                source_schemas.add(row['schema'])
                assert row['method'] == expected_method
                assert row['execution'] == execution
                assert row['abba'] == abba
                assert row['warmup_steps'] == 3 and row['repeat'] == 5
                assert len(row['repeats']) == 5
                assert row['gpu_hygiene'] == 'clean'
                assert row['input']['sample_index'] == 0
                assert Path(row['input']['path']).name == f'{args.task}.jsonl'
                assert Path(row['input']['path']).parent.name == str(length)
                if expected_method == 'nfa':
                    assert row['gate_epsilon'] == args.gate_epsilon
                    if args.subspace is not None:
                        assert row.get('subspace') == args.subspace
                else:
                    assert row['gate_epsilon'] is None
                    if args.subspace is not None:
                        assert row.get('subspace') is None
                # 旧 summary 可能存在漏项；所有数值均从 raw kernels 重算。
                row['corrected_summary'] = corrected_arm_summary(row)
                arms[length, execution, abba] = row

    source_identities = {
        (row['repo_head'], row['repo_diff_sha256']) for row in arms.values()}
    if len(source_identities) != 1:
        raise ValueError(f'源码身份不一致: {source_identities}')
    for length in args.lengths:
        input_sha = {
            row['input']['sha256'] for (l, _, _), row in arms.items() if l == length}
        if len(input_sha) != 1:
            raise ValueError(f'{length} 输入 SHA 不一致')

    records = []
    for length in args.lengths:
        for execution in EXECUTIONS:
            selected = {
                abba: arms[length, execution, abba] for abba in ABBAS}
            stages = {}
            for name, abbas in (
                    ('l1_ms', ('a1', 'a2')),
                    ('l2_ms', ('a1', 'a2')),
                    ('main_attention_ms', ('a1', 'a2')),
                    ('nfa_operator_total_ms', ('a1', 'a2')),
                    ('flashinfer_dense_ms', ('b1', 'b2'))):
                arm_medians = [
                    selected[abba]['corrected_summary'][name]['median_ms']
                    for abba in abbas]
                stages[name] = statistics.median(arm_medians)
                stages[f'{name}_arm_medians_ms'] = arm_medians
            nfa_total = stages['nfa_operator_total_ms']
            skip_by_arm = {
                abba: selected[abba]['metadata']['skip'] for abba in ('a1', 'a2')}
            if args.gate_epsilon == 0:
                for skip in skip_by_arm.values():
                    assert not skip['near_layers'] and not skip['far_layers'] \
                        and not skip['both_layers']
            coverage = {stage: set() for stage in (
                'l1_ms', 'l2_ms', 'main_attention_ms')}
            for abba in ('a1', 'a2'):
                for repeat in selected[abba]['repeats']:
                    _, memberships = classify_nfa(repeat['kernels'])
                    for kernel, stage in memberships.items():
                        coverage[stage].add(kernel)
            records.append({
                'context_length': length,
                'execution': execution,
                **stages,
                'flashinfer_over_nfa_operator':
                    stages['flashinfer_dense_ms'] / nfa_total,
                'nfa_skip': skip_by_arm['a1'],
                'nfa_skip_by_arm': skip_by_arm,
                'kernel_coverage': {
                    stage: sorted(names) for stage, names in coverage.items()},
                'input': selected['a1']['input'],
                'gpu_hygiene': 'clean',
            })

    result = {
        'schema': 'nfa-pure-kernel-abba-summary-v2',
        'task': args.task,
        'sample_index': 0,
        'gate_epsilon': args.gate_epsilon,
        'subspace': args.subspace,
        'source': {
            'head': next(iter(source_identities))[0],
            'diff_sha256': next(iter(source_identities))[1],
        },
        'source_schemas': sorted(source_schemas),
        'accounting': {
            'kernel_time': 'CUDA self-device time',
            'coverage': 'every NFA-related kernel exactly once; unknown/missing stages fail',
            'repeat_aggregation': 'sum L1+L2+Attention within each repeat, then median',
            'abba_aggregation': 'median of per-arm medians; total is not a sum of stage medians',
            'scope': 'pure operator chain, not end-to-end decode TPOT',
        },
        'records': records,
    }
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print('length execution L1 L2 Attention NFA_total FlashInfer FI/NFA')
    for row in records:
        print(row['context_length'], row['execution'],
              *(f'{row[key]:.6f}' for key in (
                  'l1_ms', 'l2_ms', 'main_attention_ms', 'nfa_operator_total_ms',
                  'flashinfer_dense_ms', 'flashinfer_over_nfa_operator')))
    print(f'validated {len(arms)} clean arms; saved -> {output}')


if __name__ == '__main__':
    main()
