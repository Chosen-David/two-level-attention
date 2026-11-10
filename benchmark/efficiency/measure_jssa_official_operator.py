"""汇总 benchmark_jssa_official.py 的外部 nsys 稳态 Graph kernel 轨迹。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from benchmark.efficiency.measure_flashinfer_operator import _kernel_instances


def _timing(values, repeat):
    return {
        'calls_total': len(values),
        'mean_calls_per_step': len(values) / repeat,
        'total_ms': sum(values),
        'mean_ms_per_step': sum(values) / repeat,
    }


def finalize(result, sqlite_path):
    instances = _kernel_instances(Path(sqlite_path))
    repeat = result['profiled_decode_steps']
    layers = result['jssa_runtime']['decode_attention_calls_warmup_and_capture'] // 2
    expected = repeat * layers
    categories = defaultdict(list)
    topk_members = set()
    for name, values in instances.items():
        if name == '_jssa_project_store_contig_fp8_kernel':
            categories['store'].extend(values)
        elif name == '_jssa_score_contig_kernel':
            categories['score'].extend(values)
        elif name in ('_indexed_attend_kv_kernel', '_split_attend_kernel',
                      '_split_reduce_kernel'):
            categories['attention'].extend(values)
        elif ('at::native::mbtopk::' in name
              or 'at::native::radixSortKVInPlace' in name
              or 'at_cuda_detail::cub::DeviceScanByKey' in name):
            categories['topk'].extend(values)
            topk_members.add(name)
    for name in ('store', 'score', 'attention'):
        if not categories[name]:
            raise RuntimeError(f'JSSA profile missing {name} kernels')
    if len(categories['store']) != expected or len(categories['score']) != expected:
        raise RuntimeError(
            f'JSSA store/score counts must be {expected}: '
            f"{len(categories['store'])}/{len(categories['score'])}")
    if result['jssa_runtime']['radix_topk_available']:
        raise RuntimeError('this classifier currently requires disclosed torch.topk fallback')
    if not categories['topk']:
        raise RuntimeError('JSSA torch.topk fallback kernels were not observed')

    score_topk = categories['score'] + categories['topk']
    selection_with_store = score_topk + categories['store']
    named_total = selection_with_store + categories['attention']
    all_kernels = [value for values in instances.values() for value in values]
    result['operator_summary'] = {
        'all_observed_model_kernels_ms': _timing(all_kernels, repeat),
        'store_ms': _timing(categories['store'], repeat),
        'score_ms': _timing(categories['score'], repeat),
        'topk_ms': _timing(categories['topk'], repeat),
        'score_topk_ms': _timing(score_topk, repeat),
        'selection_with_store_ms': _timing(selection_with_store, repeat),
        'selected_attention_ms': _timing(categories['attention'], repeat),
        'official_named_operator_total_ms': _timing(named_total, repeat),
        'scope': (
            'Official named store/score/indexed-attend plus the complete observed '
            'torch.topk kernel family. Query group-mean/projection/scale kernels '
            'remain in kernel_summary but are not attributed to named totals.'),
        'topk_implementation': 'official torch.topk fallback; not vLLM radix',
        'topk_members': sorted(topk_members),
    }
    result['kernel_summary'] = {
        name: {
            **_timing(values, repeat),
            'mean_instance_ms': statistics.mean(values),
            'median_instance_ms': statistics.median(values),
            'min_instance_ms': min(values),
            'max_instance_ms': max(values),
        } for name, values in sorted(instances.items())
    }
    result['nsys_profile_finalized'] = True
    result['profile_aggregation'] = (
        f'aggregate {repeat} steady Graph replay steps; total and arithmetic '
        'mean per step; kernel instance statistics are not per-step distributions')
    result['cross_mode_comparison_rule'] = (
        'Do not subtract profiled kernel totals from clean decode TPOT and do '
        'not interpret their difference as Python/host overhead.')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--sqlite', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    result = json.loads(output.read_text(encoding='utf-8'))
    if result['measurement_mode'] != 'operator-profile':
        raise ValueError('output is not a JSSA operator-profile receipt')
    finalize(result, args.sqlite)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({
        'output': str(output),
        'operator_summary': result['operator_summary'],
        'kernel_count': len(result['kernel_summary']),
    }, indent=2))


if __name__ == '__main__':
    main()
