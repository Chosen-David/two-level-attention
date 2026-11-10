"""汇总 AsyncTLS PyTorch 语义 Graph 的 nsys kernel 轨迹。

只归因可唯一识别的 Top-K/排序/gather kernel family；不把同名 GEMM、
softmax 或全部模型 kernel 冒称为 AsyncTLS 选择或 selected-attention 开销。
"""
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


def _is_topk_or_sort(name):
    return (
        'at::native::mbtopk::' in name
        or 'at::native::radixSortKVInPlace' in name
        or 'at_cuda_detail::cub::DeviceScanByKey' in name
        or 'at::native::sort' in name
        or 'at::native::Sort' in name
    )


def _is_gather(name):
    lowered = name.lower()
    return 'gather' in lowered or 'indexselect' in lowered


def finalize(result, sqlite_path):
    instances = _kernel_instances(Path(sqlite_path))
    repeat = result['profiled_decode_steps']
    categories = defaultdict(list)
    members = defaultdict(set)
    for name, values in instances.items():
        if _is_topk_or_sort(name):
            categories['topk_sort'].extend(values)
            members['topk_sort'].add(name)
        if _is_gather(name):
            categories['gather'].extend(values)
            members['gather'].add(name)
    if not categories['topk_sort']:
        raise RuntimeError('AsyncTLS profile missing Top-K/sort kernels')
    if not categories['gather']:
        raise RuntimeError('AsyncTLS profile missing gather kernels')

    index_family = categories['topk_sort'] + categories['gather']
    all_kernels = [value for values in instances.values() for value in values]
    result['operator_summary'] = {
        'all_observed_model_kernels_ms': _timing(all_kernels, repeat),
        'topk_sort_family_ms': _timing(categories['topk_sort'], repeat),
        'gather_family_ms': _timing(categories['gather'], repeat),
        'index_family_ms': _timing(index_family, repeat),
        'scope': (
            'Only kernel names uniquely attributable to PyTorch Top-K/sort and '
            'gather are included in index_family_ms. Block scoring, INT4 '
            'quantize-dequantize, projected QK, softmax and exact selected '
            'attention use kernels that cannot be separated reliably from other '
            'model kernels by name; they remain in kernel_summary and are not '
            'assigned fabricated stage totals.'),
        'implementation': (
            'independent PyTorch semantic reimplementation; not official '
            'TileLang and no real asynchronous offload'),
        'members': {key: sorted(value) for key, value in members.items()},
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
        raise ValueError('output is not an AsyncTLS operator-profile receipt')
    finalize(result, args.sqlite)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({
        'output': str(output),
        'operator_summary': result['operator_summary'],
        'kernel_count': len(result['kernel_summary']),
    }, indent=2))


if __name__ == '__main__':
    main()
