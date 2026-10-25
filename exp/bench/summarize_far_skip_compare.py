"""Recompute reported CPU comparisons from the final paired raw receipts."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(root):
    rows = []
    for filename in ('paired_cpu.json', 'paired_cpu_holdout.json'):
        raw = json.loads((root / filename).read_text())
        for length in raw['config']['lengths']:
            pairs = [r for r in raw['measurements'] if r['length'] == length]
            medians = {name: statistics.median([t for p in pairs for t in p['runs'][name]['wall_ms']])
                       for name in ('v1', 'optimized')}
            rows.append(dict(source=filename, length=length, near_len=raw['config']['near_len'],
                             median_wall_ms=medians,
                             cpu_only_speed_ratio=medians['v1']/medians['optimized'],
                             median_first_decode_ms={name: statistics.median([
                                 p['runs'][name]['first_decode_wall_ms'] for p in pairs])
                                 for name in ('v1', 'optimized')},
                             kv_storage_bytes={name: pairs[0]['runs'][name]['kv_storage_bytes']
                                               for name in ('v1', 'optimized')},
                             outputs_checked=sum(p['outputs_checked'] for p in pairs),
                             max_abs_output_error=max(p['max_abs_error'] for p in pairs)))
    return dict(scope='CPU single-layer synthetic eager; not GPU or full-model speedup',
                excluded=['*_initial.json', '*_pre_validation_guard.json',
                          'paired_cpu_holdout_contended.json'], comparisons=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.result_directory), indent=2))


if __name__ == '__main__':
    main()
