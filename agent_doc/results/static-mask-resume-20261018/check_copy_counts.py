"""Paired CPU allocation/dispatch probe; not a latency benchmark or GPU proof.

Usage: python check_copy_counts.py BASELINE_QWEN_SOURCE [--output result.json]
The baseline must be extracted from the recorded main commit before patching.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

import torch
import transformers

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('mask_tests', ROOT / 'tests/test_qwen_static_mask.py')
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)


def run(source):
    tests.SOURCE = source
    case = tests.StaticMask()
    case.setUp()
    # Same actual HF DynamicCache, tiny Qwen attention, input, seed and CPU SDPA.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                profile_memory=True) as prof:
        output, _, _, _ = case.forward(None, dynamic=True)
    events = {e.key: dict(count=e.count, cpu_memory_bytes=e.cpu_memory_usage)
              for e in prof.key_averages() if e.key in
              ('aten::contiguous', 'aten::clone', 'aten::_local_scalar_dense')}
    return output, events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    candidate = tests.SOURCE
    before, a = run(args.baseline)
    after, b = run(candidate)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    # Report actual dispatch counts. Entering the old StaticCache branch does
    # not imply a copy: the inner contiguity condition may already be false.
    inputs = torch.arange(48, dtype=torch.float32).reshape(1, 3, 16) / 48
    record = dict(scope='CPU full attention forward operator counts, not timing',
                  platform=platform.platform(), python=sys.version.split()[0],
                  torch=torch.__version__, transformers=transformers.__version__,
                  cuda_available=torch.cuda.is_available(), seed=20261018,
                  input_sha256=hashlib.sha256(inputs.numpy().tobytes()).hexdigest(),
                  shape=[1, 3, 16], q_heads=4, kv_heads=2, head_dim=4,
                  baseline_source_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                  candidate_source_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                  output_equal=True, baseline=a, candidate=b)
    text = json.dumps(record, indent=2) + '\n'
    if args.output: args.output.write_text(text)
    print(text)


if __name__ == '__main__': main()
