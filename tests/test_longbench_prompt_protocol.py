import copy

import pytest

from benchmark.LongBench.run_cap_manifest import (
    build_run_cap_core, finalize_run_cap_manifest, validate_run_cap_manifest,
)


def manifest(mode):
    core = build_run_cap_core(
        model_name='Qwen3-8B', model_path='/model', method='tli', dataset='qasper',
        dataset_fingerprint='test', run_timestamp='audit', effective_cap=32768,
        cap_gate_info=None, max_new_tokens=16, producer={}, prompt_mode=mode)
    return finalize_run_cap_manifest(
        core, status='complete', n_samples=1, input_tokens=[100],
        truncated_flags=[False], result_file='qasper.jsonl',
        result_sha256='a'*64, result_lines=1)


def test_protocol_binds_identity_and_cannot_be_changed_on_one_side():
    full, split = manifest('full'), manifest('split')
    assert full['cap_identity']['hash'] != split['cap_identity']['hash']
    for item in (full, split, manifest(None)):
        validate_run_cap_manifest(item)
    changed = copy.deepcopy(full)
    changed['prompt_mode'] = 'split'
    with pytest.raises((ValueError, SystemExit), match='prompt_mode'):
        validate_run_cap_manifest(changed)
