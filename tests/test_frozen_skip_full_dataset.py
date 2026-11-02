from types import SimpleNamespace

import numpy as np
import pytest

from benchmark.LongBench.prefill_frozen_skip import (
    SCREEN_TASKS, prepare_ids, select_row_indices, sampling_metadata, shard_inputs,
)
from benchmark.LongBench.export_frozen_skip import coverage_text


def test_all_five_datasets_have_complete_disjoint_eight_way_shards():
    counts = dict(zip(SCREEN_TASKS, [200, 200, 200, 200, 500]))
    selected = {ds: select_row_indices(size, 0, 42) for ds, size in counts.items()}
    inputs = [(ds, row) for ds, ids in selected.items() for row in ids]
    expected = {(ds, row) for ds, size in counts.items() for row in range(size)}
    shards = [list(shard_inputs(inputs, rank, 8)) for rank in range(8)]
    seen = set()
    for shard in shards:
        assert not seen.intersection(shard)
        assert len(shard) == len(set(shard))
        seen.update(shard)
    assert len(inputs) == sum(map(len, shards)) == 1300
    assert seen == expected
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
    assert selected == {ds: select_row_indices(size, 0, 42) for ds, size in counts.items()}
    metadata = sampling_metadata(0, counts)
    assert metadata == dict(samples_per_task=None, full_dataset=True,
                            dataset_counts=counts, selected_counts=counts)


def test_positive_sampling_keeps_historical_selection_and_manifest_fields():
    for count in (200, 500):
        assert select_row_indices(count, 20, 42) == np.random.default_rng(42).permutation(count)[:20].tolist()
        assert len(select_row_indices(count, 20, 42)) == 20
    assert sampling_metadata(20, {'qasper': 200}) == {'samples_per_task': 20}


def test_zero_and_invalid_selection_boundaries():
    assert select_row_indices(0, 0, 42) == []
    assert select_row_indices(3, 20, 42) == select_row_indices(3, 0, 42)
    with pytest.raises(ValueError):
        select_row_indices(200, -1, 42)
    with pytest.raises(ValueError):
        list(shard_inputs([], 8, 8))


def test_full_dataset_report_uses_actual_unequal_counts():
    counts = dict(zip(SCREEN_TASKS, [200, 200, 200, 200, 500]))
    protocol = dict(datasets=list(SCREEN_TASKS), **sampling_metadata(0, counts))
    text = coverage_text(protocol, 1300, 3900)
    assert '全量数据' in text and 'repobench-p 500 条' in text
    assert '1300 条配对请求' in text and '3900 组生成' in text
    assert '小样本' not in text and '各 0 条' not in text and '1000000' not in text
    legacy = coverage_text(dict(datasets=list(SCREEN_TASKS), samples_per_task=20),100,300)
    assert '各 20 条' in legacy and '不是' in legacy


class _ChatTokenizer:
    def __call__(self, text, add_special_tokens):
        return SimpleNamespace(input_ids=[ord(char) for char in text])

    def decode(self, ids, skip_special_tokens):
        return ''.join(chr(token) for token in ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                            enable_thinking):
        material = '<user>' + messages[0]['content'] + '</user>'
        return material + ('<assistant>' if add_generation_prompt else '')


def test_longbench_material_and_generation_suffix_preserve_full_token_identity():
    tokenizer = _ChatTokenizer()
    material, suffix, info = prepare_ids(
        tokenizer, {'context': 'abc'}, 'qasper', '{context}', 100)
    assert tokenizer.decode(material, True) == '<user>abc</user>'
    assert tokenizer.decode(suffix, True) == '<assistant>'
    assert info['input_tokens'] == len(material) + len(suffix)
    assert info['material_tokens'] == len(material)
    assert info['suffix_tokens'] == len(suffix)

    material, suffix, _ = prepare_ids(
        tokenizer, {'context': 'abc'}, 'repobench-p', '{context}', 100)
    assert tokenizer.decode(material, True) == 'abc'
    assert suffix == []


def test_non_suffix_chat_template_fails_closed():
    class BadTokenizer(_ChatTokenizer):
        def apply_chat_template(self, messages, *, tokenize,
                                add_generation_prompt, enable_thinking):
            if add_generation_prompt:
                return '<assistant>' + super().apply_chat_template(
                    messages, tokenize=tokenize,
                    add_generation_prompt=False,
                    enable_thinking=enable_thinking)
            return super().apply_chat_template(
                messages, tokenize=tokenize, add_generation_prompt=False,
                enable_thinking=enable_thinking)

    with pytest.raises(ValueError, match='strict prompt suffix'):
        prepare_ids(BadTokenizer(), {'context': 'abc'}, 'qasper', '{context}', 100)
