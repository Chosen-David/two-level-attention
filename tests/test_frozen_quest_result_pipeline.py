import hashlib
import json

import pytest

from benchmark.LongBench.analyze_frozen_skip import analyze
from benchmark.LongBench.eval import sample_score
from benchmark.LongBench.export_frozen_skip import export
from benchmark.LongBench.prefill_frozen_skip import prepare_quest_input
import benchmark.LongBench.validate_frozen_skip as validator


class Encoding(dict):
    __getattr__ = dict.__getitem__


class CharTokenizer:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        ids = [ord(char) for char in text]
        offsets = [(i, i+1) for i in range(len(text))] if return_offsets_mapping else None
        return Encoding(input_ids=ids, offset_mapping=offsets)

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return ''.join(chr(token) for token in ids)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False):
        del tokenize, add_generation_prompt, enable_thinking
        return messages[0]['content']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_report(length):
    sink = min(4, length)
    end = max(sink, length-128)
    far = min(end, max(sink, ((length-128-4096)//64)*64))
    nodes = {
        'sink': (sink, max(sink, 1)),
        'far': (far-sink, max(far-sink, 1)),
        'near': (end-far, max(end-far, 1)),
        'swa': (length-end, max(length-end, 1)),
    }
    factor = 2*2*1*128
    report = dict(logical_length=length, skip_near=False, skip_far=False,
        nodes={name: dict(tokens=tokens, capacity=capacity, kv_bytes=capacity*factor,
                          attention_visible=True)
               for name, (tokens, capacity) in nodes.items()},
        stored_tokens=length, attention_tokens=length,
        linked_order=list(nodes), attention_order=list(nodes), permanently_dropped=0)
    return report, sum(node['kv_bytes'] for node in report['nodes'].values())


def trial(quest, generated, prediction, score):
    prefix = len(quest.prefix_ids)
    before, before_bytes = cache_report(prefix)
    frozen, frozen_bytes = cache_report(prefix)
    final, final_bytes = cache_report(len(quest.full_ids)+len(generated)-1)
    steps = []
    for i, token_id in enumerate(quest.suffix_ids):
        position = prefix+i
        steps.append(dict(phase='suffix', logical_position=position,
            input_token_id=token_id, layers=[dict(logical_tokens=position+1,
                visible_tokens=position+1, attended_per_kv_head=[position+1],
                scorer='native-dense-prefill', selector='dense-visible-cache',
                skip_near=False, skip_far=False)]))
    profile = {region: dict(mean=.5, p95=.5, max=.5) for region in ('near', 'far')}
    return dict(generated_ids=generated, prediction=prediction, generated_tokens=1,
        score=score, profiles={'0': profile}, decisions={'0': dict(skip_near=False, skip_far=False)},
        gate_counts=dict(keep=1, near=0, far=0, both=0), tli_receipts={},
        prefill_full_kv=dict(layers=[before], kv_bytes=before_bytes),
        frozen_kv=dict(layers=[frozen], kv_bytes=frozen_bytes),
        final_kv=dict(layers=[final], kv_bytes=final_bytes),
        attention_work=dict(schema='frozen-region-attention-work-v1',
            units='unique valid attended key tokens per KV head for each single-token query',
            prefill='dense prefix', budgets='K1 coarse; K2 fine; neither is actual attended-K',
            steps=steps),
        quest_protocol=dict(freeze_logical_position=prefix,
            suffix_logical_range=[prefix, len(quest.full_ids)],
            profile_query_rows='last up to 256 prefix rows only',
            suffix_replay_tokens=len(quest.suffix_ids),
            decode_capacity=len(quest.suffix_ids)+1))


def build_complete_run(tmp_path):
    outdir = tmp_path/'run'; data = tmp_path/'data'; model = tmp_path/'model'
    (outdir/'requests'/'qasper').mkdir(parents=True)
    data.mkdir(); model.mkdir()
    row = dict(context='Material.', input='A', answers=['B'], all_classes=[])
    (data/'qasper.jsonl').write_text(json.dumps(row)+'\n')
    config = dict(hidden_size=128, num_attention_heads=1, num_key_value_heads=1,
                  head_dim=128, num_hidden_layers=1)
    (model/'config.json').write_text(json.dumps(config))
    (model/'tokenizer_config.json').write_text('{}')
    (model/'generation_config.json').write_text(json.dumps(dict(eos_token_id=0)))
    prompt = json.loads((validator.ROOT/'benchmark/LongBench/config/dataset2prompt.json').read_text())['qasper']
    quest, input_info = prepare_quest_input(CharTokenizer(), row, 'qasper', prompt, 31500)
    protocol = dict(schema=validator.QUEST_SCHEMA, model=str(model), context_cap=31500,
        samples_per_task=1, seed=42, statistic='mean', thresholds=[.05], datasets=['qasper'],
        selected_rows={'qasper': [0]}, max_new_tokens={'qasper': 1}, source_sha256={},
        dataset_sha256={'qasper': digest(data/'qasper.jsonl')},
        model_config_sha256=digest(model/'config.json'),
        tokenizer_config_sha256=digest(model/'tokenizer_config.json'), world_size=1)
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    (outdir/'manifest_rank0.json').write_text(json.dumps(dict(identity=identity, protocol=protocol)))
    arms = {
        'full': trial(quest, [ord('B')], 'B', sample_score('qasper', 'B', row['answers'], [])),
        'far_0.05': trial(quest, [ord('B')], 'B', sample_score('qasper', 'B', row['answers'], [])),
        # Quest 下 suffix 已在冻结后经过不同 gate，首答案允许合理分叉。
        'dual_0.05': trial(quest, [ord('C')], 'C', sample_score('qasper', 'C', row['answers'], [])),
    }
    (outdir/'requests'/'qasper'/'0000.json').write_text(json.dumps(dict(
        identity=identity, dataset='qasper', row_index=0, input=input_info,
        answers=row['answers'], arms=arms, status='complete')))
    return outdir, data, quest


def test_quest_result_validate_analyze_export_chain(tmp_path, monkeypatch):
    outdir, data, quest = build_complete_run(tmp_path)
    monkeypatch.setattr(validator.AutoTokenizer, 'from_pretrained', lambda _model: CharTokenizer())
    report = validator.validate(outdir, data_root=data)
    assert report['status'] == 'complete-validated'
    assert report['checked_attention_steps'] == 3*len(quest.suffix_ids)
    assert report['checked_attention_layer_steps'] == 3*len(quest.suffix_ids)
    summary = analyze(outdir)
    assert summary['arms']['full']['macro_avg'] == 100
    assert summary['arms']['dual_0.05']['macro_avg'] == 0
    assert 'avg5' not in summary['arms']['full']
    assert summary['arms']['full']['decode_attended_kv_head_tokens_mean'] > 0
    assert export(outdir)['predictions'] == 3
    header = (outdir/'scores.tsv').read_text().splitlines()[0]
    assert 'MACRO_AVG' in header and 'AVG5' not in header


def test_quest_validator_rejects_full_prompt_freeze(tmp_path, monkeypatch):
    outdir, data, quest = build_complete_run(tmp_path)
    path = outdir/'requests'/'qasper'/'0000.json'
    record = json.loads(path.read_text())
    record['arms']['full']['frozen_kv']['layers'][0]['logical_length'] = len(quest.full_ids)
    path.write_text(json.dumps(record))
    monkeypatch.setattr(validator.AutoTokenizer, 'from_pretrained', lambda _model: CharTokenizer())
    with pytest.raises(AssertionError):
        validator.validate(outdir, data_root=data)
