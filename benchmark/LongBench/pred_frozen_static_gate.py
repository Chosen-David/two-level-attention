"""LongBench-v2 static-gate Graph-ready frozen NFA runner.

This is a narrow production entrypoint for the GLM-4.7-Flash static gate:
  - dense prefill into ``GraphReadyPagedRegionCache``
  - per-layer near/far decisions are read from the precomputed static artifact
  - decode uses the graph-ready sparse gather path (no request-time gate)
  - output is a single LongBench-v2 JSONL consumable by ``benchmark/LongBench/eval.py``

The old ``pred.py`` NFA path was removed and ``prefill_frozen_skip.py`` is a
paired screen runner; neither supports LongBench-v2 static-gate evaluation.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _name, _value in {
    'NFA_OPERATOR_MODE': 'off',
    'NFA_FINE_GATHER': '0',
    'NFA_SPARSE_PREFILL': '0',
    'NFA_DECODE_DENSE': '0',
    'NFA_FROZEN_EAGER_REFERENCE': '0',
    'NFA_CUDA_GRAPH': 'auto',
}.items():
    os.environ[_name] = _value

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from benchmark.LongBench.lbv2_choice import (
    LBV2_PARSER_VERSION,
    extract_choice_official,
)
from sparse_attn.engine.frozen_runner import (
    complete_frozen_request,
    decode_frozen_request,
    prefill_frozen_request,
)
from sparse_attn.info import resolve_treatment_snapshot
from sparse_attn.ops.eager_decoding import eager_decoding_request_guard
from sparse_attn.ops.paged_region_kernels_nfa import nfa_args
from sparse_attn.patches import register_patch
from sparse_attn.policies.frozen_gate import GateConfig
from sparse_attn.policies.region_geometry import RegionGeometry
from sparse_attn.policies.static_gate import (
    default_static_gate_path,
    load_static_gate,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'glm-static-gate-lbv2-graph-ready-v2-material-suffix'


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def prepare_ids(tokenizer, row, dataset, prompt, context_cap):
    text = prompt.format(**row)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    original_length = len(ids)
    if original_length > context_cap:
        half = context_cap // 2
        text = tokenizer.decode(ids[:half], skip_special_tokens=True) + tokenizer.decode(
            ids[-(context_cap - half):], skip_special_tokens=True)
    messages = [dict(role='user', content=text)]
    material_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=False)
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    if not full_text.startswith(material_text):
        raise ValueError('chat generation suffix is not a strict prompt suffix')
    ids = tokenizer(full_text, add_special_tokens=False).input_ids
    material_ids = tokenizer(material_text, add_special_tokens=False).input_ids
    if not ids or ids[:len(material_ids)] != material_ids:
        raise ValueError('material tokenization is not a prefix of the full prompt')
    suffix_ids = ids[len(material_ids):]
    return material_ids, suffix_ids, {
        'raw_tokens': original_length,
        'input_tokens': len(ids),
        'material_tokens': len(material_ids),
        'suffix_tokens': len(suffix_ids),
        'truncated': original_length > context_cap,
        'input_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        'material_sha256': hashlib.sha256(
            json.dumps(material_ids).encode()).hexdigest(),
        'suffix_sha256': hashlib.sha256(
            json.dumps(suffix_ids).encode()).hexdigest(),
    }


def lbv2_sample_score(prediction, ground_truth):
    choice = extract_choice_official(prediction)
    return 1.0 if (
        choice is not None and choice == str(ground_truth).strip().upper()) else 0.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--dataset-path', required=True,
                        help='LongBench-v2 data.json or a directory containing it')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--static-gate-path', default=None,
                        help='override default repo static gate artifact')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0, help='limit rows for smoke')
    parser.add_argument('--skip', type=int, default=0,
                        help='skip the first N rows of this worker shard')
    parser.add_argument('--context-cap', type=int, default=0,
                        help='0 means model native context minus max_new_tokens')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--pred-postfix', default='_static_gate')
    return parser.parse_args()


def resolve_lbv2_path(value):
    path = Path(value)
    if path.is_file():
        return path
    for cand in ('lbv2.json', 'data.json'):
        child = path / cand
        if child.exists():
            return child
    raise SystemExit(f'lbv2 data not found under {value}')


def base_eos_ids(model):
    eos = model.generation_config.eos_token_id
    if eos is None:
        raise ValueError('model generation_config has no eos_token_id')
    return list(eos) if isinstance(eos, (list, tuple)) else [eos]


@torch.inference_mode()
@eager_decoding_request_guard()
def generate_static(model, ids, max_new_tokens, gate, geometry, static_gate, device,
                    suffix_ids=()):
    eos_ids = base_eos_ids(model)
    start = time.time()
    input_ids = torch.tensor([ids], device=device)
    output, audit = prefill_frozen_request(
        model, input_ids, gate, geometry, static_gate=static_gate,
        suffix_length=len(suffix_ids))
    generated, cache = decode_frozen_request(
        model, output, max_new_tokens=max_new_tokens, eos_ids=eos_ids,
        pad_token_id=getattr(model.config, 'pad_token_id', None) or 0,
        suffix_ids=suffix_ids)
    audit = complete_frozen_request(model, cache, len(generated), audit)
    return {
        'generated_ids': generated,
        'generated_tokens': len(generated),
        'wall_seconds': time.time() - start,
        '_frozen_region': audit,
    }


def main():
    args = parse_args()
    if not 0 <= args.rank < args.workers:
        raise ValueError('rank must be in [0, workers)')

    model_path = Path(args.model_path)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if getattr(config, 'model_type', None) != 'glm4_moe_lite':
        raise ValueError('static-gate LongBench-v2 currently requires glm4_moe_lite')
    if config.num_hidden_layers != 47:
        raise ValueError('GLM NFA expects 47 attention layers')

    treatment = nfa_args(subspace='nope')
    treatment.nfa_enable_static_gate = True
    static_gate_path = (
        args.static_gate_path
        or default_static_gate_path(config.model_type))
    static_gate = load_static_gate(config.model_type, path=static_gate_path)
    if static_gate is None:
        raise SystemExit(f'static gate artifact missing: {static_gate_path}')
    if static_gate.num_layers != config.num_hidden_layers:
        raise SystemExit('static gate layer count mismatch')

    device = torch.device('cuda')
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map='cuda', attn_implementation='sdpa').eval()

    n_patched = register_patch(
        model, treatment, resolve_treatment_snapshot(treatment))
    if n_patched != config.num_hidden_layers:
        raise ValueError(f'incomplete GLM NFA patch registration: {n_patched}')
    model._frozen_nfa = True

    max_new_tokens = args.max_new_tokens
    if args.context_cap > 0:
        context_cap = args.context_cap
    else:
        native_cap = min(
            128000, int(getattr(config, 'max_position_embeddings', 128000)))
        # Keep headroom for the chat-template overhead added after the
        # mid-truncation above; otherwise a 125.9K truncated prompt can still
        # exceed the tokenizer/model limit by a handful of template tokens.
        context_cap = max(1, native_cap - 2048)
    gate = GateConfig(mode='full', rows=256, statistic='mean')
    geometry = RegionGeometry(
        sink=4, near=4096, window=128,
        page=treatment.tia_block_size, max_new_tokens=max_new_tokens)

    data_path = resolve_lbv2_path(args.dataset_path)
    data = json.loads(data_path.read_text())
    if not isinstance(data, list):
        raise ValueError('LongBench-v2 data.json must be a JSON list')
    shard = data[args.rank::args.workers]
    if args.skip:
        shard = shard[args.skip:]
    if args.limit:
        shard = shard[:args.limit]

    prompt = json.loads(
        (ROOT / 'benchmark/LongBench/config/dataset2prompt.json').read_text())['lbv2']

    out_dir = Path(args.output_dir) / f'pred{args.pred_postfix}'
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f'lbv2-nfa_static_glm4-rank{args.rank:02d}.jsonl'
    ckpt_dir = out_dir / f'checkpoints_rank{args.rank:02d}'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_id = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    metadata = {
        'schema': SCHEMA,
        'run_id': run_id,
        'rank': args.rank,
        'workers': args.workers,
        'model_path': str(model_path.resolve()),
        'model_type': config.model_type,
        'static_gate_path': str(Path(static_gate_path).resolve()),
        'static_gate_skip_far': int(sum(static_gate.skip_far)),
        'static_gate_skip_near': int(sum(static_gate.skip_near)),
        'lbv2_parser': LBV2_PARSER_VERSION,
        'context_cap': context_cap,
        'max_new_tokens': max_new_tokens,
        'generation_protocol': (
            'material prefill; complete chat generation suffix consumed '
            'token-by-token; output contains answer tokens only'),
        'nfa_args': vars(treatment),
        'git_head': os.popen('git rev-parse HEAD').read().strip(),
    }
    atomic_json(out_dir / f'manifest_rank{args.rank:02d}.json', metadata)

    completed = {}
    for row in tqdm(shard, desc=f'rank{args.rank}'):
        row_id = str(row['_id'])
        ckpt_path = ckpt_dir / f'{row_id}.json'
        if ckpt_path.exists():
            completed[row_id] = json.loads(ckpt_path.read_text())
            continue
        material_ids, suffix_ids, input_info = prepare_ids(
            tokenizer, row, 'lbv2', prompt, context_cap)
        ids = material_ids + suffix_ids
        if len(ids) + max_new_tokens > int(
                getattr(config, 'max_position_embeddings', 128000)):
            raise ValueError('prompt plus generation exceeds native context')
        result = generate_static(
            model, material_ids, max_new_tokens, gate, geometry, static_gate, device,
            suffix_ids=suffix_ids)
        prediction = tokenizer.decode(
            result['generated_ids'], skip_special_tokens=True)
        pred_choice = extract_choice_official(prediction)
        answers = [row['answer']]
        all_classes = ['A', 'B', 'C', 'D']
        record = {
            'pred': prediction,
            'pred_choice': pred_choice,
            'answers': answers,
            'all_classes': all_classes,
            'length': row.get('length'),
            '_id': row_id,
            'score': lbv2_sample_score(prediction, row['answer']),
            'generated_tokens': result['generated_tokens'],
            'wall_seconds': result['wall_seconds'],
            'input_tokens': input_info['input_tokens'],
            'material_tokens': input_info['material_tokens'],
            'suffix_tokens': input_info['suffix_tokens'],
            'truncated': input_info['truncated'],
            'input_sha256': input_info['input_sha256'],
            'material_sha256': input_info['material_sha256'],
            'suffix_sha256': input_info['suffix_sha256'],
            '_frozen_region': result['_frozen_region'],
        }
        atomic_json(ckpt_path, record)
        completed[row_id] = record

    rows = [completed[str(row['_id'])] for row in shard]
    with open(pred_path, 'w', encoding='utf-8') as f:
        for record in rows:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    if rows:
        summary = {
            'schema': SCHEMA,
            'run_id': run_id,
            'rank': args.rank,
            'workers': args.workers,
            'n': len(rows),
            'accuracy': round(100 * sum(r['score'] for r in rows) / len(rows), 4),
            'pred_path': str(pred_path),
            'metadata': metadata,
        }
        atomic_json(out_dir / f'summary_rank{args.rank:02d}.json', summary)
    print(f'[static-gate-lbv2] rank={args.rank} n={len(rows)} '
          f'pred={pred_path}', flush=True)


if __name__ == '__main__':
    main()
