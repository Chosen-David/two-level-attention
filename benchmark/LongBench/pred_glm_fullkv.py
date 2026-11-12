"""GLM-4.7-Flash FullKV LongBench v1 with SDPA or FlashInfer decode."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from benchmark.LongBench.eval import sample_score
from benchmark.LongBench.prefill_frozen_skip import (
    atomic_json, select_row_indices, sha_file, shard_inputs)

ROOT = Path(__file__).resolve().parents[2]
LONGBENCH_TASKS = (
    'narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa', 'musique',
    'gov_report', 'qmsum', 'multi_news', 'trec', 'triviaqa', 'samsum',
    'passage_count', 'passage_retrieval_en', 'lcc', 'repobench-p')
NO_CHAT_TASKS = frozenset(('trec', 'triviaqa', 'samsum', 'lcc', 'repobench-p'))
SOURCES = (
    'benchmark/LongBench/pred_glm_fullkv.py',
    'benchmark/LongBench/eval.py',
    'benchmark/LongBench/metrics.py',
    'benchmark/LongBench/config/dataset2prompt.json',
    'benchmark/LongBench/config/dataset2maxlen.json',
    'sparse_attn/patches/glm4_moe_lite_flashinfer_patch.py',
    'sparse_attn/patches/qwen3_flashinfer_patch.py',
)


def prepare_ids(tokenizer, row, dataset, prompt, context_cap):
    text = prompt.format(**row)
    raw_ids = tokenizer(text, add_special_tokens=False).input_ids
    original_length = len(raw_ids)
    if original_length > context_cap:
        half = context_cap // 2
        text = (tokenizer.decode(raw_ids[:half], skip_special_tokens=True)
                + tokenizer.decode(raw_ids[-(context_cap - half):],
                                   skip_special_tokens=True))
    material_text = text
    if dataset not in NO_CHAT_TASKS:
        messages = [dict(role='user', content=text)]
        material_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False)
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        if not full_text.startswith(material_text):
            raise ValueError('chat generation suffix is not a strict prompt suffix')
    else:
        full_text = material_text
    ids = tokenizer(full_text, add_special_tokens=False).input_ids
    material_ids = tokenizer(material_text, add_special_tokens=False).input_ids
    if not ids or ids[:len(material_ids)] != material_ids:
        raise ValueError('material tokenization is not a prefix of the full prompt')
    suffix_ids = ids[len(material_ids):]
    return ids, dict(
        raw_tokens=original_length, input_tokens=len(ids),
        material_tokens=len(material_ids), suffix_tokens=len(suffix_ids),
        truncated=original_length > context_cap,
        input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        material_sha256=hashlib.sha256(
            json.dumps(material_ids).encode()).hexdigest(),
        suffix_sha256=hashlib.sha256(json.dumps(suffix_ids).encode()).hexdigest())


def stopping_policy(model, tokenizer, dataset):
    eos = model.generation_config.eos_token_id
    eos = list(eos) if isinstance(eos, list) else [eos]
    if dataset == 'samsum':
        return list(dict.fromkeys([tokenizer.eos_token_id, tokenizer.encode(
            '\n', add_special_tokens=False)[-1]])), 1
    return eos, 0


def greedy_token(logits, generated_count, eos, min_new_tokens):
    if generated_count < min_new_tokens:
        logits = logits.clone()
        logits[:, eos] = -torch.inf
    return logits.argmax(-1, keepdim=True)


@torch.inference_mode()
def generate_sdpa(model, ids, max_new_tokens, device, stop_ids, min_new_tokens):
    start = time.perf_counter()
    cache = DynamicCache(config=model.config)
    out = model(torch.tensor([ids], device=device), past_key_values=cache,
                use_cache=True, logits_to_keep=1)
    generated = []
    token = greedy_token(out.logits[:, -1], 0, stop_ids, min_new_tokens)
    generated.append(int(token))
    while len(generated) < max_new_tokens and not (
            len(generated) >= min_new_tokens and int(token) in stop_ids):
        out = model(token, past_key_values=cache, use_cache=True, logits_to_keep=1)
        token = greedy_token(
            out.logits[:, -1], len(generated), stop_ids, min_new_tokens)
        generated.append(int(token))
    return generated, {
        'backend': 'sdpa', 'attention': 'fullkv',
        'wall_seconds': time.perf_counter() - start}


@torch.inference_mode()
def generate_flashinfer(model, ids, max_new_tokens, device, stop_ids, min_new_tokens):
    from sparse_attn.patches.glm4_moe_lite_flashinfer_patch import (
        FlashInferGLM4MoeLiteRuntime,
        bind_glm4_moe_lite_flashinfer_runtime,
        clear_glm4_moe_lite_flashinfer_runtime)

    start = time.perf_counter()
    length = len(ids)
    runtime = FlashInferGLM4MoeLiteRuntime(
        model, batch_size=1, prefill_length=length,
        max_new_tokens=max_new_tokens, device=device, page_size=64)
    bind_glm4_moe_lite_flashinfer_runtime(model, runtime)
    try:
        cache_position = torch.arange(length, device=device)
        out = model(
            input_ids=torch.tensor([ids], device=device),
            position_ids=cache_position.unsqueeze(0),
            cache_position=cache_position,
            past_key_values=runtime.mask_metadata,
            use_cache=True, logits_to_keep=1)
        token = greedy_token(out.logits[:, -1], 0, stop_ids, min_new_tokens)
        generated = [int(token)]
        runtime.prepare_decode('eager', max_new_tokens - 1)
        step = 0
        while len(generated) < max_new_tokens and not (
                len(generated) >= min_new_tokens and int(token) in stop_ids):
            runtime.prepare_decode_step(step)
            position = torch.tensor([[length + step]], device=device)
            out = model(
                input_ids=token, position_ids=position,
                cache_position=position[0],
                past_key_values=runtime.mask_metadata,
                use_cache=True, logits_to_keep=1)
            token = greedy_token(
                out.logits[:, -1], len(generated), stop_ids, min_new_tokens)
            generated.append(int(token))
            step += 1
        decode_steps = len(generated) - 1
        layers = model.config.num_hidden_layers
        assert runtime.prefill_append_calls == layers
        assert runtime.decode_append_calls == layers * decode_steps
        assert runtime.decode_wrapper_calls == layers * decode_steps
        execution = {
            'backend': 'flashinfer', 'attention': 'fullkv',
            'wrapper': type(runtime.wrapper).__name__, 'page_size': runtime.page_size,
            'prefill_append_calls': runtime.prefill_append_calls,
            'decode_append_calls': runtime.decode_append_calls,
            'decode_wrapper_calls': runtime.decode_wrapper_calls,
        }
        return generated, dict(execution, wall_seconds=time.perf_counter() - start)
    finally:
        clear_glm4_moe_lite_flashinfer_runtime(model)
        del runtime


def parse_tasks(value):
    tasks = value.split(',')
    if not set(tasks) <= set(LONGBENCH_TASKS) or len(set(tasks)) != len(tasks):
        raise ValueError('datasets must be unique LongBench v1 task names')
    return tasks


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--backend', choices=['sdpa', 'flashinfer'], default='flashinfer')
    parser.add_argument('--datasets', default=','.join(LONGBENCH_TASKS))
    parser.add_argument('--samples', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--context-cap', type=int, default=31500)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.samples < 0 or not 0 <= args.rank < args.world_size:
        raise ValueError('invalid samples or shard')
    tasks = parse_tasks(args.datasets)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(8)

    prompts = json.loads((ROOT / 'benchmark/LongBench/config/dataset2prompt.json').read_text())
    lengths = json.loads((ROOT / 'benchmark/LongBench/config/dataset2maxlen.json').read_text())
    selected, inputs, dataset_counts = {}, [], {}
    for dataset in tasks:
        rows = [json.loads(line) for line in (
            Path(args.data) / f'{dataset}.jsonl').read_text().splitlines()]
        dataset_counts[dataset] = len(rows)
        indices = select_row_indices(len(rows), args.samples, args.seed)
        selected[dataset] = indices
        inputs.extend((dataset, index, rows[index]) for index in indices)
    protocol = dict(
        schema='glm-fullkv-longbench-v1', model=args.model,
        attention='fullkv', backend=args.backend, page_size=64,
        context_cap=args.context_cap, seed=args.seed,
        full_dataset=args.samples == 0, samples=args.samples,
        datasets=tasks, selected_rows=selected, dataset_counts=dataset_counts,
        max_new_tokens={name: lengths[name] for name in tasks},
        prompt_protocol=('native LongBench; head/tail cap; no-chat for '
                         'trec/triviaqa/samsum/lcc/repobench-p; samsum EOS+newline min1'),
        source_sha256={name: sha_file(ROOT / name) for name in SOURCES},
        dataset_sha256={name: sha_file(Path(args.data) / f'{name}.jsonl') for name in tasks},
        model_config_sha256=sha_file(Path(args.model) / 'config.json'),
        tokenizer_config_sha256=sha_file(Path(args.model) / 'tokenizer_config.json'),
        torch=torch.__version__, transformers=transformers.__version__,
        dtype='bfloat16', world_size=args.world_size,
        git_head=(os.environ.get('GIT_HEAD') or subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    import fcntl
    worker_lock = (outdir / f'worker_rank{args.rank}.lock').open('a+')
    try:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('this rank already has a live worker')
    manifest = outdir / f'manifest_rank{args.rank}.json'
    if manifest.exists() and json.loads(manifest.read_text())['identity'] != identity:
        raise ValueError('existing run identity differs; use a new output directory')
    atomic_json(manifest, dict(identity=identity, protocol=protocol, rank=args.rank,
                               device=args.device, pid=os.getpid(), started_at=time.time()))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device,
        attn_implementation='sdpa').eval()
    if model.config.model_type != 'glm4_moe_lite':
        raise ValueError('this runner only supports GLM-4.7-Flash')
    if args.backend == 'flashinfer':
        from sparse_attn.patches.glm4_moe_lite_flashinfer_patch import (
            register_glm4_moe_lite_flashinfer_patch)
        patched = register_glm4_moe_lite_flashinfer_patch(model)
        if patched != model.config.num_hidden_layers:
            raise RuntimeError(f'incomplete FlashInfer patch: {patched} layers')

    generate = generate_flashinfer if args.backend == 'flashinfer' else generate_sdpa
    for dataset, row_index, row in shard_inputs(inputs, args.rank, args.world_size):
        stop_ids, min_new_tokens = stopping_policy(model, tokenizer, dataset)
        path = outdir / 'requests' / dataset / f'{row_index:04d}.json'
        ids, input_info = prepare_ids(
            tokenizer, row, dataset, prompts[dataset], args.context_cap)
        if len(ids) + lengths[dataset] > model.config.max_position_embeddings:
            raise ValueError('prompt plus generation exceeds native context')
        if path.exists():
            record = json.loads(path.read_text())
            if (record['identity'] != identity
                    or record['input']['input_sha256'] != input_info['input_sha256']):
                raise ValueError('incompatible checkpoint identity')
            continue
        atomic_json(outdir / f'heartbeat_rank{args.rank}.json', dict(
            pid=os.getpid(), time=time.time(), dataset=dataset, row=row_index))
        generated, execution = generate(
            model, ids, lengths[dataset], args.device, stop_ids, min_new_tokens)
        prediction = tokenizer.decode(generated, skip_special_tokens=True)
        metric_dataset = 'repobench' if dataset == 'repobench-p' else dataset
        score = sample_score(metric_dataset, prediction, row['answers'],
                             row.get('all_classes', []))
        record = dict(
            identity=identity, dataset=dataset, row_index=row_index,
            input=input_info, answers=row['answers'], generated_ids=generated,
            prediction=prediction, generated_tokens=len(generated), score=score,
            status='complete', **execution)
        atomic_json(path, record)
        print(f'rank={args.rank} {dataset}:{row_index} backend={args.backend} '
              f'score={score:.4f} seconds={execution["wall_seconds"]:.3f}', flush=True)
        del generated, execution
        torch.cuda.empty_cache()
    atomic_json(outdir / f'worker_done_rank{args.rank}.json', dict(
        identity=identity, pid=os.getpid(), time=time.time()))


if __name__ == '__main__':
    main()
