"""Paired screen: far eviction, near staging, and prefill-frozen region gates."""
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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from benchmark.LongBench.eval import sample_score
from sparse_attn.ops.frozen_region_cache import RegionGeometry
from sparse_attn.ops.frozen_region_gate import GateConfig
from sparse_attn.ops.frozen_region_runtime import (
    complete_frozen_request, prefill_frozen_request)
from sparse_attn.ops.eager_decoding import eager_decoding_request_guard

ROOT = Path(__file__).resolve().parents[2]
SCREEN_TASKS = ('qasper', 'hotpotqa', 'musique', 'gov_report', 'repobench-p')
SOURCES = ('sparse_attn/patches/glm4_moe_lite_attn_patch.py',
           'sparse_attn/patches/qwen3_attn_patch.py', 'sparse_attn/ops/frozen_region_tli.py',
           'sparse_attn/indexer/tli_indexer.py', 'sparse_attn/ops/frozen_region_cache.py', 'sparse_attn/ops/frozen_region_gate.py',
           'benchmark/LongBench/prefill_frozen_skip.py', 'benchmark/LongBench/metrics.py',
           'benchmark/LongBench/eval.py', 'benchmark/LongBench/config/dataset2prompt.json',
           'benchmark/LongBench/config/dataset2maxlen.json')


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def prepare_ids(tokenizer, row, dataset, prompt, context_cap):
    text = prompt.format(**row)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    original_length = len(ids)
    if original_length > context_cap:
        half = context_cap//2
        text = tokenizer.decode(ids[:half], skip_special_tokens=True)+tokenizer.decode(ids[-(context_cap-half):], skip_special_tokens=True)
    if dataset != 'repobench-p':
        text = tokenizer.apply_chat_template([dict(role='user', content=text)], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError('empty prompt')
    return ids, dict(raw_tokens=original_length, input_tokens=len(ids), truncated=original_length>context_cap,
                     input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())


@torch.inference_mode()
@eager_decoding_request_guard()
def generate(model, tokenizer, ids, max_new_tokens, config, device):
    geometry = RegionGeometry(max_new_tokens=max_new_tokens)
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    start = time.time()
    with torch.inference_mode():
        out, audit = prefill_frozen_request(
            model, torch.tensor([ids], device=device), config, geometry)
        cache = out.past_key_values
        generated = []
        for _ in range(max_new_tokens):
            token = out.logits[:, -1].argmax(-1, keepdim=True)
            generated.append(int(token))
            if int(token) in eos or len(generated) == max_new_tokens:
                break
            out = model(token, past_key_values=cache,
                        attention_mask=(None if model.config.model_type == 'glm4_moe_lite'
                                        else {'full_attention':None}),
                        use_cache=True, logits_to_keep=1)
        audit = complete_frozen_request(model, cache, len(generated), audit)
    result = dict(generated_ids=generated, prediction=tokenizer.decode(generated, skip_special_tokens=True),
        generated_tokens=len(generated), wall_seconds=time.time()-start, **audit,
        measured_scope='physical node KV storage; eager gather temporaries excluded, no latency claim')
    del out, cache
    return result


@torch.inference_mode()
@eager_decoding_request_guard()
def generate_native(model, ids, max_new_tokens, device):
    cache = DynamicCache(config=model.config)
    if getattr(model, '_frozen_tli', False):
        for layer in model.model.layers:
            layer.self_attn.indexer.clear()
            layer.self_attn.indexer.skip_far = False
    out = model(torch.tensor([ids], device=device), past_key_values=cache, use_cache=True, logits_to_keep=1)
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos,list) else [eos]
    generated = []
    for _ in range(max_new_tokens):
        token = out.logits[:, -1].argmax(-1, keepdim=True); generated.append(int(token))
        if int(token) in eos or len(generated) == max_new_tokens: break
        out = model(token, past_key_values=cache, use_cache=True, logits_to_keep=1)
    return generated


def select_row_indices(row_count, samples, seed):
    """Zero selects the entire dataset; positive counts keep legacy ordering."""
    if samples < 0 or row_count < 0:
        raise ValueError('row count and samples must be nonnegative')
    permutation = np.random.default_rng(seed).permutation(row_count)
    return (permutation if samples == 0 else permutation[:samples]).tolist()


def sampling_metadata(samples, dataset_counts):
    if samples == 0:
        return dict(samples_per_task=None, full_dataset=True,
                    dataset_counts=dict(dataset_counts),
                    selected_counts=dict(dataset_counts))
    # Keep positive-sample protocol fields identical to historical runs.
    return dict(samples_per_task=samples)


def shard_inputs(inputs, rank, world_size):
    if not 0 <= rank < world_size:
        raise ValueError('invalid shard')
    for index, item in enumerate(inputs):
        if index % world_size == rank:
            yield item


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--model', default='/home/wangyuanshuo02/Qwen3-8B')
    ap.add_argument('--data', default='/home/wangyuanshuo02/datasets/LongBench/data')
    ap.add_argument('--out', default='agent_doc/results/frozen_near_far_staging_20261029')
    ap.add_argument('--samples', type=int, default=20, help='rows per task; 0 selects every row')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--context-cap', type=int, default=31500)
    ap.add_argument('--thresholds', default='0.05')
    ap.add_argument('--statistic', choices=['mean','p95'], default='mean')
    ap.add_argument('--datasets', default=','.join(SCREEN_TASKS))
    ap.add_argument('--rank', type=int, default=0)
    ap.add_argument('--world-size', type=int, default=1)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--verify-native-full', action='store_true')
    ap.add_argument('--attention', choices=['sdpa','tli'], default='sdpa')
    args = ap.parse_args()
    if args.attention == 'tli':
        os.environ['TLI_OPERATOR_MODE'] = 'off'
        os.environ['TLI_FINE_GATHER'] = '0'
        os.environ['TLI_SPARSE_PREFILL'] = '0'
        os.environ['TLI_DECODE_DENSE'] = '0'
    tasks = args.datasets.split(',')
    if not set(tasks)<=set(SCREEN_TASKS) or len(set(tasks)) != len(tasks):
        raise ValueError('only the five existing LongBench screen tasks are allowed')
    if not 0 <= args.rank < args.world_size or args.samples < 0:
        raise ValueError('invalid shard/samples')
    thresholds = [float(x) for x in args.thresholds.split(',')]
    arms = [('full', GateConfig(mode='full', statistic=args.statistic))]
    for eps in thresholds:
        for mode in ('far','dual'):
            config = GateConfig(epsilon=eps, mode=mode, statistic=args.statistic)
            arms.append((f'{mode}_{eps:g}', config))
    prompts = json.loads((ROOT/'benchmark/LongBench/config/dataset2prompt.json').read_text())
    lengths = json.loads((ROOT/'benchmark/LongBench/config/dataset2maxlen.json').read_text())
    selected, inputs, dataset_counts = {}, [], {}
    for ds in tasks:
        rows = [json.loads(line) for line in (Path(args.data)/(ds+'.jsonl')).read_text().splitlines()]
        dataset_counts[ds] = len(rows)
        indices = select_row_indices(len(rows), args.samples, args.seed)
        selected[ds] = indices
        inputs.extend((ds,i,rows[i]) for i in indices)
    protocol = dict(schema='frozen-near-far-screen-v3-independent-gates',model=args.model,context_cap=args.context_cap,
        **sampling_metadata(args.samples, dataset_counts),
        seed=args.seed,statistic=args.statistic,thresholds=thresholds,
        datasets=tasks,selected_rows=selected,max_new_tokens={ds:lengths[ds] for ds in tasks},
        source_sha256={name:sha_file(ROOT/name) for name in SOURCES},
        dataset_sha256={ds:sha_file(Path(args.data)/(ds+'.jsonl')) for ds in tasks},
        model_config_sha256=sha_file(Path(args.model)/'config.json'),
        tokenizer_config_sha256=sha_file(Path(args.model)/'tokenizer_config.json'),
        torch=torch.__version__,transformers=transformers.__version__,dtype='bfloat16',world_size=args.world_size,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        eviction='far-only: evict far; near-only: keep attention-invisible near staging and migrate to far; both: sink+SWA only',
        calibration='query-relative page boundaries, last256 rows; head means; independent near/far mass comparisons, no joint budget')
    if args.attention == 'tli':
        from sparse_attn.ops.frozen_region_tli import tli_args
        model_type = AutoConfig.from_pretrained(args.model).model_type
        treatment = tli_args(
            subspace='nope' if model_type == 'glm4_moe_lite' else 'tail')
        protocol['attention'] = ('tli-nope64-frozen-regions-reference'
                                 if model_type == 'glm4_moe_lite' else
                                 'tli-tail32-frozen-regions-reference')
        protocol['tli_args'] = vars(treatment)
        protocol['tli_environment'] = {name:os.environ[name] for name in
            ('TLI_OPERATOR_MODE','TLI_FINE_GATHER','TLI_SPARSE_PREFILL','TLI_DECODE_DENSE')}
        for path in sorted((ROOT/'sparse_attn').rglob('*.py')):
            protocol['source_sha256'][str(path.relative_to(ROOT))] = sha_file(path)
    identity = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    outdir = Path(args.out); outdir.mkdir(parents=True,exist_ok=True)
    import fcntl
    worker_lock = (outdir/f'worker_rank{args.rank}.lock').open('a+')
    try: fcntl.flock(worker_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit('this rank already has a live worker')
    manifest = outdir/f'manifest_rank{args.rank}.json'
    if manifest.exists() and json.loads(manifest.read_text())['identity'] != identity:
        raise ValueError('existing run identity differs; use a new output directory')
    atomic_json(manifest, dict(identity=identity,protocol=protocol,rank=args.rank,device=args.device,
                               pid=os.getpid(),started_at=time.time()))
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,
            device_map=args.device,attn_implementation='sdpa').eval()
    if model.config.model_type not in ('qwen3', 'glm4_moe_lite') or any(getattr(m.self_attn,'sliding_window',None) is not None for m in model.model.layers):
        raise ValueError('only full-attention Qwen3 or GLM-4.7-Flash supported')
    if model.config.model_type == 'glm4_moe_lite' and args.attention != 'tli':
        raise ValueError('GLM frozen regions currently require the TLI adapter')
    if args.attention == 'tli':
        from sparse_attn.patches import register_patch
        from sparse_attn.info import resolve_treatment_snapshot
        if register_patch(model,treatment,resolve_treatment_snapshot(treatment)) != model.config.num_hidden_layers:
            raise ValueError('incomplete TLI patch registration')
        model._frozen_tli = True
    verified = False
    for ds,row_index,row in shard_inputs(inputs, args.rank, args.world_size):
        path = outdir/'requests'/ds/f'{row_index:04d}.json'
        ids, input_info = prepare_ids(tokenizer,row,ds,prompts[ds],args.context_cap)
        if len(ids)+lengths[ds] > model.config.max_position_embeddings:
            raise ValueError('prompt plus generation exceeds native context')
        if path.exists():
            record = json.loads(path.read_text())
            if record['identity'] != identity or record['input']['input_sha256'] != input_info['input_sha256']:
                raise ValueError('incompatible checkpoint identity')
        else:
            record = dict(identity=identity,dataset=ds,row_index=row_index,input=input_info,
                          answers=row['answers'],arms={},status='running')
        for arm, config in arms:
            if arm in record['arms']: continue
            atomic_json(outdir/f'heartbeat_rank{args.rank}.json',dict(pid=os.getpid(),time=time.time(),dataset=ds,row=row_index,arm=arm))
            result = generate(model,tokenizer,ids,lengths[ds],config,args.device)
            metric_ds = 'repobench' if ds == 'repobench-p' else ds
            result['score'] = sample_score(metric_ds,result['prediction'],row['answers'],row.get('all_classes',[]))
            if arm == 'full' and args.verify_native_full and not verified:
                native = generate_native(model,ids,lengths[ds],args.device)
                if native != result['generated_ids']:
                    raise ValueError('full node-cache generation differs from native DynamicCache')
                result['native_full_parity'] = True
                verified = True
            if arm != 'full' and result['generated_ids'][0] != record['arms']['full']['generated_ids'][0]:
                raise ValueError('dense prefill first prediction changed')
            record['arms'][arm] = result
            atomic_json(path,record)
            print(f'rank={args.rank} {ds}:{row_index} {arm} score={result["score"]:.4f} gates={result["gate_counts"]}',flush=True)
            torch.cuda.empty_cache()
        record['status'] = 'complete'; atomic_json(path,record)
        atomic_json(outdir/f'heartbeat_rank{args.rank}.json',dict(pid=os.getpid(),time=time.time(),dataset=ds,row=row_index,status='complete'))
    atomic_json(outdir/f'worker_done_rank{args.rank}.json',dict(identity=identity,pid=os.getpid(),time=time.time()))


if __name__ == '__main__': main()
