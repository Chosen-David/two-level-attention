"""Generalized GLM subspace paired runner.

Keeps the origin/main GLM TLI registration (subspace='nope' guard) and
overrides each layer's _subspace_indices with an arbitrary dimension list.
This lets us explore full256 / nope-{32,64,128} / rope-{32,64} / mixed
without weakening the fail-closed production guard.

Also records per-layer near/far skip decisions and calibrated profiles so
we can compare skip-layer consistency across RULER and LongBench AVG5.
"""
import argparse
import fcntl
import gc
import hashlib
import json
import os
import socket
import time
import traceback
from pathlib import Path
from types import MethodType

for key, val in dict(
    TLI_OPERATOR_MODE='off',
    TLI_FINE_GATHER='0',
    TLI_SPARSE_PREFILL='0',
    TLI_DECODE_DENSE='0',
    TLI_CUDA_GRAPH='0',
    TLI_SCORER_BACKEND='difflib',
    TLI_BATCH_INDEXER='legacy',
).items():
    os.environ[key] = val
os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/tli-triton-cache')

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from benchmark.LongBench.prefill_frozen_skip import (
    prepare_ids,
    atomic_json,
    sha_file,
    ROOT,
    SCREEN_TASKS,
)
from benchmark.LongBench.eval import sample_score
from benchmark.RULER.score_ruler import string_match_all
from sparse_attn.ops.region_geometry import RegionGeometry
from sparse_attn.ops.frozen_region_gate import GateConfig
from sparse_attn.ops.frozen_region_runtime import (
    prefill_frozen_request,
    complete_frozen_request,
)
from sparse_attn.ops.paged_region_tli import tli_args
from sparse_attn.info import resolve_treatment_snapshot
from sparse_attn.patches import register_patch
from sparse_attn.ops.eager_decoding import eager_decoding_request_guard


ARMS = ('dense', 'selected')
RULER_TASK = 'niah_multikey_2'
GATE_MODE = 'full'
GATE_EPSILON = 0.05
GATE_STATISTIC = 'mean'
SOURCE_COMMIT = '42821f81a4739d05d8bc81e1d4cc58307e34bb7e'


class CheckpointMismatch(ValueError):
    pass


def parse_dims(text):
    dims = sorted(set(int(x) for x in text.split(',') if x.strip()))
    if not dims or any(d < 0 or d >= 256 for d in dims):
        raise ValueError('dims must be unique integers in [0, 255]')
    return dims


def build_jobs(data_dir, ruler_path, tasks, ruler_limit, per_task):
    jobs = []
    if RULER_TASK in tasks:
        rows = [json.loads(x) for x in Path(ruler_path).read_text().splitlines()]
        n = min(ruler_limit or len(rows), len(rows))
        for i in range(n):
            jobs.append(dict(task=RULER_TASK, row=i, data=rows[i], kind='ruler'))

    for task in SCREEN_TASKS:
        if task not in tasks:
            continue
        rows = [
            json.loads(x) for x in (Path(data_dir) / f'{task}.jsonl').read_text().splitlines()
        ]
        n = min(per_task or len(rows), len(rows))
        for i in range(n):
            jobs.append(dict(task=task, row=i, data=rows[i], kind='longbench'))
    return jobs


def base_eos_ids(model):
    eos = model.generation_config.eos_token_id
    if eos is None:
        raise ValueError('model generation_config has no eos_token_id')
    return list(eos) if isinstance(eos, (list, tuple)) else [eos]


def ruler_eos_ids(model, tokenizer):
    ids = base_eos_ids(model)
    nl = tokenizer.encode('\n', add_special_tokens=False)
    if nl:
        ids.append(nl[-1])
    return list(dict.fromkeys(ids))


def append_prediction(out, gen):
    token = out.logits[:, -1].argmax(-1, keepdim=True)
    gen.append(int(token))
    return token


@torch.inference_mode()
@eager_decoding_request_guard()
def generate_native(model, ids, max_new_tokens, device, eos_ids, min_new_tokens=1):
    cache = DynamicCache(config=model.config)
    out = model(torch.tensor([ids], device=device), past_key_values=cache,
                use_cache=True, logits_to_keep=1)
    generated = []
    while len(generated) < max_new_tokens:
        token = append_prediction(out, generated)
        if len(generated) >= min_new_tokens and int(token) in eos_ids:
            break
        if len(generated) >= max_new_tokens:
            break
        out = model(torch.tensor([[generated[-1]]], device=device),
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
    return generated


@torch.inference_mode()
@eager_decoding_request_guard()
def generate_selected(model, ids, max_new_tokens, device, eos_ids, min_new_tokens=1):
    geometry = RegionGeometry(max_new_tokens=max_new_tokens)
    gate = GateConfig(mode=GATE_MODE, statistic=GATE_STATISTIC, epsilon=GATE_EPSILON)
    out, audit = prefill_frozen_request(
        model, torch.tensor([ids], device=device), gate, geometry)
    cache = out.past_key_values
    generated = []
    while len(generated) < max_new_tokens:
        token = append_prediction(out, generated)
        if len(generated) >= min_new_tokens and int(token) in eos_ids:
            break
        if len(generated) >= max_new_tokens:
            break
        out = model(torch.tensor([[generated[-1]]], device=device),
                    past_key_values=cache,
                    attention_mask=None,
                    use_cache=True, logits_to_keep=1)
    audit = complete_frozen_request(model, cache, len(generated), audit)
    return generated, audit


def patch_subspace(model, dims):
    fixed = tuple(sorted(dict.fromkeys(dims)))
    for layer in model.model.layers:
        idx = layer.self_attn.indexer
        idx._subspace_indices = MethodType(
            lambda self, device, dims=fixed: torch.tensor(dims, device=device, dtype=torch.long),
            idx,
        )
        idx._idx_sub_cache = None
        idx._qat_idx_cache = None
        idx._inc = None


def layer_state_arrays(audit, n_layers):
    decisions = audit.get('decisions', {})
    profiles = audit.get('profiles', {})
    states = []
    skip_near = []
    skip_far = []
    mean_near = []
    mean_far = []
    p95_near = []
    p95_far = []
    for layer_idx in range(n_layers):
        decision = decisions.get(layer_idx) or {}
        skip_n = bool(decision.get('skip_near', False))
        skip_f = bool(decision.get('skip_far', False))
        states.append('both' if skip_n and skip_f else
                      'near' if skip_n else
                      'far' if skip_f else 'keep')
        skip_near.append(skip_n)
        skip_far.append(skip_f)
        prof = profiles.get(layer_idx) or {}
        mean_near.append(float(prof.get('near', {}).get('mean', float('nan'))))
        mean_far.append(float(prof.get('far', {}).get('mean', float('nan'))))
        p95_near.append(float(prof.get('near', {}).get('p95', float('nan'))))
        p95_far.append(float(prof.get('far', {}).get('p95', float('nan'))))
    return dict(layer_states=states, skip_near=skip_near, skip_far=skip_far,
                profile_mean_near=mean_near, profile_mean_far=mean_far,
                profile_p95_near=p95_near, profile_p95_far=p95_far)


def main():
    ap = argparse.ArgumentParser()
    for name in ('model', 'data', 'ruler', 'out', 'model-manifest'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--rank', type=int, required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--model-type', choices=['glm', 'qwen'], default='glm')
    ap.add_argument('--dims', default='', help='comma separated GLM K dims in [0,255]')
    ap.add_argument('--schema', required=True)
    ap.add_argument('--selector-label', default='subspace')
    ap.add_argument('--ruler-limit', type=int, default=0, help='screen RULER rows; 0=all')
    ap.add_argument('--per-task', type=int, default=0, help='screen LB rows per task; 0=all')
    ap.add_argument('--limit', type=int, default=0, help='smoke only; first N jobs')
    ap.add_argument('--gate-mode', choices=['full','far','near','dual'], default='full')
    ap.add_argument('--epsilon', type=float, default=0.05)
    ap.add_argument('--statistic', choices=['mean','p95'], default='mean')
    ap.add_argument('--tasks', default='', help='comma separated subset; empty means all')
    args = ap.parse_args()
    global GATE_MODE, GATE_EPSILON, GATE_STATISTIC
    GATE_MODE, GATE_EPSILON, GATE_STATISTIC = args.gate_mode, args.epsilon, args.statistic

    if not 0 <= args.rank < args.workers:
        raise ValueError('rank outside worker range')
    dims = parse_dims(args.dims) if args.model_type == 'glm' else []
    if SOURCE_COMMIT != '42821f81a4739d05d8bc81e1d4cc58307e34bb7e':
        raise RuntimeError('source commit constant mismatch')

    task_filter = None
    if args.tasks:
        task_filter = set(x.strip() for x in args.tasks.split(',') if x.strip())
        allowed = set(SCREEN_TASKS) | {RULER_TASK}
        if not task_filter <= allowed:
            raise ValueError(f'unsupported tasks {task_filter - allowed}')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    worker_lock = (out / f'worker_rank{args.rank}.lock').open('a')
    fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    model_manifest = json.loads(Path(args.model_manifest).read_text())
    for name, metadata in model_manifest.items():
        if not (Path(args.model) / name).exists():
            raise RuntimeError(f'missing model file {name}')
        if (Path(args.model) / name).stat().st_size != metadata['size']:
            raise RuntimeError(f'model size mismatch {name}')
        if not name.endswith('.safetensors'):
            actual = sha_file(Path(args.model) / name)
            if actual != metadata['sha256']:
                raise RuntimeError(f'model hash mismatch {name}')

    prompts = json.loads((ROOT / 'benchmark/LongBench/config/dataset2prompt.json').read_text())
    lengths = json.loads((ROOT / 'benchmark/LongBench/config/dataset2maxlen.json').read_text())
    jobs = build_jobs(args.data, args.ruler, task_filter or set(SCREEN_TASKS) | {RULER_TASK},
                      args.ruler_limit, args.per_task)
    if args.limit:
        jobs = jobs[: args.limit]
    if not jobs:
        raise ValueError('empty job list')

    source = {
        str(p.relative_to(ROOT)): sha_file(p)
        for base in ('sparse_attn', 'benchmark/LongBench', 'benchmark/RULER')
        for p in (ROOT / base).rglob('*.py')
    }
    datasets_present = set(job['task'] for job in jobs)
    data_hashes = {}
    for task in datasets_present:
        if task == RULER_TASK:
            data_hashes[task] = sha_file(Path(args.ruler))
        else:
            data_hashes[task] = sha_file(Path(args.data) / f'{task}.jsonl')

    protocol = dict(
        schema=args.schema,
        source_commit=SOURCE_COMMIT,
        arms=list(ARMS),
        selector=args.selector_label,
        model_type=args.model_type,
        selector_dims=dims,
        screen=dict(ruler_limit=args.ruler_limit, per_task=args.per_task,
                    first_n=True),
        ruler_prompt='raw input, tokenizer default special tokens; no truncation',
        ruler_max_new=64,
        ruler_min_new=2,
        ruler_eos_policy='[eos_token_id, newline_token_id] official pred_ruler parity',
        longbench_min_new=1,
        gate_mode=args.gate_mode,
        epsilon=args.epsilon,
        statistic=args.statistic,
        seed=42,
        method='tli',
        subspace='nope',
        k1=128,
        k2=1024,
        sink=4,
        swa=128,
        near_len=4096,
        skip=False,
        scoring_backend='difflib',
        graph=False,
        operator='off',
        dtype='bfloat16',
        longbench_context_cap=31500,
        model=str(Path(args.model).resolve()),
        model_config=sha_file(Path(args.model) / 'config.json'),
        tokenizer_config=sha_file(Path(args.model) / 'tokenizer_config.json'),
        model_manifest=model_manifest,
        data=data_hashes,
        tasks=sorted(datasets_present),
        expected={t: sum(j['task'] == t for j in jobs) for t in sorted(datasets_present)},
        source=source,
        script_sha256=sha_file(__file__),
        prompt_config=sha_file(ROOT / 'benchmark/LongBench/config/dataset2prompt.json'),
        generation_config=sha_file(ROOT / 'benchmark/LongBench/config/dataset2maxlen.json'),
        torch=torch.__version__,
        transformers=transformers.__version__,
        workers=args.workers,
        scope='GLM subspace diagnostic on latest origin/main; not benchmark/paper acceptance',
    )
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()

    with (out / 'protocol.lock').open('a') as protocol_lock:
        fcntl.flock(protocol_lock, fcntl.LOCK_EX)
        shared = out / 'protocol.json'
        if shared.exists() and json.loads(shared.read_text())['identity'] != identity:
            raise CheckpointMismatch('shared run protocol mismatch')
        atomic_json(shared, dict(identity=identity, protocol=protocol))

    manifest = out / f'manifest_rank{args.rank}.json'
    if manifest.exists() and json.loads(manifest.read_text())['identity'] != identity:
        raise ValueError('run identity mismatch')
    atomic_json(manifest, dict(identity=identity, protocol=protocol, pid=os.getpid(),
                               host=socket.gethostname(), started_at=time.time()))

    torch.manual_seed(42)
    torch.set_num_threads(6)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa'
    ).eval()
    expected_type = 'glm4_moe_lite' if args.model_type == 'glm' else 'qwen3'
    if model.config.model_type != expected_type:
        raise ValueError(f'expected model_type={expected_type}, got {model.config.model_type}')

    native = [x.self_attn.forward for x in model.model.layers]
    treatment = tli_args(subspace='nope' if args.model_type == 'glm' else 'tail')
    if register_patch(model, treatment, resolve_treatment_snapshot(treatment)) != model.config.num_hidden_layers:
        raise RuntimeError('incomplete TLI patch registration')
    patched = [x.self_attn.forward for x in model.model.layers]
    if args.model_type == 'glm':
        patch_subspace(model, dims)

    device = 'cuda:0'
    primary = [i for i in range(len(jobs)) if i % args.workers == args.rank]
    order = primary + [i for i in range(len(jobs)) if i % args.workers != args.rank]
    failed_seen = []

    for i in order:
        job = jobs[i]
        task = job['task']
        row = job['row']
        data = job['data']
        dest = out / 'requests' / task / f'{row:04d}.json'
        dest.parent.mkdir(parents=True, exist_ok=True)
        lock = dest.with_suffix('.lock').open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            continue

        record = None
        try:
            record = json.loads(dest.read_text()) if dest.exists() else None
            if record and record['identity'] != identity:
                raise CheckpointMismatch('request identity mismatch')
            if record and record['status'] == 'failed':
                failed_seen.append(dict(task=task, row=row))
                continue

            if job['kind'] == 'longbench':
                ids, meta = prepare_ids(tokenizer, data, task, prompts[task], 31500)
                max_new = lengths[task]
                min_new = 1
                answers = data['answers']
                eos_ids = base_eos_ids(model)
            else:
                ids = tokenizer(data['input'], add_special_tokens=True).input_ids
                meta = dict(
                    input_tokens=len(ids),
                    truncated=False,
                    input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                )
                max_new = 64
                min_new = 2
                answers = data['outputs']
                eos_ids = ruler_eos_ids(model, tokenizer)

            if len(ids) + max_new > model.config.max_position_embeddings:
                raise ValueError('context exceeds native model limit')
            if record and (
                record['input']['input_sha256'] != meta['input_sha256']
                or record['answers'] != answers
            ):
                raise CheckpointMismatch('request input/answers differ from checkpoint')
            if record and record['status'] == 'complete':
                if set(record['arms']) != set(ARMS):
                    raise CheckpointMismatch('incomplete completed record')
                continue

            record = record or dict(
                identity=identity,
                task=task,
                row=row,
                source_index=data.get('index', row),
                input=meta,
                answers=answers,
                arms={},
                status='running',
            )
            atomic_json(dest, record)

            for arm in ARMS:
                if arm in record['arms']:
                    continue
                atomic_json(
                    out / f'heartbeat_rank{args.rank}.json',
                    dict(pid=os.getpid(), time=time.time(), task=task, row=row, arm=arm, status='running'),
                )
                for layer, forward in zip(model.model.layers, native if arm == 'dense' else patched):
                    layer.self_attn.forward = forward
                model._frozen_tli = arm != 'dense'
                for layer in model.model.layers:
                    layer.self_attn.indexer.clear()
                torch.manual_seed(42)
                gc.collect()
                torch.cuda.empty_cache()
                with torch.inference_mode():
                    with eager_decoding_request_guard(device):
                        if arm == 'dense':
                            tokens = generate_native(model, ids, max_new, device, eos_ids, min_new)
                        else:
                            tokens, audit = generate_selected(model, ids, max_new, device, eos_ids, min_new)
                prediction = tokenizer.decode(tokens, skip_special_tokens=True)
                if arm == 'selected' and tokens[0] != record['arms']['dense']['generated_ids'][0]:
                    raise ValueError('dense prefill first prediction changed')
                if job['kind'] == 'ruler':
                    score_pct = string_match_all([prediction], [answers])
                else:
                    metric_ds = 'repobench' if task == 'repobench-p' else task
                    score_pct = 100 * sample_score(metric_ds, prediction, answers, data.get('all_classes', []))
                arm_record = dict(
                    generated_ids=tokens,
                    generated_tokens=len(tokens),
                    prediction=prediction,
                    raw_text=tokenizer.decode(tokens),
                    score_pct=score_pct,
                    route=('native_dense_sdpa' if arm == 'dense'
                           else f'{args.model_type}_{args.schema}_tli_paged_origin_main'),
                )
                if arm == 'selected':
                    arm_record['tli_audit_summary'] = dict(
                        status='complete',
                        gate_counts=audit.get('gate_counts'),
                        **layer_state_arrays(audit, model.config.num_hidden_layers),
                    )
                record['arms'][arm] = arm_record
                record['status'] = 'complete' if len(record['arms']) == len(ARMS) else 'running'
                atomic_json(dest, record)
                print(json.dumps(dict(rank=args.rank, task=task, row=row, arm=arm,
                                      tokens=len(tokens), score_pct=score_pct)), flush=True)
        except CheckpointMismatch:
            raise
        except Exception as exc:
            failure = record or dict(identity=identity, task=task, row=row, arms={})
            failure.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
            atomic_json(dest, failure)
            print(traceback.format_exc(), flush=True)
            raise
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()

    atomic_json(
        out / f'worker_done_rank{args.rank}.json',
        dict(identity=identity, pid=os.getpid(), time=time.time(),
             status='scan_finished_not_global_acceptance', failed_seen=failed_seen),
    )


if __name__ == '__main__':
    main()
