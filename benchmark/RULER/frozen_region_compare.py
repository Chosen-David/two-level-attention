"""Paired RULER64K current dual-vs-far gates, using the frozen LongBench engine.

Source prompts are already formatted: no chat wrapping or context truncation.
The existing RULER producer ignores EOS at position zero, then stops at EOS.
"""
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
from benchmark.LongBench.prefill_frozen_skip import (
    ROOT, atomic_json, sha_file, generate, generate_native)
from benchmark.RULER.score_ruler import string_match_all
from benchmark.RULER.yarn_receipt import resolve_yarn_config
from sparse_attn.ops.frozen_region_gate import GateConfig
from sparse_attn.ops.frozen_region_tli import tli_args

TASK = 'niah_multikey_2'
ARMS = ('full', 'far_0.05', 'dual_0.05')
ENVIRONMENT = dict(TLI_OPERATOR_MODE='off', TLI_FINE_GATHER='0',
    TLI_SPARSE_PREFILL='0', TLI_DECODE_DENSE='0', TLI_CUDA_GRAPH='0')


def row_ids(tokenizer, row):
    return tokenizer(row['input'], truncation=False).input_ids


def score(prediction, answers):
    if not answers:
        raise ValueError('empty reference answers')
    return float(string_match_all([prediction], [answers])) / 100


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--model', default='/home/wangyuanshuo02/Qwen3-8B')
    ap.add_argument('--data', default='/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER/65536/niah_multikey_2.jsonl')
    ap.add_argument('--out', required=True)
    ap.add_argument('--samples', type=int, default=100)
    ap.add_argument('--verify-native-full', action='store_true')
    args = ap.parse_args()
    if not 1 <= args.samples <= 100:
        raise ValueError('samples must be 1..100')
    os.environ.update(ENVIRONMENT)
    torch.set_num_threads(8)
    torch.manual_seed(42)
    np.random.seed(42)
    data = Path(args.data)
    all_rows = [json.loads(line) for line in data.read_text().splitlines()]
    if len(all_rows) != 100 or len({r['index'] for r in all_rows}) != 100:
        raise ValueError('requires the existing 100 unique RULER samples')
    rows = all_rows[:args.samples]
    factor, rope = resolve_yarn_config(True, 2.0, 65536, {65536: 2.0}, 40960)
    sources = list((ROOT/'sparse_attn').rglob('*.py')) + [
        ROOT/'benchmark/LongBench/prefill_frozen_skip.py',
        ROOT/'benchmark/RULER/frozen_region_compare.py',
        ROOT/'benchmark/RULER/validate_frozen_compare.py',
        ROOT/'benchmark/RULER/score_ruler.py', ROOT/'benchmark/RULER/yarn_receipt.py']
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    stop_ids = list(dict.fromkeys([tokenizer.eos_token_id, tokenizer.encode('\n', add_special_tokens=False)[-1]]))
    protocol = dict(stop_ids=stop_ids, schema='frozen-ruler-dual-vs-far-v1', dataset=TASK, context_length=65536,
        model=args.model, data=str(data), data_sha256=sha_file(data), samples=args.samples,
        selected_indices=[r['index'] for r in rows], arms=list(ARMS), epsilon=0.05,
        statistic='mean', calibration_rows=256, independent_gates=True,
        max_new_tokens=64, stop_after_tokens=2, seed=42, dtype='bfloat16', batch=1,
        prompt='tokenizer(raw source input, truncation=False); no additional chat template',
        rope_scaling=rope, effective_yarn_factor=factor,
        model_config_sha256=sha_file(Path(args.model)/'config.json'),
        tokenizer_config_sha256=sha_file(Path(args.model)/'tokenizer_config.json'),
        generation_config_sha256=sha_file(Path(args.model)/'generation_config.json'),
        tli_args=vars(tli_args()), environment=ENVIRONMENT,
        torch=torch.__version__, transformers=transformers.__version__,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)):sha_file(p) for p in sources},
        comparison='primary dual_0.05 minus far_0.05; gates-off auxiliary, not dense FullKV')
    identity = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    out = Path(args.out); out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock = (out/'worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest = out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['identity'] != identity:
        raise ValueError('run identity changed; use a fresh output directory')
    atomic_json(manifest, dict(identity=identity,protocol=protocol,pid=os.getpid(),
        started_at=time.time(),device=torch.cuda.get_device_name(0),
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True)))
    config = AutoConfig.from_pretrained(args.model)
    if config.model_type != 'qwen3': raise ValueError('Qwen3 only')
    config.rope_scaling = rope
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config,
        dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa').eval()
    if any(getattr(m.self_attn,'sliding_window',None) is not None for m in model.model.layers):
        raise ValueError('requires full attention layers')
    from sparse_attn.patches import register_patch
    from sparse_attn.info import resolve_treatment_snapshot
    treatment = tli_args()
    if register_patch(model,treatment,resolve_treatment_snapshot(treatment)) != config.num_hidden_layers:
        raise ValueError('incomplete TLI registration')
    model._frozen_tli = True
    native_verified = False
    for row in rows:
        idx = row['index']; path = out/'requests'/f'{idx:04d}.json'
        ids = row_ids(tokenizer,row)
        if not 60000 <= len(ids) <= 65536 or len(ids)+64 > 40960*factor:
            raise ValueError(f'unexpected 64K token length {len(ids)}')
        input_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        if path.exists():
            record = json.loads(path.read_text())
            if record['identity'] != identity or record['input_sha256'] != input_hash:
                raise ValueError('checkpoint mismatch')
        else:
            record = dict(identity=identity,index=idx,dataset=TASK,input_tokens=len(ids),
                input_sha256=input_hash,answers=row['outputs'],arms={},status='running')
        for arm in ARMS:
            if arm in record['arms']: continue
            atomic_json(out/'heartbeat.json',dict(pid=os.getpid(),time=time.time(),index=idx,arm=arm))
            mode = arm.split('_')[0]
            trial = generate(model,tokenizer,ids,64,GateConfig(mode=mode,epsilon=.05),
                             'cuda:0',stop_ids=stop_ids,stop_after_tokens=2)
            trial['score'] = score(trial['prediction'],row['outputs'])
            if arm == 'full' and args.verify_native_full and not native_verified:
                native = generate_native(model,ids,64,'cuda:0',stop_ids=stop_ids,stop_after_tokens=2)
                if native != trial['generated_ids']:
                    raise ValueError('native DynamicCache and node-cache gates-off differ')
                trial['native_full_parity'] = True; native_verified = True
            if arm != 'full' and trial['generated_ids'][0] != record['arms']['full']['generated_ids'][0]:
                raise ValueError('prefill first token differs')
            record['arms'][arm] = trial
            atomic_json(path,record)
            print(f'{TASK}:{idx} {arm} score={trial["score"]:.4f} tokens={trial["generated_tokens"]} seconds={trial["wall_seconds"]:.1f} gates={trial["gate_counts"]}',flush=True)
            torch.cuda.empty_cache()
        record['status']='complete'; atomic_json(path,record)
        atomic_json(out/'heartbeat.json',dict(pid=os.getpid(),time=time.time(),index=idx,status='complete'))
    atomic_json(out/'GENERATED.json',dict(identity=identity,completed=time.time(),samples=len(rows),status='pending independent validation'))


if __name__ == '__main__': main()
