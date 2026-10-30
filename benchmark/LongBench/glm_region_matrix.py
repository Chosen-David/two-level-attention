"""Eight-worker resumable paired GLM quality matrix; one file lock per input."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

os.environ['TLI_SCORER_BACKEND']='difflib'

import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from benchmark.LongBench.prefill_frozen_skip import (
    prepare_ids,generate,generate_native,GateConfig,atomic_json,sha_file,SCREEN_TASKS,ROOT)
from benchmark.LongBench.eval import sample_score
from benchmark.RULER.score_ruler import string_match_all
from sparse_attn.ops.frozen_region_tli import tli_args
from sparse_attn.info import resolve_treatment_snapshot
from sparse_attn.patches import register_patch

ARMS=('dense','full','far','dual')

class CheckpointMismatch(ValueError):
    pass


def build_jobs(data,ruler):
    jobs=[]
    rows=[json.loads(x) for x in Path(ruler).read_text().splitlines()]
    for i,row in enumerate(rows):jobs.append(dict(task='niah_multikey_2',row=i,data=row,kind='ruler'))
    # Interleave tasks so all GPUs share long generation and long input work.
    datasets={d:[json.loads(x) for x in (Path(data)/(d+'.jsonl')).read_text().splitlines()] for d in SCREEN_TASKS}
    for i in range(max(map(len,datasets.values()))):
        for task,rows in datasets.items():
            if i<len(rows):jobs.append(dict(task=task,row=i,data=rows[i],kind='longbench'))
    return jobs


def main():
    ap=argparse.ArgumentParser()
    for key in ('model','data','ruler','out'):ap.add_argument('--'+key,required=True)
    ap.add_argument('--rank',type=int,required=True)
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--limit',type=int,default=0,help='smoke only; distinct run directory')
    ap.add_argument('--selector',choices=('tail32','full256-reference'),default='tail32')
    args=ap.parse_args()
    if not 0<=args.rank<args.workers:raise ValueError('rank outside worker range')
    for k,v in dict(TLI_OPERATOR_MODE='off',TLI_FINE_GATHER='0',TLI_SPARSE_PREFILL='0',
                    TLI_DECODE_DENSE='0',TLI_CUDA_GRAPH='0',TLI_SCORER_BACKEND='difflib',
                    TLI_BATCH_INDEXER='legacy').items():os.environ[k]=v
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    worker_lock=(out/f'worker{args.rank}.lock').open('a')
    fcntl.flock(worker_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=build_jobs(args.data,args.ruler)
    if args.limit:jobs=jobs[:args.limit]
    source={str(p.relative_to(ROOT)):sha_file(p) for base in ('sparse_attn','benchmark/LongBench','benchmark/RULER') for p in (ROOT/base).rglob('*.py')}
    protocol=dict(schema='glm-regions-matrix-v2',selector=args.selector,
        selector_scope='diagnostic reference; not paper-method repair' if args.selector=='full256-reference' else 'original tail32',
        selector_backend='torch' if args.selector=='full256-reference' else 'existing',arms=ARMS,seed=42,threshold=.05,rows=256,
        method='tli',k1=128,k2=1024,near_pages=32,near_len=4096,sink=4,swa=128,
        scorer_backend='difflib',ruler_eos_policy='ignore early EOS until min_new_tokens; no logits suppression',
        graph=False,operator='off',dtype='bfloat16',longbench_context_cap=31500,
        ruler_prompt='raw input, tokenizer default special tokens; no truncation',
        ruler_max_new=64,ruler_min_new=2,longbench_min_new=1,
        model=str(Path(args.model).resolve()),model_config=sha_file(Path(args.model)/'config.json'),
        tokenizer_config=sha_file(Path(args.model)/'tokenizer_config.json'),
        data={d:sha_file(Path(args.data)/(d+'.jsonl')) for d in SCREEN_TASKS},
        ruler_data=sha_file(args.ruler),source=source,
        prompt_config=sha_file(ROOT/'benchmark/LongBench/config/dataset2prompt.json'),
        generation_config=sha_file(ROOT/'benchmark/LongBench/config/dataset2maxlen.json'),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        torch=torch.__version__,transformers=transformers.__version__,workers=args.workers,
        expected={t:sum(j['task']==t for j in jobs) for t in sorted({j['task'] for j in jobs})})
    identity=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    with (out/'protocol.lock').open('a') as protocol_lock:
        fcntl.flock(protocol_lock,fcntl.LOCK_EX)
        shared=out/'protocol.json'
        if shared.exists() and json.loads(shared.read_text())['identity']!=identity:
            raise CheckpointMismatch('shared run protocol mismatch')
        atomic_json(shared,dict(identity=identity,protocol=protocol))
    manifest=out/f'manifest_rank{args.rank}.json'
    if manifest.exists() and json.loads(manifest.read_text())['identity']!=identity:raise ValueError('run identity mismatch')
    atomic_json(manifest,dict(identity=identity,protocol=protocol,pid=os.getpid(),host=socket.gethostname()))
    torch.manual_seed(42);torch.set_num_threads(6)
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,
        device_map='cuda:0',attn_implementation='sdpa').eval()
    if model.config.model_type!='glm4_moe_lite':raise ValueError('GLM only')
    native=[x.self_attn.forward for x in model.model.layers]
    treatment=tli_args(args.selector)
    assert register_patch(model,treatment,resolve_treatment_snapshot(treatment))==47
    patched=[x.self_attn.forward for x in model.model.layers]
    prompts=json.loads((ROOT/'benchmark/LongBench/config/dataset2prompt.json').read_text())
    lengths=json.loads((ROOT/'benchmark/LongBench/config/dataset2maxlen.json').read_text())
    # Primary deterministic partition; then work-steal unlocked unfinished inputs.
    primary=[i for i in range(len(jobs)) if i%args.workers==args.rank]
    order=primary+[i for i in range(len(jobs)) if i%args.workers!=args.rank]
    failed_seen=[]
    for i in order:
        job=jobs[i];task=job['task'];row=job['row'];data=job['data']
        dest=out/'requests'/task/f'{row:04d}.json';dest.parent.mkdir(parents=True,exist_ok=True)
        lock=dest.with_suffix('.lock').open('a')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:lock.close();continue
        try:
            record=json.loads(dest.read_text()) if dest.exists() else None
            if record and record['identity']!=identity:raise CheckpointMismatch('request identity mismatch')
            if record and record['status']=='failed':
                failed_seen.append(dict(task=task,row=row));continue
            if job['kind']=='longbench':
                ids,meta=prepare_ids(tokenizer,data,task,prompts[task],31500)
                max_new=lengths[task];min_new=1;answers=data['answers']
            else:
                ids=tokenizer(data['input'],add_special_tokens=True).input_ids
                meta=dict(input_tokens=len(ids),truncated=False,input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
                max_new=64;min_new=2;answers=data['outputs']
            if len(ids)+max_new>model.config.max_position_embeddings:raise ValueError('context exceeds native model limit')
            if record and (record['input']['input_sha256']!=meta['input_sha256'] or record['answers']!=answers):
                raise CheckpointMismatch('request input/answers differ from checkpoint')
            if record and record['status']=='complete':
                if set(record['arms'])!=set(ARMS):raise CheckpointMismatch('incomplete completed record')
                continue
            record=record or dict(identity=identity,task=task,row=row,source_index=data.get('index',row),
                input=meta,answers=answers,arms={},status='running')
            atomic_json(dest,record)
            for arm in ARMS:
                if arm in record['arms']:continue
                atomic_json(out/f'heartbeat_rank{args.rank}.json',dict(pid=os.getpid(),time=time.time(),task=task,row=row,arm=arm,status='running'))
                for layer,forward in zip(model.model.layers,native if arm=='dense' else patched):layer.self_attn.forward=forward
                model._frozen_tli=arm!='dense'
                # Release the previous sparse request's index scratch before dense.
                for layer in model.model.layers:
                    layer.self_attn.indexer.clear()
                torch.manual_seed(42);gc.collect();torch.cuda.empty_cache()
                if arm=='dense':
                    tokens=generate_native(model,ids,max_new,'cuda:0',min_new_tokens=min_new)
                    result=dict(generated_ids=tokens,generated_tokens=len(tokens),prediction=tokenizer.decode(tokens,skip_special_tokens=True),route='native_dense_sdpa')
                else:
                    result=generate(model,tokenizer,ids,max_new,GateConfig(mode=arm,epsilon=.05),'cuda:0',min_new_tokens=min_new)
                    result['route']='glm_frozen_region_tli'
                    if result['generated_ids'][0]!=record['arms']['dense']['generated_ids'][0]:raise ValueError('dense prefill prediction changed')
                result['score_pct']=(string_match_all([result['prediction']],[answers]) if job['kind']=='ruler' else
                    100*sample_score('repobench' if task=='repobench-p' else task,result['prediction'],answers,data.get('all_classes',[])))
                record['arms'][arm]=result;atomic_json(dest,record)
                print(json.dumps(dict(rank=args.rank,task=task,row=row,arm=arm,score_pct=result['score_pct'])),flush=True)
            record['status']='complete';atomic_json(dest,record)
        except CheckpointMismatch:
            raise
        except Exception as exc:
            failure=record or dict(identity=identity,task=task,row=row,arms={})
            failure.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
            atomic_json(dest,failure)
            print(traceback.format_exc(),flush=True)
            # Failures require review; never count as zero or retry infinitely.
            raise
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    atomic_json(out/f'worker_done_rank{args.rank}.json',dict(identity=identity,pid=os.getpid(),time=time.time(),status='scan_finished_not_global_acceptance',failed_seen=failed_seen))

if __name__=='__main__':main()
