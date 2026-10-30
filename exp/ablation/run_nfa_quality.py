"""Restartable NFA A/B/C/E/F quality runner using the existing frozen evaluator.

Full-prompt prefill protocol, explicitly NOT Quest prefix/suffix replay.
Timing from this eager reference is diagnostic and not deployment throughput.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from benchmark.LongBench.prefill_frozen_skip import (
    prepare_ids,generate,atomic_json,sha_file,sample_score)
from sparse_attn.ops.frozen_region_gate import GateConfig
from sparse_attn.ops.frozen_region_cache import RegionGeometry
from sparse_attn.ops.frozen_region_tli import tli_args
from sparse_attn.patches import register_patch
from sparse_attn.info import resolve_treatment_snapshot
from exp.ablation.nfa_matrix import matrix

ROOT=Path(__file__).resolve().parents[2]


def treatment(spec):
    args=tli_args()
    args.tli_subspace=spec['subspace']
    args.tia_level2_cmp_ratio=128//spec['dimensions']
    if spec['subspace']=='full': args.tli_l2_subspace='full'
    args.tli_near_len=spec['near'];args.tli_near_pages=spec['near_pages']
    args.tli_sink_tokens=spec['sink'];args.tli_swa_size=spec['swa']
    args.tia_level1_topk=spec['k1'];args.tia_level2_topk=spec['k2']
    args.tli_far_method=spec['far_score'];args.tli_near_method=spec['near_score']
    return args


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--model',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--cell',required=True)
    ap.add_argument('--dataset',choices=['hotpotqa','gov_report'],required=True)
    ap.add_argument('--start',type=int,default=0);ap.add_argument('--end',type=int,default=200)
    ap.add_argument('--context-cap',type=int,default=31500)
    ap.add_argument('--selection-protocol',choices=['production','strict'],default='production')
    ap.add_argument('--smoke-output-limit',type=int,default=0)
    args=ap.parse_args()
    for key,value in dict(TLI_OPERATOR_MODE='off',TLI_FINE_GATHER='0',TLI_SPARSE_PREFILL='0',TLI_DECODE_DENSE='0').items():os.environ[key]=value
    torch.set_num_threads(8)
    cells={c['id']:c for c in matrix()};cell=cells[args.cell];spec=cell['spec']
    dataset=Path(args.data)/(args.dataset+'.jsonl')
    rows=[json.loads(line) for line in dataset.read_text().splitlines()]
    if not 0<=args.start<args.end<=len(rows):raise ValueError('invalid row range')
    prompt_path=ROOT/'benchmark/LongBench/config/dataset2prompt.json'
    maxlen_path=ROOT/'benchmark/LongBench/config/dataset2maxlen.json'
    prompts=json.loads(prompt_path.read_text());lengths=json.loads(maxlen_path.read_text())
    max_new=args.smoke_output_limit or lengths[args.dataset]
    sources={str(p.relative_to(ROOT)):sha_file(p) for folder in ('sparse_attn','benchmark/LongBench','exp/ablation') for p in sorted((ROOT/folder).rglob('*.py'))}
    protocol=dict(schema='nfa-ablation-quality-v1',cell=cell,dataset=args.dataset,selection_protocol=args.selection_protocol,
        dataset_sha256=sha_file(dataset),model_config=sha_file(Path(args.model)/'config.json'),
        tokenizer_config=sha_file(Path(args.model)/'tokenizer_config.json'),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources=sources,
        torch=torch.__version__,transformers=transformers.__version__,dtype='bfloat16',
        input_protocol='full-prompt dense prefill then freeze then ordinary greedy decode',
        truncation='tokenize raw formatted prompt; head/tail to 31500 then decode and chat-template retokenize',
        context_cap=args.context_cap,prompt_sha256=sha_file(prompt_path),maxlen_sha256=sha_file(maxlen_path),
        max_new_tokens=max_new,smoke=bool(args.smoke_output_limit),
        gate='last256 query-relative regional means; independent threshold; once per layer/request',
        generation='greedy, stop EOS, no thinking',candidate_mass='full Q/K probabilities incl protected union; sampled decode steps 1,2,4,...512',
        timing_scope='eager quality reference wall only; not efficiency results')
    identity=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    out=Path(args.out)/args.cell/args.dataset;out.mkdir(parents=True,exist_ok=True)
    lock=(out/f'rows_{args.start}_{args.end}.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=out/f'manifest_{args.start}_{args.end}.json'
    if manifest.exists() and json.loads(manifest.read_text())['identity']!=identity:raise ValueError('checkpoint identity changed')
    atomic_json(manifest,dict(identity=identity,protocol=protocol,start=args.start,end=args.end))
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval()
    if model.config.model_type!='qwen3':raise ValueError('this ablation round is Qwen3 only')
    sparse=treatment(spec)
    assert register_patch(model,sparse,resolve_treatment_snapshot(sparse))==model.config.num_hidden_layers
    if args.selection_protocol == 'strict':
        for layer in model.model.layers:
            layer.self_attn.indexer.ablation_spec=dict(spec,measure_mass='B_selection' in cell['groups'])
    model._frozen_tli=True
    geometry=RegionGeometry(sink=spec['sink'],window=spec['swa'],near=spec['near'],max_new_tokens=max_new)
    gate=GateConfig(epsilon=spec['epsilon'],mode=spec['mode'])
    for i in range(args.start,args.end):
        target=out/'requests'/f'{i:04d}.json'
        row=rows[i];ids,info=prepare_ids(tokenizer,row,args.dataset,prompts[args.dataset],args.context_cap)
        if len(ids)+max_new>32768:raise ValueError('actual chat input plus output exceeds 32Ki token budget')
        if target.exists():
            old=json.loads(target.read_text())
            if old['identity']!=identity or old['input']!=info:raise ValueError('stale result')
            continue
        atomic_json(out/f'heartbeat_{args.start}_{args.end}.json',dict(row=i,pid=os.getpid(),time=time.time()))
        result=generate(model,tokenizer,ids,max_new,gate,'cuda:0',geometry=geometry,reclaim=spec['reclaim'])
        result['score']=sample_score(args.dataset,result['prediction'],row['answers'],row.get('all_classes',[]))
        atomic_json(target,dict(identity=identity,dataset=args.dataset,row_index=i,input=info,answers=row['answers'],result=result,status='complete'))
        print(json.dumps(dict(cell=args.cell,dataset=args.dataset,row=i,score=result['score'],tokens=result['generated_tokens'],gates=result['gate_counts'])),flush=True)
        torch.cuda.empty_cache()
    atomic_json(out/f'done_{args.start}_{args.end}.json',dict(identity=identity,time=time.time()))

if __name__=='__main__':main()
