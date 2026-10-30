"""Historical E155 paired task bootstrap; sample uncertainty, not run noise."""
import hashlib,json
from pathlib import Path
import numpy as np
from benchmark.LongBench.eval import sample_score,validate_runcap_sidecar
BASE=Path('/home/wangyuanshuo02/tmp_e155d_collect')
OUT=Path('/home/wangyuanshuo02/two-level-attention/agent_doc/results/nfa_ablation_20261030')
HOSTS={2048:'local',4096:'pc32',8192:'pc31'}
result=dict(schema='nfa-near-length-paired-v1',bootstrap=dict(replicates=10000,seed=20261030,unit='paired request within each task',interval='percentile95; sample uncertainty, NOT independent run noise'),tasks={},status='pending independent verification')
for task in ('hotpotqa','gov_report'):
    rows={};meta={};scores={}
    for nl,host in HOSTS.items():
        path=next((BASE/host/f'pred_nb32ntoffnl{nl}').glob(task+'*.jsonl'))
        raw=path.read_bytes(); rr=[json.loads(line) for line in raw.splitlines()]
        assert len(rr)==200 and len({r['_id'] for r in rr})==200
        m=json.load(Path(str(path)+'.tli_manifest.json').open())
        cap=json.load(Path(str(path)+'.runcap.json').open());validate_runcap_sidecar(str(path),False)
        assert m['tli_sink_tokens']==4 and m['tli_swa_size']==128 and m['tli_near_len']==nl
        assert m['tli_near_pages']==32 and m['tli_near_tokens']=='off'
        assert not m['tli_enable_layer_skip'] and not m['tli_dyn_dgate'] and not m['tli_sparse_prefill']
        assert m['tia_level1_topk']==128 and m['tia_level2_topk']==1024
        rows[nl]={r['_id']:r for r in rr}
        scores[nl]={r['_id']:sample_score(task,r['pred'],r['answers'],r.get('all_classes',[]))*100 for r in rr}
        token_lengths=dict(zip([r['_id'] for r in rr],cap['input_tokens']['per_sample']))
        meta[nl]=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),params=m,producer=cap['producer'],
            effective_cap=cap['effective_cap'],max_new_tokens=cap['max_new_tokens'],dataset_fingerprint=cap['dataset_fingerprint'],
            input_tokens_by_id=token_lengths,actual_budget_mean=float(np.mean([r['budget'] for r in rr if r['budget'] is not None])),
            budget_observed_requests=sum(r['budget'] is not None for r in rr),per_request_score=scores[nl])
    ids=sorted(rows[4096]);assert all(set(rows[n])==set(ids) for n in HOSTS)
    for nl in HOSTS:
        assert all(rows[nl][i]['answers']==rows[4096][i]['answers'] for i in ids)
    # Equality of input LENGTH is weaker than input-token hash; state that limit.
    checks=dict(unique_ids_and_answers_aligned=True,
        input_lengths_identical=all(meta[n]['input_tokens_by_id']==meta[4096]['input_tokens_by_id'] for n in HOSTS),
        producer_git_identical=len({meta[n]['producer']['git_commit'] for n in HOSTS})==1,
        producer_script_hash_identical=len({meta[n]['producer']['script_sha256'] for n in HOSTS})==1,
        other_params_identical=all({k:v for k,v in meta[n]['params'].items() if k!='tli_near_len'}=={k:v for k,v in meta[4096]['params'].items() if k!='tli_near_len'} for n in HOSTS),
        input_token_hashes_available=False)
    rng=np.random.default_rng(20261030);sample=rng.integers(0,len(ids),size=(10000,len(ids)))
    contrasts={}
    for left,right in [(2048,4096),(8192,4096),(8192,2048)]:
        d=np.array([scores[left][i]-scores[right][i] for i in ids]);boot=d[sample].mean(1)
        contrasts[f'{left}_minus_{right}']=dict(delta_points=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),n=200,changed_score_requests=int(np.count_nonzero(d)))
    result['tasks'][task]=dict(arms=meta,checks=checks,contrasts=contrasts)
(OUT/'near_length_paired.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({t:dict(checks=x['checks'],contrasts=x['contrasts']) for t,x in result['tasks'].items()},indent=2))
