"""Mechanical RULER frozen-cache validation and paired statistics; not independent admission."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from benchmark.LongBench.validate_frozen_skip import bounds
from benchmark.LongBench.prefill_frozen_skip import ROOT, atomic_json, sha_file

if not __debug__: raise RuntimeError('assertions required')


def validate(outdir, allow_partial=False):
    outdir=Path(outdir)
    manifest=json.loads((outdir/'manifest.json').read_text())
    protocol=manifest['protocol'];identity=manifest['identity']
    assert protocol['schema']=='frozen-ruler-dual-vs-far-v1'
    assert hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()==identity
    for name,digest in protocol['source_sha256'].items(): assert sha_file(ROOT/name)==digest,name
    model=Path(protocol['model']);model_cfg=json.loads((model/'config.json').read_text())
    for name in ['model_config','tokenizer_config','generation_config']:
        filename='config.json' if name=='model_config' else name+'.json'
        assert sha_file(model/filename)==protocol[name+'_sha256']
    assert protocol['rope_scaling']==dict(rope_type='yarn',type='yarn',factor=2.,original_max_position_embeddings=40960,beta_fast=32,beta_slow=1)
    assert protocol['epsilon']==.05 and protocol['statistic']=='mean' and protocol['calibration_rows']==256
    assert protocol['environment']['TLI_OPERATOR_MODE']=='off'
    factor=4*model_cfg['num_key_value_heads']*model_cfg.get('head_dim',model_cfg['hidden_size']//model_cfg['num_attention_heads'])
    layers=model_cfg['num_hidden_layers']
    tokenizer=AutoTokenizer.from_pretrained(model)
    eos=list(dict.fromkeys([tokenizer.eos_token_id,tokenizer.encode('\n',add_special_tokens=False)[-1]]))
    assert protocol['stop_ids']==eos
    data=Path(protocol['data']);assert sha_file(data)==protocol['data_sha256']
    source_rows=[json.loads(line) for line in data.read_text().splitlines()]
    rows={r['index']:r for r in source_rows};assert len(rows)==len(source_rows)==100
    expected=set(protocol['selected_indices']);assert len(expected)==protocol['samples']
    assert protocol['selected_indices']==[r['index'] for r in source_rows[:protocol['samples']]]
    records=[];seen=set()
    for path in sorted((outdir/'requests').glob('*.json')):
        record=json.loads(path.read_text());idx=record['index']
        assert idx in expected and idx not in seen and record['identity']==identity
        seen.add(idx)
        if record['status']=='complete': records.append(record)
        else: assert record['status']=='running' and allow_partial
    if not allow_partial: assert len(records)==len(expected) and seen==expected
    near_layers=near_migrations=native_checks=checked_arms=0;max_profile_difference=0.
    for record in records:
        row=rows[record['index']];assert record['answers']==row['outputs']
        ids=tokenizer(row['input'],truncation=False).input_ids
        assert 60000<=len(ids)<=65536 and len(ids)==record['input_tokens']
        assert hashlib.sha256(json.dumps(ids).encode()).hexdigest()==record['input_sha256']
        assert set(record['arms'])=={'full','far_0.05','dual_0.05'}
        full=record['arms']['full']
        assert all(record['arms']['far_0.05']['decisions'][li]['skip_far']==d['skip_far'] for li,d in record['arms']['dual_0.05']['decisions'].items())
        for arm,trial in record['arms'].items():
            checked_arms+=1;generated=trial['generated_ids']
            assert 2<=len(generated)<=64 and len(generated)==trial['generated_tokens']
            assert not any(t in eos for t in generated[1:-1])
            assert generated[-1] in eos or len(generated)==64
            assert generated[0]==full['generated_ids'][0]
            assert tokenizer.decode(generated,skip_special_tokens=True)==trial['prediction']
            answers=row['outputs'];assert len(answers)>0
            reference_score=sum(answer.lower() in trial['prediction'].lower() for answer in answers)/len(answers)
            assert math.isfinite(trial['score']) and abs(reference_score-trial['score'])<1e-12
            native_checks+=bool(trial.get('native_full_parity'))
            assert len(trial['tli_receipts'])==layers
            for li,receipt in trial['tli_receipts'].items():
                assert receipt['scorer']=='TLIIndexer.compute_score' and receipt['selector']=='TLIIndexer.compute_mask'
                assert receipt['dimensions']==list(range(48,64))+list(range(112,128))
                assert 0<min(receipt['selected_per_kv_head'])<=max(receipt['selected_per_kv_head'])<=1024
                final=trial['final_kv']['layers'][int(li)]
                assert receipt['logical_tokens']==final['logical_length'] and receipt['visible_tokens']==final['attention_tokens']
                for name in ['skip_near','skip_far']: assert receipt[name]==final[name]
            for profile in trial['profiles'].values():
                for region in ['near','far']:
                    assert all(math.isfinite(v) and 0<=v<=1.000001 for v in profile[region].values())
            assert len(trial['profiles'])==len(trial['decisions'])==layers
            counted={k:0 for k in ['keep','near','far','both']}
            mode,eps=('full',0.) if arm=='full' else (arm.split('_')[0],float(arm.split('_')[1]))
            for li in range(layers):
                profile=trial['profiles'][str(li)];decision=trial['decisions'][str(li)]
                before=trial['prefill_full_kv']['layers'][li]
                frozen=trial['frozen_kv']['layers'][li];final=trial['final_kv']['layers'][li]
                n=profile['near'][protocol['statistic']];f=profile['far'][protocol['statistic']]
                sn=mode=='dual' and n<eps and before['nodes']['near']['tokens']>0
                sf=mode in ['far','dual'] and f<eps and before['nodes']['far']['tokens']>0
                assert decision==dict(skip_near=sn,skip_far=sf)
                counted['both' if sn and sf else 'near' if sn else 'far' if sf else 'keep']+=1
                names=['sink']+([] if sf else ['far'])+([] if sn and sf else ['near_staging' if sn else 'near'])+['swa']
                for state in [frozen,final]:
                    assert state['skip_near']==sn and state['skip_far']==sf
                    assert set(state['nodes'])==set(names) and state['linked_order']==names
                    assert state['attention_order']==[name for name in names if name!='near_staging']
                    assert state['stored_tokens']==sum(nd['tokens'] for nd in state['nodes'].values())
                    assert state['attention_tokens']==sum(nd['tokens'] for name,nd in state['nodes'].items() if name!='near_staging')
                    assert state['stored_tokens']+state['permanently_dropped']==state['logical_length']
                    for name,node in state['nodes'].items():
                        assert node['kv_bytes']==node['capacity']*factor
                        assert node['attention_visible']==(name!='near_staging')
                        assert 0<=node['tokens']<=node['capacity']
                assert frozen['logical_length']==len(ids)
                assert final['logical_length']==len(ids)+len(generated)-1
                sink,far,end=bounds(final['logical_length'])
                assert final['nodes']['sink']['tokens']==sink
                assert final['nodes']['swa']['tokens']==final['logical_length']-end
                if not sf:assert final['nodes']['far']['tokens']==far-sink
                if not (sn and sf):assert final['nodes']['near_staging' if sn else 'near']['tokens']==end-far
                if sn and not sf:
                    near_layers+=1
                    assert frozen['permanently_dropped']==final['permanently_dropped']==0
                    assert sum(nd['kv_bytes'] for nd in frozen['nodes'].values())==sum(nd['kv_bytes'] for nd in before['nodes'].values())
                    near_migrations+=final['nodes']['far']['tokens']>frozen['nodes']['far']['tokens']
                for region in ['near','far']:
                    for stat in ['mean','p95','max']:
                        diff=abs(profile[region][stat]-full['profiles'][str(li)][region][stat])
                        max_profile_difference=max(diff,max_profile_difference)
                        assert diff<1e-6
            assert counted==trial['gate_counts']
            for phase in ['prefill_full_kv','frozen_kv','final_kv']:
                assert trial[phase]['kv_bytes']==sum(n['kv_bytes'] for layer in trial[phase]['layers'] for n in layer['nodes'].values())
    report=dict(status='complete-validated' if len(records)==len(expected) else 'partial-validated',
        identity=identity,complete_requests=len(records),expected_requests=len(expected),checked_arms=checked_arms,
        checked_request_layers=checked_arms*layers,native_full_generation_checks=native_checks,
        near_only_staging_layers=near_layers,near_only_layers_with_far_growth=near_migrations,
        max_prefill_profile_difference=max_profile_difference,scope='Mechanical self-validation; independent review required')
    atomic_json(outdir/'validation.json',report)
    return report,records


def summarize(outdir,records):
    outdir=Path(outdir);rng=np.random.default_rng(42)
    summary=dict(status='pending independent validation',n=len(records),arms={},comparisons={})
    for arm in ('full','far_0.05','dual_0.05'):
        summary['arms'][arm]=dict(score_0_100=float(np.mean([r['arms'][arm]['score'] for r in records])*100),
            gate_counts={state:sum(r['arms'][arm]['gate_counts'][state] for r in records) for state in ('keep','far','near','both')},
            near_active_requests=sum(any(d['skip_near'] for d in r['arms'][arm]['decisions'].values()) for r in records),
            far_active_requests=sum(any(d['skip_far'] for d in r['arms'][arm]['decisions'].values()) for r in records))
    for arm,base in [('dual_0.05','far_0.05'),('far_0.05','full'),('dual_0.05','full')]:
        delta=np.array([100*(r['arms'][arm]['score']-r['arms'][base]['score']) for r in records])
        bootstrap=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
        summary['comparisons'][arm+'-vs-'+base]=dict(delta_pp=float(delta.mean()),
            ci95_pp=np.quantile(bootstrap,[.025,.975]).tolist(),degraded=int((delta<0).sum()),
            improved=int((delta>0).sum()),same_score=int((delta==0).sum()),
            same_generated_ids=sum(r['arms'][arm]['generated_ids']==r['arms'][base]['generated_ids'] for r in records))
    summary['scope']='10000 paired request bootstrap, seed42, no equivalence claim; gates-off is TLI, not dense FullKV; no latency claim'
    atomic_json(outdir/'summary.json',summary)
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('outdir');ap.add_argument('--allow-partial',action='store_true');args=ap.parse_args()
    report,records=validate(args.outdir,args.allow_partial)
    if records:summarize(args.outdir,records)
    print(json.dumps(report,indent=2))
