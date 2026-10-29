"""Mechanical result validation, separate from experiment generation.

Recomputes input identities and official scores, then verifies the recorded gate,
cache topology, migration, KV bytes and token conservation for every arm/layer.
This is executable self-validation, not an independent human/agent review.
"""
import argparse
import hashlib
import json
from pathlib import Path
from string import Formatter

from transformers import AutoTokenizer
from benchmark.LongBench.eval import sample_score
from benchmark.LongBench.metrics import SCORER_BACKEND_ID


ROOT=Path(__file__).resolve().parents[2]
if not __debug__:raise RuntimeError('run validation with Python optimization disabled')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bounds(length):
    sink=min(4,length);end=max(sink,length-128)
    far=min(end,max(sink,((length-128-4096)//64)*64))
    return sink,far,end


def validate(outdir,allow_partial=False):
    outdir=Path(outdir)
    manifests=[json.loads(p.read_text()) for p in sorted(outdir.glob('manifest_rank*.json'))]
    if not manifests:raise ValueError('no manifests')
    protocol=manifests[0]['protocol'];identity=manifests[0]['identity']
    assert protocol['schema'] in ('frozen-near-far-screen-v2-near-staging','frozen-near-far-screen-v3-independent-gates')
    assert hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()==identity
    assert len(manifests)==protocol['world_size']
    assert all(m['protocol']==protocol and m['identity']==identity for m in manifests)
    for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest,name
    model=Path(protocol['model']);model_cfg=json.loads((model/'config.json').read_text())
    assert sha(model/'config.json')==protocol['model_config_sha256']
    assert sha(model/'tokenizer_config.json')==protocol['tokenizer_config_sha256']
    factor=2*2*model_cfg['num_key_value_heads']*model_cfg.get('head_dim',model_cfg['hidden_size']//model_cfg['num_attention_heads'])
    layers=model_cfg['num_hidden_layers']
    tokenizer=AutoTokenizer.from_pretrained(model)
    eos=json.loads((model/'generation_config.json').read_text())['eos_token_id']
    eos=eos if isinstance(eos,list) else [eos]
    prompts=json.loads((ROOT/'benchmark/LongBench/config/dataset2prompt.json').read_text())
    # Data root is the one used for this local experiment; hashes bind exact bytes.
    data_root=Path('/home/wangyuanshuo02/datasets/LongBench/data')
    datasets={}
    for ds,digest in protocol['dataset_sha256'].items():
        path=data_root/(ds+'.jsonl');assert sha(path)==digest
        datasets[ds]=[json.loads(line) for line in path.read_text().splitlines()]
        assert {field for _,field,_,_ in Formatter().parse(prompts[ds]) if field} <= {'context','input'}
    expected={(ds,i) for ds,ids in protocol['selected_rows'].items() for i in ids}
    arms=['full']+[f'{mode}_{eps:g}' for eps in protocol['thresholds'] for mode in ['far','dual']]
    records=[];partial=0;seen=set()
    for path in sorted(outdir.glob('requests/*/*.json')):
        record=json.loads(path.read_text());assert record['identity']==identity
        key=(record['dataset'],record['row_index']);assert key in expected
        assert key not in seen,'duplicate request identity'
        seen.add(key)
        if record['status']!='complete':partial+=1;continue
        records.append(record)
    if not allow_partial:assert len(records)==len(expected) and seen==expected and not partial
    near_layers=near_migrations=0;native_checks=0;checked_arms=0;max_profile_difference=0.
    per_task={ds:0 for ds in protocol['datasets']}
    for record in records:
        ds,index=record['dataset'],record['row_index'];row=datasets[ds][index]
        assert record['answers']==row['answers']
        text=prompts[ds].format(**row);ids=tokenizer(text,add_special_tokens=False).input_ids
        rawlen=len(ids)
        if rawlen>protocol['context_cap']:
            n=protocol['context_cap']//2
            text=tokenizer.decode(ids[:n],skip_special_tokens=True)+tokenizer.decode(ids[-(protocol['context_cap']-n):],skip_special_tokens=True)
        if ds!='repobench-p':text=tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tokenizer(text,add_special_tokens=False).input_ids
        assert len(ids)==record['input']['input_tokens']
        assert rawlen==record['input']['raw_tokens']
        assert hashlib.sha256(json.dumps(ids).encode()).hexdigest()==record['input']['input_sha256']
        assert set(record['arms'])==set(arms)
        per_task[ds]+=1
        full=record['arms']['full']
        for arm,trial in record['arms'].items():
            checked_arms+=1
            generated=trial['generated_ids'];limit=protocol['max_new_tokens'][ds]
            assert 0<len(generated)<=limit and len(generated)==trial['generated_tokens']
            assert not any(t in eos for t in generated[:-1])
            assert generated[-1] in eos or len(generated)==limit
            assert generated[0]==full['generated_ids'][0]
            assert tokenizer.decode(generated,skip_special_tokens=True)==trial['prediction']
            score=sample_score('repobench' if ds=='repobench-p' else ds,trial['prediction'],row['answers'],row.get('all_classes',[]))
            assert abs(score-trial['score'])<1e-12
            if trial.get('native_full_parity'):native_checks+=1
            if protocol.get('attention','sdpa').startswith('tli') and len(generated)>1:
                assert protocol['tli_environment']['TLI_OPERATOR_MODE']=='off'
                receipts=trial['tli_receipts'];assert len(receipts)==layers
                for li,receipt in receipts.items():
                    assert receipt['scorer']=='TLIIndexer.compute_score'
                    assert receipt['selector']=='TLIIndexer.compute_mask'
                    assert receipt['dimensions']==list(range(48,64))+list(range(112,128))
                    assert max(receipt['selected_per_kv_head'])<=1024
                    assert min(receipt['selected_per_kv_head'])>0
                    state=trial['final_kv']['layers'][int(li)]
                    assert receipt['logical_tokens']==state['logical_length']
                    assert receipt['visible_tokens']==state['attention_tokens']
                    for name in ['skip_near','skip_far']:assert receipt[name]==state[name]
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
                if protocol['schema']=='frozen-near-far-screen-v2-near-staging' and sn and sf and n+f>eps:sn,sf=(n<=f,f<n)
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
    report=dict(status='partial-validated' if len(records)<len(expected) else 'complete-validated',identity=identity,
        complete_requests=len(records),expected_requests=len(expected),checked_arms=checked_arms,per_task=per_task,
        checked_request_layers=checked_arms*layers,near_only_staging_layers=near_layers,
        near_only_layers_with_far_growth=near_migrations,native_full_generation_checks=native_checks,
        max_prefill_profile_difference=max_profile_difference,scorer_backend=SCORER_BACKEND_ID,
        scope='Executable local self-validation, not independent reviewer admission. Verifies logged invariants and official score recomputation; actual values/migration also covered by native-model regression tests.')
    (outdir/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('outdir');ap.add_argument('--allow-partial',action='store_true');args=ap.parse_args()
    print(json.dumps(validate(args.outdir,args.allow_partial),indent=2))
