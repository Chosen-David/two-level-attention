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
from benchmark.LongBench.prefill_frozen_skip import (
    dataset_stopping_policy, prepare_quest_input,
)


ROOT=Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT=Path('/home/wangyuanshuo02/datasets/LongBench/data')
QUEST_SCHEMA='frozen-near-far-v1-quest-prefix-v1'
if not __debug__:raise RuntimeError('run validation with Python optimization disabled')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tail_dimensions(model_cfg):
    head_dim=model_cfg.get('head_dim',model_cfg['hidden_size']//model_cfg['num_attention_heads'])
    nope=model_cfg.get('qk_nope_head_dim',head_dim//2)
    rope=model_cfg.get('qk_rope_head_dim',head_dim-nope)
    return list(range(nope-16,nope))+list(range(nope+rope-16,nope+rope))


def validate_attention_work(trial,quest,generated,layers,num_kv_heads,tli_enabled,dimensions):
    work=trial['attention_work']
    assert work['schema']=='frozen-region-attention-work-v1'
    assert 'K1' in work['budgets'] and 'K2' in work['budgets']
    expected_ids=list(quest.suffix_ids)+list(generated[:-1])
    steps=work['steps'];assert len(steps)==len(expected_ids)
    for i,(step,token_id) in enumerate(zip(steps,expected_ids)):
        suffix=i<len(quest.suffix_ids)
        position=len(quest.prefix_ids)+i
        assert step['phase']==('suffix' if suffix else 'answer')
        assert step['logical_position']==position and step['input_token_id']==token_id
        assert len(step['layers'])==layers
        for li,entry in enumerate(step['layers']):
            assert entry['logical_tokens']==position+1
            assert entry['skip_near']==trial['decisions'][str(li)]['skip_near']
            assert entry['skip_far']==trial['decisions'][str(li)]['skip_far']
            counts=entry['attended_per_kv_head']
            assert len(counts)==num_kv_heads
            assert all(type(count) is int and 0<count<=entry['visible_tokens'] for count in counts)
            assert entry['visible_tokens']<=entry['logical_tokens']
            if tli_enabled:
                assert entry['scorer']=='TLIIndexer.compute_score'
                assert entry['selector']=='TLIIndexer.compute_mask'
                assert entry['dimensions']==dimensions
                assert counts==entry['selected_per_kv_head']
                assert max(counts)<=1024
                if i==len(steps)-1:
                    final=trial['tli_receipts'][str(li)]
                    assert all(final[key]==value for key,value in entry.items() if key in final)
            else:
                assert entry['selector']=='dense-visible-cache'
                assert counts==[entry['visible_tokens']]*num_kv_heads
    return len(steps),len(steps)*layers


def bounds(length):
    sink=min(4,length);end=max(sink,length-128)
    far=min(end,max(sink,((length-128-4096)//64)*64))
    return sink,far,end


def validate(outdir,allow_partial=False,data_root=DEFAULT_DATA_ROOT):
    outdir=Path(outdir)
    manifests=[json.loads(p.read_text()) for p in sorted(outdir.glob('manifest_rank*.json'))]
    if not manifests:raise ValueError('no manifests')
    protocol=manifests[0]['protocol'];identity=manifests[0]['identity']
    quest_mode=protocol['schema']==QUEST_SCHEMA
    assert protocol['schema'] in (QUEST_SCHEMA,'frozen-near-far-screen-v2-near-staging','frozen-near-far-screen-v3-independent-gates')
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
    # 数据根可显式覆盖；文件哈希仍绑定本次实验的精确字节。
    data_root=Path(data_root)
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
    checked_attention_steps=checked_attention_layer_steps=0
    per_task={ds:0 for ds in protocol['datasets']}
    for record in records:
        ds,index=record['dataset'],record['row_index'];row=datasets[ds][index]
        assert record['answers']==row['answers']
        if quest_mode:
            quest,input_info=prepare_quest_input(
                tokenizer,row,ds,prompts[ds],protocol['context_cap'])
            assert input_info==record['input']
            ids=list(quest.full_ids)
        else:
            quest=None
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
        arm_eos,min_new_tokens=(dataset_stopping_policy(eos,tokenizer,ds)
                                 if quest_mode else (eos,0))
        for arm,trial in record['arms'].items():
            checked_arms+=1
            generated=trial['generated_ids'];limit=protocol['max_new_tokens'][ds]
            assert 0<len(generated)<=limit and len(generated)==trial['generated_tokens']
            assert not any(t in arm_eos for t in generated[:-1])
            assert generated[-1] in arm_eos or len(generated)==limit
            assert not any(t in arm_eos for t in generated[:min_new_tokens])
            if not quest_mode:assert generated[0]==full['generated_ids'][0]
            if quest_mode:
                protocol_trace=trial['quest_protocol']
                assert protocol_trace['freeze_logical_position']==len(quest.prefix_ids)
                assert protocol_trace['suffix_logical_range']==[len(quest.prefix_ids),len(ids)]
                assert protocol_trace['suffix_replay_tokens']==len(quest.suffix_ids)
                assert protocol_trace['decode_capacity']==len(quest.suffix_ids)+limit
                steps,layer_steps=validate_attention_work(
                    trial,quest,generated,layers,model_cfg['num_key_value_heads'],
                    protocol.get('attention','').startswith('tli'),tail_dimensions(model_cfg))
                checked_attention_steps+=steps;checked_attention_layer_steps+=layer_steps
            assert tokenizer.decode(generated,skip_special_tokens=True)==trial['prediction']
            score=sample_score('repobench' if ds=='repobench-p' else ds,trial['prediction'],row['answers'],row.get('all_classes',[]))
            assert abs(score-trial['score'])<1e-12
            if trial.get('native_full_parity'):native_checks+=1
            decode_calls=max(0,len(generated)-1)+(len(quest.suffix_ids) if quest_mode else 0)
            if protocol.get('attention','sdpa').startswith('tli') and decode_calls>0:
                assert protocol['tli_environment']['TLI_OPERATOR_MODE']=='off'
                receipts=trial['tli_receipts'];assert len(receipts)==layers
                for li,receipt in receipts.items():
                    assert receipt['scorer']=='TLIIndexer.compute_score'
                    assert receipt['selector']=='TLIIndexer.compute_mask'
                    expected_dimensions=(tail_dimensions(model_cfg) if quest_mode
                                         else list(range(48,64))+list(range(112,128)))
                    assert receipt['dimensions']==expected_dimensions
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
                assert before['logical_length']==frozen['logical_length']==(len(quest.prefix_ids) if quest_mode else len(ids))
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
        checked_attention_steps=checked_attention_steps,
        checked_attention_layer_steps=checked_attention_layer_steps,
        max_prefill_profile_difference=max_profile_difference,scorer_backend=SCORER_BACKEND_ID,
        scope='Executable local self-validation, not independent reviewer admission. Verifies logged invariants and official score recomputation; Quest runs additionally verify canonical prefix/suffix identity and actual per-step attended-K receipts.')
    (outdir/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('outdir');ap.add_argument('--allow-partial',action='store_true')
    ap.add_argument('--data',default=str(DEFAULT_DATA_ROOT),help='LongBench JSONL data root; hashes must match the manifest')
    args=ap.parse_args()
    print(json.dumps(validate(args.outdir,args.allow_partial,args.data),indent=2))
