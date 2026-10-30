"""Re-score and audit existing paired frozen requests for the two ablation tasks."""
import hashlib,json,collections
from pathlib import Path
from benchmark.LongBench.eval import sample_score
P=Path('/home/wangyuanshuo02/two-level-attention')
R=P/'agent_doc/results/frozen_near_far_full5_independent005_20261029'
O=P/'agent_doc/results/nfa_ablation_20261030'
manifest=json.load((R/'manifest_rank0.json').open())
result=dict(source=str(R),manifest_sha256=hashlib.sha256((R/'manifest_rank0.json').read_bytes()).hexdigest(),
    protocol=manifest['protocol'],per_task={},raw_bindings=[],validation='pending independent reuse review')
for task in ('hotpotqa','gov_report'):
    aggregate={}
    files=sorted((R/'requests'/task).glob('*.json'))
    assert len(files)==200
    for f in files:
        raw=f.read_bytes();x=json.loads(raw)
        assert x['status']=='complete' and x['identity']==manifest['identity']
        result['raw_bindings'].append(dict(path=str(f),sha256=hashlib.sha256(raw).hexdigest()))
        for arm,a in x['arms'].items():
            score=sample_score(task,a['prediction'],x['answers'],[])
            assert abs(score-a['score'])<1e-9
            for layer,d in a['decisions'].items():
                profile=a['profiles'][layer]
                expected=dict(skip_near=arm.startswith('dual') and profile['near']['mean']<.05,
                    skip_far=arm!='full' and profile['far']['mean']<.05)
                # Absent regions are never labelled skipped.
                nodes=a['prefill_full_kv']['layers'][int(layer)]['nodes']
                expected['skip_near'] &= nodes['near']['tokens']>0
                expected['skip_far'] &= nodes['far']['tokens']>0
                assert d==expected
            z=aggregate.setdefault(arm,dict(n=0,score_sum=0,gates=collections.Counter(),generated_tokens=0,kv_ratio_sum=0))
            z['n']+=1;z['score_sum']+=score;z['gates'].update(a['gate_counts']);z['generated_tokens']+=a['generated_tokens']
            def nbytes(report):return sum(n['kv_bytes'] for layer in report['layers'] for n in layer['nodes'].values())
            z['kv_ratio_sum']+=nbytes(a['frozen_kv'])/nbytes(a['prefill_full_kv'])
    for arm,z in aggregate.items():
        z['score']=100*z.pop('score_sum')/z['n'];z['mean_generated_tokens']=z['generated_tokens']/z['n']
        z['mean_frozen_resident_kv_ratio']=z.pop('kv_ratio_sum')/z['n']
    result['per_task'][task]=aggregate
(O/'historical_frozen_recheck.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['per_task'],indent=2))
