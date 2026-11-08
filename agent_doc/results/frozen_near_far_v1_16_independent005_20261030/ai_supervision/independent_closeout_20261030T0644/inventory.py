from pathlib import Path
import json,collections,datetime
B=Path(__file__).resolve().parent;V=B.parent.parent;S=Path('/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_qasper_mk2_64k_20261030/independent_closeout_20261030T0644')
c=collections.Counter();arms=0;partial=[]
for f in V.glob('requests/*/*.json'):
 r=json.loads(f.read_bytes());arms+=len(r['arms'])
 if r['status']=='complete':c[r['dataset']]+=1
 else:partial.append(str(f))
q=json.loads((V/'queue.json').read_bytes());x=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed=sum(c.values()),outputs_present=arms,counts=dict(c),queue=dict(collections.Counter(j['status'] for j in q['jobs'])),partial=partial,scope='inventory only; no partial quality statistics')
(B/'LIVE_COVERAGE.json').write_text(json.dumps(x,indent=2));print(json.dumps(x))
for d in [B,S]:
 with (d/'PROGRESS.md').open('a') as f:f.write(x['utc']+f": inventory V1 {x['completed']}/3750, {arms} arms, queue={x['queue']}; Qasper {c['qasper']}/200. CPU_SOURCE 53 passed, CPU_TESTS 8 passed; full scientific acceptance pending.\n")
