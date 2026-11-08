from pathlib import Path
"""Independent derived-artifact checks. No producer imports. Requires full raw CPU audit separately."""
import csv, io, json, math, hashlib
TASKS=['narrativeqa','qasper','multifieldqa_en','hotpotqa','2wikimqa','musique','gov_report','qmsum','multi_news','trec','triviaqa','samsum','passage_count','passage_retrieval_en','lcc','repobench-p']
def verify(p, records, summary, index, preds, cfg, ident, ck, bindings):
    arms=['full','far_0.05','dual_0.05']; ds=cfg['datasets']
    ck(ds==TASKS,'fixed 16 task order');ck(summary['protocol']==cfg,'summary protocol')
    ck(summary['completed_requests']==summary['expected_requests']==3750 and summary['partial_requests']==0,'summary denominators')
    ck(summary['completed_per_task']=={d:sum(k[0]==d for k in records) for d in ds},'summary task counts')
    ck(set(summary['arms'])==set(arms),'summary arms')
    names=[a+'-vs-'+b for a,b in [(arms[1],arms[0]),(arms[2],arms[0]),(arms[2],arms[1])]]
    ck(set(summary['comparisons'])==set(names),'comparison exact set')
    for name,c in summary['comparisons'].items():
        ck(set(c['per_task'])==set(ds),'comparison task set '+name)
        for scope,e in list(c['per_task'].items())+[('avg16',c['avg16'])]:
            ci=e['ci95_pp'];ck(isinstance(ci,list) and len(ci)==2 and all(math.isfinite(float(x)) for x in ci) and ci[0]<=ci[1],'CI shape/order '+name+'/'+scope)
    def eq(actual,want,label):
        if want is None:ck(actual is None,label)
        elif isinstance(want,dict):
            ck(isinstance(actual,dict) and set(actual)==set(want),label+' keys')
            if isinstance(actual,dict):
                for k,v in want.items():
                    if k in actual:eq(actual[k],v,label+'/'+k)
        elif isinstance(want,(float,int)):
            ck(isinstance(actual,(int,float)) and math.isfinite(actual) and abs(actual-want)<1e-10,label)
        else:ck(actual==want,label)
    def avg(v):return sum(v)/len(v) if v else None
    rs=list(records.values())
    for arm in arms:
        counts=dict.fromkeys(['keep','near','far','both'],0); near=far=exact=changed=0; ratios=[];work=[];nk=[];nw=[];ta={}
        for d in ds:
            group=[r for r in rs if r['dataset']==d]
            ta[d]=dict(n=len(group),near_active=sum(any(x['skip_near'] for x in r['arms'][arm]['decisions'].values()) for r in group),far_active=sum(any(x['skip_far'] for x in r['arms'][arm]['decisions'].values()) for r in group),exact_full_generation=sum(r['arms'][arm]['generated_ids']==r['arms']['full']['generated_ids'] for r in group))
        for r in rs:
            a=r['arms'][arm];before=a['prefill_full_kv'];after=a['frozen_kv'];ratios.append(after['kv_bytes']/before['kv_bytes'])
            local=dict.fromkeys(counts,0)
            for li,d in a['decisions'].items():
                sn,sf=d['skip_near'],d['skip_far'];local['both' if sn and sf else 'near' if sn else 'far' if sf else 'keep']+=1
                if sn and not sf:
                    b,z=before['layers'][int(li)],after['layers'][int(li)]
                    nk.append(sum(n['kv_bytes'] for n in z['nodes'].values())/sum(n['kv_bytes'] for n in b['nodes'].values()));nw.append(z['attention_tokens']/b['attention_tokens'])
                if arm==arms[2]:changed+=sf!=r['arms'][arms[1]]['decisions'][li]['skip_far']
            eq(a['gate_counts'],local,'raw gate counts')
            for k,v in local.items():counts[k]+=v
            work.append(sum(z['attention_tokens'] for z in after['layers'])/sum(z['attention_tokens'] for z in before['layers']))
        expected=dict(gate_counts=counts,near_active_requests=sum(x['near_active'] for x in ta.values()),far_active_requests=sum(x['far_active'] for x in ta.values()),exact_full_generation=sum(x['exact_full_generation'] for x in ta.values()),per_task_activation=ta,frozen_kv_ratio_mean=avg(ratios),frozen_attention_token_ratio_mean=avg(work),near_only_layer_count=len(nk),near_only_layer_kv_ratio_mean=avg(nk),near_only_layer_attention_ratio_mean=avg(nw),far_decisions_differ_from_far_only=changed)
        for k,v in expected.items():eq(summary['arms'][arm][k],v,'summary '+arm+'/'+k)
    ck(index['identity']==ident and Path(index['root']).resolve()==Path(json.loads((Path(__file__).resolve().parent.parent/'CAPTURE_BINDINGS.json').read_text())['fixed/v1/raw_index.json']['origin']).parent.resolve(),'index identity/root')
    for x in index['records']:
        k=(x['dataset'],x['row_index']);path='requests/%s/%04d.json'%k
        ck(x['path']==path and k in records,'index key/path')
        if k in records:ck(x['input_sha256']==records[k]['input']['input_sha256'] and x['bytes']==(p/path).stat().st_size,'index input/bytes')
    for x in preds:
        r=records[(x['dataset'],x['row_index'])];a=r['arms'][x['arm']]
        ck(x['answers']==r['answers'] and x['input_sha256']==r['input']['input_sha256'] and x['gate_counts']==a['gate_counts'],'prediction extra binding')
    def tsv(name,header,rows):
        f=p/name;b=f.read_bytes();bindings[str(f.resolve())]=hashlib.sha256(b).hexdigest();got=list(csv.reader(io.StringIO(b.decode()),delimiter='\t'))
        want=[[str(x) if x is not None else '' for x in row] for row in [header]+rows]
        ck(got==want,'TSV '+name)
    rows=[]
    for arm in arms:
        a=summary['arms'][arm];c=summary['comparisons'].get(arm+'-vs-full',{}).get('avg16')
        rows.append([arm]+[a['per_task'][d] for d in ds]+[a['avg16'],c['delta_pp'] if c else 0,*(c['ci95_pp'] if c else ['',''])])
    tsv('scores.tsv',['arm']+ds+['AVG16','delta_vs_full_pp','ci95_low_pp','ci95_high_pp'],rows)
    rows=[]
    for name,c in summary['comparisons'].items():
        for d,e in c['per_task'].items():rows.append([name,d,e['n'],e['delta_pp'],*e['ci95_pp'],e['degraded'],e['improved']])
        e=c['avg16'];rows.append([name,'AVG16',3750,e['delta_pp'],*e['ci95_pp'],'',''])
    tsv('paired_differences.tsv',['comparison','scope','n','delta_pp','ci95_low_pp','ci95_high_pp','degraded','improved'],rows)
    rows=[];tr=[]
    for arm in arms:
        a=summary['arms'][arm];rows.append([arm,*[a['gate_counts'][k] for k in ['keep','near','far','both']],a['near_active_requests'],a['far_active_requests'],a['frozen_kv_ratio_mean'],a['frozen_attention_token_ratio_mean'],a['near_only_layer_kv_ratio_mean'],a['near_only_layer_attention_ratio_mean'],a['exact_full_generation'],a['far_decisions_differ_from_far_only']])
        for d,e in a['per_task_activation'].items():tr.append([arm,d,e['n'],e['near_active'],e['far_active'],e['exact_full_generation']])
    tsv('activation_storage.tsv',['arm','keep_layers','near_only_layers','far_only_layers','both_layers','near_active_requests','far_active_requests','frozen_resident_KV_ratio','frozen_attention_token_ratio','near_only_layer_KV_ratio','near_only_layer_attention_token_ratio','exact_full_generations','changed_far_decisions_vs_far_only'],rows)
    tsv('per_task_activation.tsv',['arm','dataset','n','near_active_requests','far_active_requests','exact_full_generation'],tr)
    rows=[]
    for k,r in sorted(records.items()):
        for arm,a in r['arms'].items():rows.append([*k,arm,a['score']*100,(a['score']-r['arms']['full']['score'])*100,r['input']['input_tokens'],a['generated_tokens'],*[a['gate_counts'][s] for s in ['near','far','both']]])
    tsv('per_request_scores.tsv',['dataset','row_index','arm','score','delta_vs_full_pp','input_tokens','generated_tokens','near_only_layers','far_only_layers','both_layers'],rows)
