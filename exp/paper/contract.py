"""Fail-closed public experiment identities and timing receipts. Standard library."""
import hashlib
import json
import math
import re
from pathlib import Path

IDENTITY = ('model_revision', 'tokenizer_revision', 'dataset_revision', 'scorer_sha256',
            'template', 'thinking', 'bos_eos', 'rope', 'dtype', 'generation',
            'phase_route_by_layer', 'first_dense_layers', 'batch', 'chunk',
            'max_new_tokens', 'config_sha256', 'producer_commit', 'backend_id',
            'selection_semantics', 'comparison_track')
LIFECYCLE = ('compile', 'plan', 'capture', 'prepare', 'metadata_update', 'injected_copy', 'replay', 'sync')
TIMING = {'T0': 'kernel_ms', 'T1': 'attention_pipeline_wall_ms',
          'T2': 'decode_after_first_wall_ms', 'T3': 'request_wall_ms'}


def need(condition, message):
    if not condition: raise ValueError(message)


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(obj): return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def known(value):
    if value is None: return False
    if isinstance(value, str): return value.strip() != '' and value.upper() not in ('UNFROZEN','UNKNOWN','NR','TODO','HEAD','MAIN','LATEST')
    if isinstance(value, dict): return bool(value) and all(known(v) for v in value.values())
    if isinstance(value, list): return all(known(v) for v in value)
    return type(value) in (bool, int, float) and (type(value) is not float or math.isfinite(value))


def hash_value(value): return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def protocol_errors(p):
    errors = []
    def check(ok, message):
        if not ok: errors.append(message)
    check(p.get('schema') == 'paper-run-v1', 'schema')
    check(p.get('status') == 'frozen', 'status:UNFROZEN')
    identity = p.get('identity', {})
    for key in IDENTITY: check(known(identity.get(key)), 'identity.'+key)
    for key in ('model_revision','tokenizer_revision','dataset_revision','template','rope','dtype','generation','backend_id'):
        check(isinstance(identity.get(key),str) and known(identity[key]),'string identity:'+key)
    for fields in (identity,p.get('arm_overrides',{}).get('dense',{}),p.get('arm_overrides',{}).get('method',{})):
        layers=fields.get('first_dense_layers')
        check(isinstance(layers,list) and all(type(x) is int and x>=0 for x in layers) and len(set(layers))==len(layers),'dense layer list')
    check(identity.get('comparison_track') in ('native', 'controlled'), 'comparison_track')
    for k in ('scorer_sha256','config_sha256'): check(hash_value(identity.get(k)), 'identity hash:'+k)
    check(isinstance(identity.get('producer_commit'), str) and bool(re.fullmatch('[0-9a-f]{40}', identity['producer_commit'])), 'producer_commit')
    route = identity.get('phase_route_by_layer', {})
    check(isinstance(route, dict) and set(route)=={'material','question','decode'} and all(isinstance(v,list) and v and all(known(x) for x in v) for v in route.values()), 'phase_route_by_layer')
    for k in ('batch','chunk','max_new_tokens'):
        check(type(identity.get(k)) is int and identity[k] > 0, 'positive int:'+k)
    overrides=p.get('arm_overrides',{})
    check(isinstance(overrides,dict) and set(overrides)=={'dense','method'}, 'arm_overrides')
    for arm in ('dense','method'):
        override=overrides.get(arm,{})
        allowed={'backend_id','selection_semantics','config_sha256','phase_route_by_layer','first_dense_layers'}
        check(set(override)==allowed and all(known(v) for v in override.values()), 'arm identity:'+arm)
        check(hash_value(override.get('config_sha256')), 'arm config hash:'+arm)
        check(isinstance(override.get('backend_id'),str) and known(override.get('backend_id')), 'arm backend:'+arm)
        semantics=override.get('selection_semantics')
        check(isinstance(semantics,dict) and bool(semantics) and all(known(v) for v in semantics.values()), 'arm semantics:'+arm)
        routes=override.get('phase_route_by_layer')
        check(isinstance(routes,dict) and set(routes)=={'material','question','decode'} and all(isinstance(v,list) and v and all(isinstance(x,str) and known(x) for x in v) for v in routes.values()), 'arm routes:'+arm)
    if identity.get('comparison_track')=='controlled':
        factor=p.get('controlled_factor')
        check(isinstance(factor,str) and known(factor),'controlled_factor')
        a=overrides.get('dense',{}).get('selection_semantics',{})
        b=overrides.get('method',{}).get('selection_semantics',{})
        check(isinstance(a,dict) and isinstance(b,dict) and set(a)==set(b) and all(a[k]==b[k] for k in a if k!=factor),'controlled semantics differ beyond factor')
    planned = p.get('samples', [])
    check(isinstance(planned,list) and bool(planned), 'samples')
    seen=set(); document_splits={}; hashes={}
    for row in planned:
        sid=row.get('sample_id'); doc=row.get('document_id'); split=row.get('split')
        check(known(sid) and sid not in seen, 'sample ID'); seen.add(sid)
        check(known(doc) and split in ('calibration','development','confirmation'), 'document split')
        check(doc not in document_splits or document_splits[doc]==split, 'document leakage')
        document_splits[doc]=split
        h=row.get('input_ids_sha256'); check(hash_value(h), 'input hash')
        check(h not in hashes or hashes[h]==doc, 'same input relabeled as independent documents'); hashes[h]=doc
    coverage=p.get('support_units')
    check(isinstance(coverage,dict) and set(coverage)==seen, 'planned support sample coverage')
    if isinstance(coverage,dict):
        for arms in coverage.values():
            check(isinstance(arms,dict) and set(arms)=={'dense','method'}, 'planned support arm coverage')
            if not isinstance(arms,dict): continue
            for units in arms.values():
                check(isinstance(units,dict) and bool(units), 'planned support units')
                if not isinstance(units,dict): continue
                for unit,geom in units.items():
                    fields={'query_position','length','sink','swa'}
                    check(isinstance(unit,str) and known(unit) and isinstance(geom,dict) and set(geom)==fields and all(type(v) is int and v>=0 for v in geom.values()) and geom['query_position']<geom['length'], 'planned support geometry')
    for k in ('context_limit','repeats','warmup'):
        check(type(p.get(k)) is int and p[k] >= (0 if k=='warmup' else 1), 'int:'+k)
    check(p.get('timing_level') in TIMING, 'timing_level')
    check(p.get('cache_state') in ('process-cold','request-cold','steady-warm'), 'cache_state')
    check(p.get('order_policy') in ('AB/BA','seeded-interleave'), 'order_policy')
    if p.get('order_policy')=='seeded-interleave': check(type(p.get('order_seed')) is int,'order_seed')
    check(p.get('run_mode') in ('natural','fixed-workload','fixed-QK'), 'run_mode')
    check(known(p.get('timer_boundary')) and known(p.get('sync_policy')), 'timing boundary/sync')
    check(p.get('arms') == ['dense','method'], 'paired arms')
    return errors


def validate_timing(row, p, protocol_sha):
    errors=protocol_errors(p); need(not errors, 'unfrozen protocol: '+','.join(errors))
    need(row.get('protocol_sha256')==protocol_sha, 'protocol identity mismatch')
    expected_identity=dict(p['identity'],**p['arm_overrides'].get(row.get('arm'),{}))
    need(row.get('identity')==expected_identity, 'run identity mismatch')
    samples={r['sample_id']:r for r in p['samples']}
    sample=samples.get(row.get('sample_id')); need(sample is not None, 'unplanned sample')
    for key in ('document_id','input_ids_sha256'): need(row.get(key)==sample[key], 'sample identity mismatch')
    need(row.get('arm') in p['arms'], 'unknown arm')
    need(type(row.get('repetition')) is int and 0<=row['repetition']<p['repeats'], 'unplanned repetition')
    need(row.get('status') in ('ok','failed','skipped','oom'), 'missing result status')
    need(type(row.get('measured_this_run')) is bool, 'missing measured_this_run')
    if row['status']!='ok': return
    need(row['measured_this_run'], 'resume record is not a timing measurement')
    for k in ('timing_level','cache_state','run_mode','timer_boundary','sync_policy'):
        need(row.get(k)==p[k], 'timing mismatch:'+k)
    need(row.get('aggregation')=='direct-wall', 'component sums are not measured wall')
    metric=TIMING[p['timing_level']]; value=row.get(metric)
    need(type(value) in (int,float) and math.isfinite(value) and value>0, 'missing/nonfinite wall metric')
    need(type(row.get('actual_ctx_tokens')) is int and 0<row['actual_ctx_tokens']<=p['context_limit'], 'unsupported context')
    n=row.get('actual_generated_tokens')
    need(type(n) is int and 0<=n<=p['identity']['max_new_tokens'], 'generated token count')
    if p['timing_level']=='T2': need(n>1, 'TPOT undefined with fewer than two outputs')
    need(set(row.get('lifecycle',{}))==set(LIFECYCLE), 'incomplete lifecycle')
    for event in row['lifecycle'].values():
        need(set(event)=={'ms','included_in_wall','overlap'}, 'lifecycle schema')
        need(type(event['ms']) in (int,float) and math.isfinite(event['ms']) and event['ms']>=0, 'lifecycle time')
        need(type(event['included_in_wall']) is bool and type(event['overlap']) is bool, 'lifecycle attribution')
    need(type(row.get('fallback_count')) is int and row['fallback_count']==0, 'target route fallback prevents eligible timing')
    need(row.get('actual_routes')==expected_identity['phase_route_by_layer'], 'actual phase routes differ')
    expected_order = (['dense','method'] if row['repetition']%2==0 else ['method','dense'])
    if p['order_policy']=='seeded-interleave':
        import random
        rng=random.Random(p['order_seed']+row['repetition'])
        rng.shuffle(expected_order)
    need(type(row.get('order')) is int and row['order']==expected_order.index(row['arm']), 'pair order mismatch')
    support=row.get('support')
    need(isinstance(support,dict) and bool(support), 'missing actual unique support receipt')
    planned=p['support_units'][row['sample_id']][row['arm']]
    need(set(support)==set(planned),'actual support coverage differs from frozen units')
    for unit,value in support.items():
        need(all(value.get(k)==v for k,v in planned[unit].items()), 'actual support geometry differs from frozen unit')
        need(known(unit) and isinstance(value,dict), 'support unit')
        from sparse_attn.ops.selection_contract import support_receipt, require_support
        geometry = ('query_position','length','sink','swa','k_cap')
        need(all(k in value for k in geometry) and 'support_ids' in value, 'missing row support geometry/IDs')
        need(value['length']==row['actual_ctx_tokens'], 'support sequence length mismatch')
        expected=require_support(support_receipt(value['support_ids'],**{k:value[k] for k in geometry}))
        need(value==expected, 'support receipt differs from actual unique causal IDs')



def summarize(rows,p,sha):
    """No CI invented. Preserve failure denominator and pair repetitions exactly."""
    errors=protocol_errors(p); need(not errors, 'unfrozen protocol: '+','.join(errors))
    expected={(s['sample_id'],a,i) for s in p['samples'] for a in p['arms'] for i in range(p['repeats'])}
    indexed={}
    for row in rows:
        validate_timing(row,p,sha)
        key=(row['sample_id'],row['arm'],row['repetition'])
        need(key not in indexed,'duplicate timing row');indexed[key]=row
    missing=expected-set(indexed); failed=[k for k,r in indexed.items() if r['status']!='ok']
    out=dict(status='blocked' if missing or failed else 'descriptive-only', n_planned=len(expected),
             n_received=len(rows),n_missing=len(missing),n_failed=len(failed),
             timing_level=p['timing_level'], metric=TIMING[p['timing_level']],pairs=[],
             confidence_interval=None,quality_claim=None,gpu_acceptance=None)
    if out['status']=='blocked': return out
    metric=out['metric']
    for sample in p['samples']:
        for i in range(p['repeats']):
            a=indexed[sample['sample_id'],'dense',i];b=indexed[sample['sample_id'],'method',i]
            need(a['actual_ctx_tokens']==b['actual_ctx_tokens'], 'pair context mismatch')
            if p['run_mode']=='fixed-workload': need(a['actual_generated_tokens']==b['actual_generated_tokens'],'fixed workload mismatch')
            if p['identity']['comparison_track']=='controlled':
                need(set(a['support'])==set(b['support']) and all(a['support'][u]['k_real']==b['support'][u]['k_real'] for u in a['support']), 'controlled realized support mismatch')
                need(all(all(a['support'][u][field]==b['support'][u][field] for field in ('query_position','length','sink','swa')) for u in a['support']), 'controlled row/protection mismatch')
            if p['timing_level']=='T2':
                a_value=a[metric]/(a['actual_generated_tokens']-1)
                b_value=b[metric]/(b['actual_generated_tokens']-1)
            else:
                a_value,b_value=a[metric],b[metric]
            out['pairs'].append(dict(sample_id=sample['sample_id'],document_id=sample['document_id'],repetition=i,
                dense_ms=a[metric],method_ms=b[metric],dense_over_method=a_value/b_value,
                comparison_metric='tpot_ms' if p['timing_level']=='T2' else metric,
                dense_comparison_ms=a_value,method_comparison_ms=b_value,
                dense_tokens=a['actual_generated_tokens'],method_tokens=b['actual_generated_tokens']))
    return out
