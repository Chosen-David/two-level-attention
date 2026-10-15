import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from exp.paper.contract import *
from exp.paper.observations import mass_partition,o1_mass,protected_ids
from exp.paper.workflow import plan
from sparse_attn.ops.selection_contract import support_receipt,require_support,row_support_receipt,match_support


def protocol():
    route={x:['dense'] for x in ('material','question','decode')}
    identity={k:'fixed' for k in IDENTITY}
    identity.update(scorer_sha256='a'*64,config_sha256='b'*64,producer_commit='c'*40,
        phase_route_by_layer=route,first_dense_layers=[],batch=1,chunk=64,max_new_tokens=8,
        comparison_track='native',selection_semantics={'kind':'dense'})
    overrides={a:{k:identity[k] for k in ('backend_id','selection_semantics','config_sha256','phase_route_by_layer','first_dense_layers')} for a in ('dense','method')}
    return dict(schema='paper-run-v1',status='frozen',identity=identity,arm_overrides=overrides,
        samples=[dict(sample_id='s',document_id='d',input_ids_sha256='d'*64,split='confirmation')],
        context_limit=128,repeats=1,warmup=0,timing_level='T3',cache_state='request-cold',
        order_policy='AB/BA',run_mode='natural',timer_boundary='model-entry-to-last-output-complete',sync_policy='before-after',arms=['dense','method'])


def row(p,arm):
    s=p['samples'][0]
    return dict(s,arm=arm,protocol_sha256='e'*64,identity=dict(p['identity'],**p['arm_overrides'][arm]),
        repetition=0,status='ok',measured_this_run=True,timing_level='T3',cache_state=p['cache_state'],run_mode='natural',
        timer_boundary=p['timer_boundary'],sync_policy=p['sync_policy'],aggregation='direct-wall',
        request_wall_ms=10 if arm=='dense' else 20,actual_ctx_tokens=64,actual_generated_tokens=3,
        lifecycle={k:dict(ms=0,included_in_wall=True,overlap=False) for k in LIFECYCLE},fallback_count=0,
        actual_routes=p['identity']['phase_route_by_layer'],order=0 if arm=='dense' else 1,
        support={'query0:head0':support_receipt(list(range(64)),query_position=63,length=64,sink=2,swa=0,k_cap=64)})


class Contract(unittest.TestCase):
    def test_default_plan_never_launches(self):
        p=json.loads((Path(__file__).parents[1]/'protocol.unfrozen.json').read_text())
        r=plan(p);self.assertEqual(r['protocol_status'],'blocked');self.assertEqual(r['launch_commands'],[])
    def test_protocol_freeze_and_negative_types(self):
        p=protocol();self.assertEqual(protocol_errors(p),[])
        for key,value in [('model_revision',False),('batch',True),('first_dense_layers',-1),('producer_commit','main')]:
            q=copy.deepcopy(p);q['identity'][key]=value;self.assertTrue(protocol_errors(q))
        q=copy.deepcopy(p);q['samples'].append(dict(q['samples'][0],sample_id='s2',split='development'))
        self.assertIn('document leakage',protocol_errors(q))
    def test_direct_timing_preserves_negative_and_missing(self):
        p=protocol();a=row(p,'dense');b=row(p,'method')
        r=summarize([a,b],p,'e'*64);self.assertEqual(r['pairs'][0]['dense_over_method'],.5)
        self.assertIsNone(r['confidence_interval']);self.assertEqual(summarize([a],p,'e'*64)['n_missing'],1)
        b['status']='oom';self.assertEqual(summarize([a,b],p,'e'*64)['n_failed'],1)
    def test_false_timing_or_route_rejected(self):
        p=protocol()
        for key,value in [('aggregation','sum-of-medians'),('fallback_count',1),('actual_routes',{'decode':['sparse']}),
                          ('request_wall_ms',float('nan')),('request_wall_ms',True),('measured_this_run',False),('order',1),('support',{})]:
            r=row(p,'dense');r[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): validate_timing(r,p,'e'*64)
    def test_support_tamper_and_tpot(self):
        p=protocol()
        for field,value in [('k_cap',1),('duplicate_count',99),('missing_protected_count',1),('query_position',3)]:
            r=row(p,'dense');r['support']['query0:head0'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_timing(r,p,'e'*64)
        p['timing_level']='T2';a=row(p,'dense');b=row(p,'method')
        for r in (a,b):r['timing_level']='T2';r['decode_after_first_wall_ms']=10
        b['actual_generated_tokens']=6
        result=summarize([a,b],p,'e'*64)
        self.assertEqual(result['pairs'][0]['dense_over_method'],2.5)
        self.assertEqual(result['pairs'][0]['comparison_metric'],'tpot_ms')
    def test_controlled_support_not_cap_only(self):
        p=protocol();p['identity']['comparison_track']='controlled';p['controlled_factor']='implementation'
        a=row(p,'dense');b=row(p,'method');b['support']['query0:head0']=support_receipt(list(range(32)),query_position=63,length=64,sink=2,swa=0,k_cap=64)
        with self.assertRaises(ValueError):summarize([a,b],p,'e'*64)
    def test_controlled_row_binding(self):
        p=protocol();p['identity']['comparison_track']='controlled';p['controlled_factor']='implementation'
        a=row(p,'dense');b=row(p,'method')
        a['support']['query0:head0']=support_receipt(list(range(32)),query_position=32,length=64,sink=2,swa=0,k_cap=64)
        b['support']['query0:head0']=support_receipt(list(range(32)),query_position=33,length=64,sink=2,swa=0,k_cap=64)
        with self.assertRaises(ValueError):summarize([a,b],p,'e'*64)
    def test_support_union_and_invalid_indices(self):
        r=support_receipt([0,1,2],query_position=2,length=3,sink=2,swa=2,k_cap=1)
        self.assertEqual(r['protected_real'],3);require_support(r)
        for ids in ([0,0,1,2],[0,1],[0,1,3],[True,1,2]):
            with self.assertRaises(ValueError):require_support(support_receipt(ids,query_position=2,length=3,sink=2,swa=2,k_cap=3))
        # Group union with each row's SWA must be counted, not assumed equal K.
        r=row_support_receipt([0,1,7],query_position=4,length=8,sink=1,swa=2,k_cap=2)
        self.assertEqual(r['support_ids'],[0,1,3,4]);self.assertEqual(r['cap_excess'],1)
        with self.assertRaises(ValueError):match_support({'u':r},{'u':r})
    def test_mass_conservation_and_protection_overlap(self):
        p=np.ones(5)/5;prot=protected_ids(4,3,3);self.assertEqual(prot,list(range(5)))
        m=mass_partition(p,[1,2],prot);self.assertAlmostEqual(m['selected_protected_mass'],.4)
        self.assertEqual(m['selected_nonprotected_mass'],0)
        bins=o1_mass(p,4,3,[0,4],[0,2,5]);self.assertAlmostEqual(sum(x['mass'] for x in bins),.6)
        for ids in ([1,1],[5],[-1]):
            with self.assertRaises(ValueError):mass_partition(p,ids,[])
        with self.assertRaises(ValueError):o1_mass(p,4,3,[],[0,4])

if __name__=='__main__':unittest.main()
