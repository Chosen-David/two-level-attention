"""Execute actual pred.py function AST with CPU model/tokenizer/clock doubles.

Heavy loader imports are excluded; this does not certify real HF/GPU behavior.
"""
import ast
from contextlib import contextmanager
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from exp.paper import receipt

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'benchmark/LongBench/pred.py'
tree=ast.parse(SOURCE.read_text())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('get_pred','_manual_generate','_paper_capture_request','_split_prompt_question','_drop_leading_bos')]
ns={'torch':torch,'tqdm':lambda x:x,'post_process':lambda x,name:x,'build_chat':lambda t,p,m:p,
    'get_metrics':lambda:SimpleNamespace(get_select_tokens=lambda:0,clear=lambda:None)}
exec(compile(ast.Module(body=functions,type_ignores=[]),str(SOURCE),'exec'),ns)

class Clock:
    def __init__(self):self.value=0
    def tick(self,ms):self.value+=ms/1000
    def __call__(self):return self.value
    def sync(self):self.tick(1)

class Attention(torch.nn.Module):
    def forward(self,input_ids,cache):
        receipt.record_qwen_route(0,'sparse-prefill:reference' if cache is None else 'sparse-decode')
Attention.forward.__module__='sparse_attn.patches.qwen3_attn_patch'
Attention.forward.__name__='qwen3_attn_forward'

class Model:
    def __init__(self,clock,tokens=(7,8,3,4,5)):
        self.clock=clock;self.tokens=iter(tokens);self.inputs=[]
        self.model=SimpleNamespace(layers=[SimpleNamespace(self_attn=Attention())])
    def __call__(self,input_ids,past_key_values,use_cache):
        self.clock.tick(10);self.inputs.append(input_ids.tolist()[0])
        self.model.layers[0].self_attn(input_ids,past_key_values)
        logits=torch.zeros(1,input_ids.shape[1],10);logits[:,-1,next(self.tokens)]=1
        return SimpleNamespace(logits=logits,past_key_values=object())

class Batch:
    def __init__(self,ids):self.input_ids=torch.tensor([ids],dtype=torch.long)
    def to(self,*a):return self
    def __setitem__(self,key,value):
        if key!='input_ids':raise KeyError(key)
        self.input_ids=value

class Tokenizer:
    eos_token_id=9;bos_token_id=None
    def __call__(self,text,**kwargs):return Batch([ord(c)%9+1 for c in text])
    def decode(self,ids,**kwargs):return ','.join(map(str,ids))

class Capture(unittest.TestCase):
    def test_llama_chat_template_does_not_add_second_bos(self):
        class BosTokenizer(Tokenizer):
            bos_token_id=1
            def __call__(self,text,**kwargs):
                ids=([1,2] if text.startswith('<bos>') else [2]) if text else []
                if kwargs.get('add_special_tokens',True):ids=[1]+ids
                return Batch(ids)
        clock=Clock();model=Model(clock,tokens=[3])
        with patch.dict(ns,eager_decoding_request_guard=self.guard(clock),
                        build_chat=lambda t,p,m:'<bos>'+p),patch.object(torch.cuda,'empty_cache'):
            ns['get_pred'](model,BosTokenizer(),[{'text':'abcd','answers':[],'all_classes':[],'length':4}],64,1,'{text}','hotpotqa','cpu','Llama-3.1-8B')
        self.assertEqual(model.inputs,[[1,2]])

    def test_full_prompt_is_one_prefill_and_split_is_explicit(self):
        for mode, widths in [('full', [4, 1]), ('split', [1, 1, 1, 1, 1])]:
            clock=Clock();model=Model(clock,tokens=[3]*10)
            with patch.dict(ns,eager_decoding_request_guard=self.guard(clock)),patch.object(torch.cuda,'empty_cache'):
                rows=ns['get_pred'](model,Tokenizer(),[{'text':'abcd','answers':[],'all_classes':[],'length':4}],64,2,'{text}','trec','cpu','Qwen3-8B',prompt_mode=mode)
            self.assertEqual(list(map(len,model.inputs)),widths)
            self.assertEqual(rows[0]['prompt_mode'],mode)
    def guard(self,clock,fail=False):
        @contextmanager
        def run():
            clock.tick(2)
            yield
            clock.tick(3)
            if fail:raise RuntimeError('guard rejected')
        return run
    def test_exact_boundary_question_routing_and_token_identity(self):
        clock=Clock();model=Model(clock);a=torch.tensor([[1,2]]);b=torch.tensor([[6,7]])
        capture=receipt.QwenRequestCapture(model,a,b,sync=clock.sync,clock=clock)
        with capture,self.guard(clock)():tokens=ns['_manual_generate'](model,a,b,3,9,capture)
        r=capture.receipt
        self.assertEqual(tokens,[3,4,5]);self.assertAlmostEqual(r['ttft_ms'],33)
        self.assertAlmostEqual(r['decode_after_first_wall_ms'],21);self.assertAlmostEqual(r['tpot_ms'],10.5)
        self.assertAlmostEqual(r['request_wall_ms'],58)
        self.assertEqual(model.inputs,[[1,2],[6],[7],[3],[4]])
        self.assertEqual(r['input_ids_sha256'],hashlib.sha256(torch.tensor([[1,2,6,7]]).numpy().astype('<i8').tobytes()).hexdigest())
        self.assertEqual([c['phase'] for c in r['phase_calls']],['material','question','question','decode','decode'])
        self.assertEqual([c['layers'][0]['route'] for c in r['phase_calls']][1:3],['sparse-decode']*2)
        self.assertIsNone(r['support']);self.assertFalse(r['scientific_acceptance'])
    def test_first_eos_and_single_token_have_no_tpot(self):
        for maximum,eos in [(3,3),(1,9)]:
            clock=Clock();model=Model(clock,tokens=[3]);a=torch.tensor([[1,2]]);b=torch.empty((1,0),dtype=torch.long)
            capture=receipt.QwenRequestCapture(model,a,b,sync=clock.sync,clock=clock)
            with capture,self.guard(clock)():tokens=ns['_manual_generate'](model,a,b,maximum,eos,capture)
            self.assertEqual(tokens,[3]);self.assertIsNone(capture.receipt['tpot_ms']);self.assertEqual(len(model.inputs),1)
    def test_guard_failure_and_model_failure_never_publish(self):
        for fail_model in (False,True):
            clock=Clock();model=Model(clock,tokens=[] if fail_model else [3]);a=torch.tensor([[1,2]]);b=torch.empty((1,0),dtype=torch.long)
            capture=receipt.QwenRequestCapture(model,a,b,sync=clock.sync,clock=clock)
            with self.assertRaises((RuntimeError,StopIteration)):
                with capture,self.guard(clock,fail=not fail_model)():ns['_manual_generate'](model,a,b,1,9,capture)
            self.assertIsNone(capture.receipt);self.assertIsNone(receipt._QWEN_REQUEST.get())
    def test_unsupported_generate_and_incomplete_hooks(self):
        a=torch.tensor([[1,2]])
        for dataset,name in [('samsum','Qwen3-8B'),('trec','DeepSeek')]:
            with self.assertRaises(ValueError):ns['_paper_capture_request']('timing',None,a,a,dataset,name)
        model=Model(Clock());model.model.layers[0].self_attn=torch.nn.Identity()
        with self.assertRaises(ValueError):receipt.QwenRequestCapture(model,a,a)
        model=Model(Clock());capture=receipt.QwenRequestCapture(model,a,a,sync=lambda:None)
        with self.assertRaises(ValueError):
            with capture:capture.call('material',lambda **kw:None,input_ids=a)
        self.assertIsNone(capture.receipt)
    def test_nonfinite_reverse_and_zero_clock_rejected(self):
        for times in ([0.,float('inf'),float('inf'),float('inf')], [0.,float('nan'),1.,2.], [3.,2.,4.,5.], [0.,0.,0.,0.], [0.,1e308,1e308,1e308]):
            clock=Clock();model=Model(clock,tokens=[3]);a=torch.tensor([[1,2]]);b=torch.empty((1,0),dtype=torch.long)
            counter=iter(times);capture=receipt.QwenRequestCapture(model,a,b,sync=lambda:None,clock=lambda:next(counter))
            with self.assertRaises(ValueError):
                with capture,self.guard(clock)():ns['_manual_generate'](model,a,b,1,9,capture)
            self.assertIsNone(capture.receipt)
    def test_native_hook_and_cleanup(self):
        class Native(torch.nn.Module):
            def forward(self,input_ids,cache):return None
        Native.forward.__module__='transformers.models.qwen3.modeling_qwen3'
        Native.forward.__qualname__='Qwen3Attention.forward'
        clock=Clock();model=Model(clock,tokens=[3]);attn=Native();model.model.layers[0].self_attn=attn
        a=torch.tensor([[1,2]]);b=torch.empty((1,0),dtype=torch.long)
        capture=receipt.QwenRequestCapture(model,a,b,sync=clock.sync,clock=clock)
        with capture,self.guard(clock)():ns['_manual_generate'](model,a,b,1,9,capture)
        self.assertEqual(capture.receipt['phase_calls'][0]['layers'],[{'layer':0,'route':'native-qwen3-forward'}])
        self.assertFalse(attn._forward_hooks)
    def test_new_request_counter_reset_or_missing_field_fails(self):
        for before,after in [({'m':{'calls':1}},{'m':{'calls':0}}),({'m':{'calls':1}},{}),({'m':{'calls':1}},{'m':{}})]:
            clock=Clock();model=Model(clock,tokens=[3]);a=torch.tensor([[1,2]]);b=torch.empty((1,0),dtype=torch.long)
            capture=receipt.QwenRequestCapture(model,a,b,sync=clock.sync,clock=clock)
            with patch.object(receipt,'runtime_counters',side_effect=[before,after]),self.assertRaises(ValueError):
                with capture,self.guard(clock)():ns['_manual_generate'](model,a,b,1,9,capture)
            self.assertIsNone(capture.receipt)
    def test_duplicate_and_missing_layer_events_fail(self):
        for duplicate in (False,True):
            clock=Clock();model=Model(clock,tokens=[3]);a=torch.tensor([[1,2]]);b=torch.empty((1,0),dtype=torch.long)
            capture=receipt.QwenRequestCapture(model,a,b,sync=clock.sync,clock=clock)
            original=capture.route
            def route(layer,name):
                if duplicate:original(layer,name);original(layer,name)
            with patch.object(capture,'route',side_effect=route),self.assertRaises(ValueError):
                with capture,self.guard(clock)():ns['_manual_generate'](model,a,b,1,9,capture)
            self.assertIsNone(capture.receipt)
    def test_get_pred_guard_failure_cannot_append_record(self):
        original=receipt.QwenRequestCapture;clock=Clock();model=Model(clock,tokens=[3,4,5,6]);captures=[]
        def factory(*args,**kw):
            c=original(*args,**kw,sync=clock.sync,clock=clock);captures.append(c);return c
        with patch.dict(ns,eager_decoding_request_guard=self.guard(clock,fail=True)),patch.object(torch.cuda,'empty_cache'),patch.object(receipt,'QwenRequestCapture',side_effect=factory):
            with self.assertRaises(RuntimeError):
                ns['get_pred'](model,Tokenizer(),[{'text':'abcd','answers':[],'all_classes':[],'length':4}],64,2,'{text}','trec','cpu','Qwen3-8B',paper_capture_mode='timing')
        self.assertIsNone(captures[0].receipt)
    def test_get_pred_real_record_path_opt_in_and_default_disabled(self):
        original=receipt.QwenRequestCapture
        for enabled in (False,True):
            clock=Clock();model=Model(clock,tokens=[3,4,5,6])
            def factory(*args,**kw):return original(*args,**kw,sync=clock.sync,clock=clock)
            with patch.dict(ns,eager_decoding_request_guard=self.guard(clock)),patch.object(torch.cuda,'empty_cache'),patch.object(receipt,'QwenRequestCapture',side_effect=factory):
                records=ns['get_pred'](model,Tokenizer(),[{'text':'abcd','answers':[],'all_classes':[],'length':4}],64,2,'{text}','trec','cpu','Qwen3-8B',data_fp='hash',paper_capture_mode='timing' if enabled else 'off')
            self.assertEqual(len(records),1);self.assertEqual('paper_request_capture' in records[0],enabled)
            if enabled:self.assertEqual(records[0]['paper_request_capture']['actual_input_tokens'],4)
    def test_plan_branch_precedes_seed_or_model_load(self):
        main=next(n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=="__name__ == '__main__'")
        prefix=ast.Module(body=main.body[:2],type_ignores=[])
        import json
        local={'parse_args':lambda:SimpleNamespace(paper_capture='plan'),'json':json}
        with self.assertRaises(SystemExit) as raised:exec(compile(prefix,str(SOURCE),'exec'),local)
        self.assertEqual(raised.exception.code,0)

if __name__=='__main__':unittest.main()
