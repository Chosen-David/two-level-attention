from types import SimpleNamespace
import torch
from benchmark.LongBench.prefill_frozen_skip import V1_TASKS,NO_CHAT_TASKS,prepare_ids,stopping_policy,greedy_token

class Tokenizer:
    eos_token_id=9
    def __call__(self,text,**kwargs):return SimpleNamespace(input_ids=[1,2])
    def apply_chat_template(self,*args,**kwargs):self.chat=True;return 'chat'
    def encode(self,*args,**kwargs):return [7]

def test_v1_task_and_prompt_scope():
    assert len(V1_TASKS)==len(set(V1_TASKS))==16
    for ds in V1_TASKS:
        t=Tokenizer();t.chat=False
        prepare_ids(t,{'context':'text'},ds,'{context}',100)
        assert t.chat==(ds not in NO_CHAT_TASKS)

def test_samsum_newline_stop_and_first_token_minimum():
    t=Tokenizer();model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[9,10]))
    ids,minimum=stopping_policy(model,t,'samsum')
    assert ids==[9,7] and minimum==1
    logits=torch.zeros(1,12);logits[0,7]=5;logits[0,9]=4;logits[0,3]=3
    assert greedy_token(logits,0,ids,minimum).item()==3
    assert greedy_token(logits,1,ids,minimum).item()==7
    assert stopping_policy(model,t,'qasper')==([9,10],0)
