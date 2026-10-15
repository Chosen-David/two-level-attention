import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from sparse_attn.ops.eager_prefill import sparse_prefill_attn

class Runtime(unittest.TestCase):
    def test_default_reference_and_candidate_dense(self):
        idx=SimpleNamespace(args=SimpleNamespace(tia_level2_topk=128,tia_block_size=64),sink_tokens=1,sliding_window_size=1)
        q=torch.randn(1,7,4,8);k=torch.randn(1,7,2,8);v=torch.randn_like(k)
        outputs=[]
        for runtime in ('reference','candidate'):
            with patch.dict(os.environ,TLI_PREFILL_RUNTIME=runtime,TLI_PREFILL_SELECT_GROUP='0',TLI_PREFILL_PREFETCH='0'):
                outputs.append(sparse_prefill_attn(idx,q,k,v,chunk_size=3))
        torch.testing.assert_close(*outputs,rtol=0,atol=0)
    def test_candidate_preserves_main_guards_and_schema(self):
        idx=SimpleNamespace(args=SimpleNamespace(tia_level2_topk=128))
        q=torch.zeros(1,4,2,8);k=torch.zeros(1,4,1,8)
        with patch.dict(os.environ,TLI_PREFILL_RUNTIME='candidate'):
            for qq,kk,vv,chunk in [(q.expand(2,-1,-1,-1),k,k,2),(q,k,k,True),(q,k,k,0),(q,k,k.double(),2),(q,k,k.to('meta'),2)]:
                with self.assertRaises(ValueError):sparse_prefill_attn(idx,qq,kk,vv,chunk_size=chunk)
    def test_grouped_candidate_matches_actual_row_oracle(self):
        from sparse_attn.ops.candidate_prefill import _chunk_rows_attn
        from sparse_attn.ops.selection_contract import row_support_receipt
        q=torch.zeros(1,5,4,2);k=torch.zeros(1,9,2,2)
        v=torch.arange(9,dtype=torch.float32).view(1,9,1,1).expand(1,9,2,2)
        sels=[]
        for end,ids in [(6,[0,5]),(8,[1,7]),(9,[2,8])]:
            mask=torch.zeros(2,end,dtype=torch.bool);mask[:,ids]=True;sels.append(mask)
        for sink,swa in [(0,0),(0,2),(3,1)]:
            with patch.dict(os.environ,TLI_PREFILL_FUSED_ATTN='0'):
                actual=_chunk_rows_attn(q,k,v,None,4,2,1.,sink_tok=sink,swa_tok=swa,sel_list=sels,grp=2)
            for r in range(5):
                shared=sels[r//2][0].nonzero().flatten().tolist()
                receipt=row_support_receipt(shared,query_position=4+r,length=9,sink=sink,swa=swa,k_cap=9)
                expected=sum(receipt['support_ids'])/receipt['k_real']
                torch.testing.assert_close(actual[0,r],torch.full_like(actual[0,r],expected))
    def test_unknown_runtime_rejected(self):
        idx=SimpleNamespace(args=SimpleNamespace(tia_level2_topk=128))
        q=torch.zeros(1,4,2,8);k=torch.zeros(1,4,1,8)
        with patch.dict(os.environ,TLI_PREFILL_RUNTIME='unknown'):
            with self.assertRaises(ValueError):sparse_prefill_attn(idx,q,k,k)

if __name__=='__main__':unittest.main()
