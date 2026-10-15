"""CPU random-weight actual DeepseekV2Attention.forward parity, no downloads.

Reuses the registered reconstruction diagnostic's AST loader. Full source hash
is reported and optionally required; no pretrained checkpoint or GPU claims.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple
import warnings
from unittest.mock import patch
import torch
from verify_e141_qk_reconstruction import definitions


def finite_probability_error(got, ref):
    """Fail before max aggregation can hide NaN; finite inputs may overflow too."""
    if not torch.isfinite(got).all() or not torch.isfinite(ref).all():
        raise ValueError('nonfinite got/ref probabilities')
    row_error = float((got-ref).abs().max())
    if not math.isfinite(row_error):
        raise ValueError('nonfinite probability row error')
    return row_error


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-source',type=Path,required=True)
    parser.add_argument('--require-sha256', default='7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c')
    parser.add_argument('--probe-source',type=Path,default=Path(__file__).resolve().parents[2]/'agent_doc/guide/paper_figs/trace/e141_mla_nope_probe.py')
    args=parser.parse_args()
    actual_hash=hashlib.sha256(args.upstream_source.read_bytes()).hexdigest()
    if args.require_sha256 and actual_hash!=args.require_sha256:
        raise ValueError('upstream identity mismatch')
    namespace=dict(torch=torch,nn=torch.nn,Optional=Optional,Tuple=Tuple,Cache=object,
                   DeepseekV2Config=object,warnings=warnings)
    official_hash=definitions(args.upstream_source,{'DeepseekV2RMSNorm','DeepseekV2RotaryEmbedding',
        'rotate_half','apply_rotary_pos_emb','DeepseekV2Attention'},namespace)
    probe=dict(torch=torch,DEV='cpu',LAYERS=[0],NOPE_D=128,ROPE_D=64)
    probe_hash=definitions(args.probe_source,{'get_mla_qk','rotate_half','head_mean_probs'},probe)
    torch.set_num_threads(1);torch.manual_seed(20261015)
    config=SimpleNamespace(attention_dropout=0.,hidden_size=8,num_attention_heads=2,
        max_position_embeddings=32,rope_theta=12345.,q_lora_rank=None,qk_rope_head_dim=64,
        kv_lora_rank=512,v_head_dim=128,qk_nope_head_dim=128,attention_bias=False,rope_scaling=None)
    attn=namespace['DeepseekV2Attention'](config,layer_idx=0).eval()
    norm=namespace['DeepseekV2RMSNorm'](8)
    with torch.no_grad():norm.weight.copy_(torch.linspace(.5,1.5,8))
    # Noncontiguous hidden states, different per-token magnitudes.
    hidden=(torch.randn(1,17,16)*torch.linspace(.2,8,17).view(1,17,1))[...,::2]
    captured={}
    matmul=torch.matmul
    def capture(a,b,*args,**kwargs):
        result=matmul(a,b,*args,**kwargs)
        if a.shape==(1,2,17,192) and b.shape==(1,2,192,17):
            captured['q']=a.detach().clone();captured['k']=b.transpose(-1,-2).detach().clone()
            captured['logits']=result.detach().clone()*attn.softmax_scale
        return result
    class Model:
        training=False
        def __init__(self):
            self.config=config
            self.model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attn,input_layernorm=norm)])
        def __call__(self,**kwargs):
            mask=torch.full((17,17),float('-inf')).triu(1)[None,None]
            with patch.object(torch,'matmul',capture):
                _,captured['probs'],_=attn(norm(hidden),attention_mask=mask,
                    position_ids=kwargs.get('position_ids',torch.arange(17)[None]),output_attentions=True,use_cache=False)
            return SimpleNamespace(hidden_states=(hidden,))
    with torch.no_grad():
        qn,qr,kn,kr,scales=probe['get_mla_qk'](Model(),torch.zeros(1,17,dtype=torch.long))
        query=torch.cat((qn[0],qr[0]),dim=-1).transpose(0,1)[None]
        key=torch.cat((kn[0],kr[0].expand(-1,2,-1)),dim=-1).transpose(0,1)[None]
        logits=torch.matmul(query,key.transpose(-1,-2))*scales[0]
        errors={'q':float((query-captured['q']).abs().max()),
                'k':float((key-captured['k']).abs().max()),
                'logits':float((logits-captured['logits']).abs().max()),'probabilities':0.}
        for i in range(17):
            got=probe['head_mean_probs'](dict(nope=qn[0][i],rope=qr[0][i]),kn[0][:i+1],kr[0][:i+1],scales[0])
            ref=captured['probs'][0,:,i,:i+1].mean(0)
            row_error=finite_probability_error(got,ref)
            errors['probabilities']=max(errors['probabilities'],row_error)
    fixture=hashlib.sha256()
    for value in (hidden,norm.weight,*attn.state_dict().values()):fixture.update(value.detach().contiguous().numpy().tobytes())
    passed=all(math.isfinite(x) and x<=1e-6 for x in errors.values())
    print(json.dumps(dict(scope='random weights; actual eager Attention.forward; CPU fp32, not full checkpoint',
        torch=torch.__version__,upstream_sha256=official_hash,probe_sha256=probe_hash,
        fixture_sha256=fixture.hexdigest(),hidden_noncontiguous=not hidden.is_contiguous(),
        errors=errors,parity=passed,gpu_used=False),indent=2))
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
