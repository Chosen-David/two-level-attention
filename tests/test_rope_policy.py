import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from benchmark.rope_policy import configure_rope,verify_runtime_rope
from benchmark.RULER.yarn_receipt import resolve_yarn_config

MODEL='/home/wangyuanshuo02/Qwen3-8B'


@pytest.mark.parametrize('length,factor',[(65536,2.),(131072,4.)])
def test_runtime_frequencies_and_position_capacity(length,factor):
    config=AutoConfig.from_pretrained(MODEL)
    snapshot=config.max_position_embeddings
    record=configure_rope(config,length,64,yarn=True,factor=factor)
    assert record['snapshot_max_position_embeddings']==snapshot
    assert config.max_position_embeddings>=length+64
    assert config.rope_scaling['original_max_position_embeddings']==32768
    verified=verify_runtime_rope(config)
    assert verified['requested_factor']==factor and verified['status']=='verified'
    rotary=Qwen3RotaryEmbedding(config=config,device='cpu')
    model=SimpleNamespace(config=config,model=SimpleNamespace(rotary_emb=rotary))
    assert verify_runtime_rope(model)['inv_freq']==verified['inv_freq']
    # A runtime factor override cannot pass just because JSON still says2/4.
    rotary.inv_freq=rotary.inv_freq*.9
    with pytest.raises(ValueError,match='framework RoPE'):verify_runtime_rope(model)


def test_unscaled_snapshot_unchanged():
    config=AutoConfig.from_pretrained(MODEL);before=copy.deepcopy(config.to_dict())
    configure_rope(config,32768)
    assert config.to_dict()==before
    assert verify_runtime_rope(config)['attention_scaling']==1


@pytest.mark.parametrize('length,yarn,factor',[(32768,True,2),(65536,False,None),(65536,True,4),(131072,True,2)])
def test_rejects_nonmain_qwen_protocol(length,yarn,factor):
    with pytest.raises(ValueError):configure_rope(AutoConfig.from_pretrained(MODEL),length,yarn=yarn,factor=factor)


def test_glm_keeps_published_config_and_rejects_qwen_override():
    class Config:
        model_type='glm4_moe_lite';max_position_embeddings=202752;rope_scaling=None
        def to_dict(self):return dict(vars(type(self)))
    c=Config();before=vars(c).copy()
    assert configure_rope(c,131072)['snapshot_preserved']
    assert vars(c)==before
    with pytest.raises(ValueError):configure_rope(c,65536,yarn=True,factor=2)


def test_resolver_separates_original_from_runtime():
    factor,rs=resolve_yarn_config(True,None,65536,{65536:2.},40960)
    assert factor==2 and rs['original_max_position_embeddings']==32768


def corrected_receipt():
    from benchmark.RULER.yarn_receipt import validate_corrected_rope_policy
    c=AutoConfig.from_pretrained(MODEL)
    policy=configure_rope(c,65536,yarn=True)
    record=dict(rope_policy=policy,runtime_rope_verification=verify_runtime_rope(c),
                context_length=65536,rope_scaling=c.rope_scaling,generation_params={'max_gen':64})
    assert validate_corrected_rope_policy(record) is None
    return record


def test_consumer_rejects_false_or_missing_runtime_proof():
    from benchmark.RULER.yarn_receipt import validate_corrected_rope_policy,effective_config_sha256
    record=corrected_receipt();before=effective_config_sha256(record)
    record['runtime_rope_verification']['inv_freq'][0]*=.9
    assert validate_corrected_rope_policy(record)
    record=corrected_receipt();record['rope_policy']['runtime_max_position_embeddings']+=1
    assert validate_corrected_rope_policy(record)
    assert before!=effective_config_sha256(record)
    assert validate_corrected_rope_policy({'rope_scaling':{'factor':2,'original_max_position_embeddings':40960}})


def test_main_cannot_bypass_rope_by_omitting_yarn_cli():
    from benchmark.RULER.score_ruler_formal import _resolve_yarn_identity
    args=SimpleNamespace(rope_policy='main',yarn=False,yarn_factor=None,model_path=MODEL)
    with pytest.raises(SystemExit):
        _resolve_yarn_identity(args,{'cells':{},'missing':['L65536/none/task']})
    with pytest.raises(SystemExit):
        _resolve_yarn_identity(args,{'cells':{'L65536/none/task':dict(yarn_enabled=False,context_length=65536,config_fingerprint={'model_path':MODEL})},'missing':[]})
