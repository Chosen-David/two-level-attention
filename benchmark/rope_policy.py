"""Versioned evaluation RoPE policy and runtime frequency verification.

Model position capacity and YaRN's training length are different quantities.
This module never rewrites a GLM snapshot's published RoPE settings.
"""
import copy
import hashlib
import math

QWEN_ORIGINAL_CONTEXT = 32768
POLICY = 'near-far-rope-32768-v1'


def configure_rope(config, context_length, max_new_tokens=64, *, yarn=False, factor=None):
    """Configure before model construction; preserve non-Qwen snapshot settings."""
    if context_length <= 0 or max_new_tokens < 0:
        raise ValueError('invalid input/output capacity')
    model_type = config.model_type
    original_runtime = config.max_position_embeddings
    if model_type != 'qwen3':
        if yarn or factor is not None:
            raise ValueError('Qwen YaRN overrides are not allowed for other models')
        if context_length + max_new_tokens > original_runtime:
            raise ValueError('input plus output exceeds published position capacity')
        return dict(policy=POLICY,model_type=model_type,snapshot_preserved=True,
                    snapshot_max_position_embeddings=original_runtime,
                    runtime_max_position_embeddings=original_runtime,
                    rope_scaling=copy.deepcopy(getattr(config,'rope_scaling',None)),
                    rope_parameters=copy.deepcopy(getattr(config,'rope_parameters',None)))
    required = {65536:2.0,131072:4.0}.get(context_length)
    if context_length <= QWEN_ORIGINAL_CONTEXT:
        if yarn or factor is not None:
            raise ValueError('32K evaluation must not enable additional YaRN')
        existing = getattr(config,'rope_scaling',None) or getattr(config,'rope_parameters',None)
        if existing and existing.get('rope_type',existing.get('type','default'))!='default':
            raise ValueError('32K Qwen snapshot already has rope scaling; use unscaled snapshot')
        if context_length + max_new_tokens > original_runtime:
            raise ValueError('input plus output exceeds unscaled snapshot capacity')
        return dict(policy=POLICY,model_type=model_type,snapshot_preserved=True,
                    snapshot_max_position_embeddings=original_runtime,
                    runtime_max_position_embeddings=original_runtime,rope_scaling=None)
    if required is None or not yarn or (factor is not None and factor != required):
        raise ValueError('Qwen long-context main protocol requires 64K factor2 or128K factor4')
    theta=getattr(config,'rope_theta',None) or config.rope_parameters['rope_theta']
    config.rope_scaling = dict(rope_type='yarn',type='yarn',factor=required,
        original_max_position_embeddings=QWEN_ORIGINAL_CONTEXT,beta_fast=32,beta_slow=1)
    config.max_position_embeddings = max(context_length+max_new_tokens,original_runtime)
    # Newer Transformers moves these values into rope_parameters. Update only
    # Qwen, and verify the constructed rotary module rather than trusting this.
    if hasattr(config,'rope_parameters'):
        config.rope_parameters = dict(config.rope_scaling,rope_theta=theta)
    return dict(policy=POLICY,model_type=model_type,snapshot_preserved=False,
        snapshot_max_position_embeddings=original_runtime,
        runtime_max_position_embeddings=config.max_position_embeddings,
        original_max_position_embeddings=QWEN_ORIGINAL_CONTEXT,
        requested_factor=required,rope_scaling=copy.deepcopy(config.rope_scaling))


def reference_yarn(config, device='cpu'):
    """Explicit static-YaRN oracle, independent of framework factor resolution."""
    import torch
    params=getattr(config,'rope_scaling',None) or getattr(config,'rope_parameters',None)
    dim=int(getattr(config,'head_dim',config.hidden_size//config.num_attention_heads)*getattr(config,'partial_rotary_factor',1.0))
    base=getattr(config,'rope_theta',None) or params['rope_theta']
    factor=float(params['factor']);original=int(params['original_max_position_embeddings'])
    low=math.floor(dim*math.log(original/(32*2*math.pi))/(2*math.log(base)))
    high=math.ceil(dim*math.log(original/(2*math.pi))/(2*math.log(base)))
    low=max(low,0);high=min(high,dim-1)
    if low==high: high+=.001
    ramp=((torch.arange(dim//2,dtype=torch.float32,device=device)-low)/(high-low)).clamp(0,1)
    unscaled=1.0/(base**(torch.arange(0,dim,2,dtype=torch.float32,device=device)/dim))
    return unscaled*(1-ramp)+unscaled/factor*ramp,1+.1*math.log(factor)


def verify_runtime_rope(model_or_config):
    """Fail before inference if the actual Qwen rotary differs from target.

    Accepts a constructed model (checks its module) or a config (CPU preflight).
    Evidence contains actual frequencies, attention factor and a rotation probe.
    """
    import torch
    import transformers
    config=getattr(model_or_config,'config',model_or_config)
    if config.model_type!='qwen3':
        return dict(policy=POLICY,model_type=config.model_type,status='snapshot-preserved-no-qwen-override',
                    transformers=transformers.__version__)
    if hasattr(model_or_config,'model'):
        rotary=model_or_config.model.rotary_emb
    else:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
        rotary=Qwen3RotaryEmbedding(config=config,device='cpu')
    actual=rotary.inv_freq.detach().float().cpu()
    params=getattr(config,'rope_scaling',None) or getattr(config,'rope_parameters',None)
    is_yarn=params and params.get('rope_type',params.get('type'))=='yarn'
    if is_yarn:
        if params.get('original_max_position_embeddings')!=QWEN_ORIGINAL_CONTEXT:
            raise ValueError('Qwen evaluation YaRN original context must be32768')
        expected,scale=reference_yarn(config)
    else:
        dim=int(getattr(config,'head_dim',config.hidden_size//config.num_attention_heads)*getattr(config,'partial_rotary_factor',1.0))
        base=getattr(config,'rope_theta',None) or params['rope_theta']
        expected=1.0/(base**(torch.arange(0,dim,2,dtype=torch.float32)/dim));scale=1.
    actual_scale=float(rotary.attention_scaling)
    if actual.shape!=expected.shape or not torch.allclose(actual,expected,rtol=2e-6,atol=1e-10) or not math.isclose(actual_scale,scale,rel_tol=1e-7):
        raise ValueError('framework RoPE differs from explicit policy; possible factor override; use verified Transformers version')
    device=rotary.inv_freq.device
    positions=torch.tensor([[0,32767,65535,131071]],device=device)
    with torch.no_grad():
        cos,sin=rotary(torch.zeros(1,4,actual.numel()*2,device=device,dtype=torch.float32),positions)
    angles=positions.cpu().float().unsqueeze(-1)*expected
    angles=torch.cat([angles,angles],dim=-1)
    cos_error=float((cos.cpu()-angles.cos()*scale).abs().max())
    sin_error=float((sin.cpu()-angles.sin()*scale).abs().max())
    # Same float32 formula; tolerates implementation ordering without hiding
    # an effective factor mismatch (already checked directly above).
    if max(cos_error,sin_error)>5e-4:raise ValueError('runtime rotary result differs from reference')
    return dict(policy=POLICY,status='verified',transformers=transformers.__version__,
        runtime_max_position_embeddings=config.max_position_embeddings,
        original_max_position_embeddings=params.get('original_max_position_embeddings') if is_yarn else None,
        requested_factor=params['factor'] if is_yarn else None,attention_scaling=actual_scale,
        verification_source_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),
        framework_rope_source_sha256=hashlib.sha256(__import__('inspect').getsource(type(rotary)).encode()).hexdigest(),
        rotary_dim=actual.numel()*2,rope_theta=float(getattr(config,'rope_theta',None) or params['rope_theta']),
        inv_freq=actual.tolist(),inv_freq_sha256=hashlib.sha256(actual.numpy().tobytes()).hexdigest(),
        max_inv_freq_error=float((actual-expected).abs().max()),max_cos_error=cos_error,max_sin_error=sin_error)
