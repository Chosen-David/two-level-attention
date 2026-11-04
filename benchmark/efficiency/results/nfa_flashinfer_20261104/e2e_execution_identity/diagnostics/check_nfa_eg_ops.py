import json
import sys
import torch

sys.path.insert(0, '/tmp/nfa-topk64-main-candidate')
from sparse_attn.ops.qwen3_rmsnorm_triton import (
    fused_rmsnorm_decode, fused_rmsnorm_decode_single)
from sparse_attn.ops.qwen3_mlp_silu_triton import fused_silu_mul_decode


def rms_reference(x, weight, eps):
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(-1, keepdim=True)
    return weight * (x32 * torch.rsqrt(variance + eps)).to(x.dtype)


def compare(fn, ref):
    mismatches = int((fn != ref).sum())
    return {
        'equal': torch.equal(fn, ref),
        'mismatches': mismatches,
        'numel': ref.numel(),
        'max_abs_diff': float((fn.float() - ref.float()).abs().max()),
    }


torch.manual_seed(20261105)
results = {}
with torch.cuda.device(0):
    for shape, name in (((1, 1, 32, 128), 'q_norm'),
                        ((1, 1, 8, 128), 'k_norm'),
                        ((1, 1, 4096), 'hidden_norm')):
        for index in range(4):
            x = torch.randn(shape, device='cuda', dtype=torch.bfloat16).contiguous()
            weight = torch.randn(shape[-1], device='cuda', dtype=torch.bfloat16).contiguous()
            reference = rms_reference(x, weight, 1e-6)
            results[f'{name}_{index}_three_kernel'] = compare(
                fused_rmsnorm_decode(x, weight, 1e-6), reference)
            results[f'{name}_{index}_single_kernel'] = compare(
                fused_rmsnorm_decode_single(x, weight, 1e-6), reference)
    for index in range(4):
        gate = (torch.randn(1, 1, 12288, device='cuda') * 8).to(torch.bfloat16).contiguous()
        up = torch.randn(1, 1, 12288, device='cuda', dtype=torch.bfloat16).contiguous()
        reference = torch.nn.functional.silu(gate) * up
        results[f'silu_mul_{index}'] = compare(
            fused_silu_mul_decode(gate, up), reference)

with open('/tmp/nfa_eg_diag_ops.json', 'w') as handle:
    json.dump(results, handle, indent=2)
for key, value in results.items():
    print(key, value)
