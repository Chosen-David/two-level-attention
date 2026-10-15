"""CPU diagnostic of the E141 probe, not a GPU/model-quality benchmark.

Pass a locally downloaded, trusted official modeling_deepseek.py. Only the
RMSNorm and rotary definitions are extracted; no model weights are downloaded.
Exit 1 means Q/K parity failed, including when the known defect reproduces.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def definitions(path, names, namespace):
    source = path.read_bytes()
    tree = ast.parse(source, filename=str(path))
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f"missing definitions in {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return hashlib.sha256(source).hexdigest()


class FixtureModel:
    def __init__(self, hidden, attention, norm):
        self.hidden = hidden
        self.config = SimpleNamespace(num_attention_heads=2)
        self.model = SimpleNamespace(layers=[SimpleNamespace(
            self_attn=attention, input_layernorm=norm)])

    def __call__(self, **kwargs):
        return SimpleNamespace(hidden_states=(self.hidden,))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-source", type=Path, required=True)
    parser.add_argument("--probe-source", type=Path, default=Path(__file__).resolve().parents[2]
                        / "agent_doc/guide/paper_figs/trace/e141_mla_nope_probe.py")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261015)
    upstream = {"torch": torch, "nn": torch.nn}
    official_hash = definitions(args.upstream_source,
                               {"DeepseekV2RMSNorm", "rotate_half", "apply_rotary_pos_emb"}, upstream)
    probe = {"torch": torch, "DEV": "cpu", "LAYERS": [0], "NOPE_D": 128, "ROPE_D": 64}
    probe_hash = definitions(args.probe_source,
                            {"get_mla_qk", "rotate_half", "head_mean_probs"}, probe)
    norm_type = upstream["DeepseekV2RMSNorm"]
    hidden = torch.randn(1, 5, 8) * torch.tensor([1., 3., 10., 2., 7.]).view(1, 5, 1)
    norm = norm_type(8)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(.5, 1.5, 8))
    angle = torch.arange(5).float()[:, None] * torch.linspace(.07, 1.13, 32)[None, :]
    angle = torch.cat((angle, angle), dim=-1)
    cos, sin = angle.cos(), angle.sin()
    attn = SimpleNamespace(q_proj=torch.nn.Linear(8, 2 * 192, bias=False),
                           kv_a_proj_with_mqa=torch.nn.Linear(8, 576, bias=False),
                           kv_a_layernorm=norm_type(512),
                           kv_b_proj=torch.nn.Linear(512, 2 * 256, bias=False),
                           rotary_emb=lambda value, seq_len: (cos[:seq_len], sin[:seq_len]),
                           softmax_scale=192 ** -.5)
    fixture = hashlib.sha256()
    for tensor in (hidden, norm.weight, attn.q_proj.weight, attn.kv_a_proj_with_mqa.weight,
                   attn.kv_a_layernorm.weight, attn.kv_b_proj.weight, cos, sin):
        fixture.update(tensor.detach().contiguous().numpy().tobytes())
    with torch.no_grad():
        normalized = norm(hidden)
        q = attn.q_proj(normalized).view(1, 5, 2, 192)
        packed = attn.kv_a_proj_with_mqa(normalized)
        kn = attn.kv_b_proj(attn.kv_a_layernorm(packed[..., :512])).view(1, 5, 2, 256)[0, ..., :128]
        qr, kr = upstream["apply_rotary_pos_emb"](
            q[..., 128:].transpose(1, 2), packed[..., 512:].unsqueeze(1),
            cos, sin, torch.arange(5).unsqueeze(0))
        qr, kr = qr.transpose(1, 2)[0], kr.transpose(1, 2)[0]
        ids = torch.zeros(1, 5, dtype=torch.long)
        raw = probe["get_mla_qk"](FixtureModel(hidden, attn, norm), ids)
        # Feed already-normalized states to isolate the RoPE defect from RMSNorm.
        rope_only = probe["get_mla_qk"](FixtureModel(normalized, attn, torch.nn.Identity()), ids)
        reference_nope = q[0, ..., :128]
        actual = probe["head_mean_probs"](
            {"nope": rope_only[0][0][-1], "rope": rope_only[1][0][-1]},
            rope_only[2][0], rope_only[3][0], attn.softmax_scale)
        expected = probe["head_mean_probs"](
            {"nope": reference_nope[-1], "rope": qr[-1]}, kn, kr, attn.softmax_scale)
        errors = {
            "norm_q_nope_max_abs": (raw[0][0] - reference_nope).abs().max().item(),
            "rope_only_q_max_abs": (rope_only[1][0] - qr).abs().max().item(),
            "rope_only_k_max_abs": (rope_only[3][0] - kr).abs().max().item(),
            "rope_only_probability_max_abs": (actual - expected).abs().max().item(),
        }
    result = {"torch": torch.__version__, "device": "cpu", "dtype": "float32",
              "shape": {"batch": 1, "tokens": 5, "hidden": 8, "heads": 2, "nope": 128, "rope": 64},
              "probe_sha256": probe_hash, "upstream_sha256": official_hash,
              "fixture_sha256": fixture.hexdigest(), "errors": errors,
              "parity": all(value <= 1e-6 for value in errors.values())}
    print(json.dumps(result, indent=2))
    return 0 if result["parity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
