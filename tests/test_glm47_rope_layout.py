"""Check selected frequencies against HF's actual GLM rotary output layout."""
from pathlib import Path
import runpy

import torch
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
    apply_rotary_pos_emb_interleave,
)


def test_tail_selects_both_components_of_low_frequency_pairs():
    helpers = runpy.run_path(str(Path(__file__).parents[1] /
                                "exp/bench/test_glm47_d256_gates.py"))
    indexer = helpers["make_indexer"](256)
    # HF consumes interleaved pairs but returns cat(real, imag). Tag each
    # frequency independently so the oracle does not duplicate index offsets.
    q = torch.zeros(1, 1, 1, 64)
    q[..., ::2] = 1
    angles = torch.linspace(0.05, 1.55, 32)
    cos = angles.cos().repeat(2).view(1, 1, 64)
    sin = angles.sin().repeat(2).view(1, 1, 64)
    rotated, _ = apply_rotary_pos_emb_interleave(q, q, cos, sin)
    key = torch.cat((torch.full((1, 1, 1, 192), -99.), rotated), -1)
    expected = torch.cat((angles[-16:].cos(), angles[-16:].sin()))
    for indices in (indexer._subspace_indices(torch.device("cpu")),
                    indexer._qat_indices(torch.device("cpu"))):
        selected = key[0, 0, 0, indices]
        torch.testing.assert_close(selected, expected)
        torch.testing.assert_close(selected[:16].square() + selected[16:].square(),
                                   torch.ones(16))
