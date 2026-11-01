import pytest
import torch
from einops import repeat, einsum


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(autouse=True)
def enable_prototype(monkeypatch):
    monkeypatch.setenv("NFA_FINE_TOPK_TRITON", "1")


def _make_inputs():
    torch.manual_seed(0)
    S = 512
    Hkv = 4
    G = 4
    H = Hkv * G
    D = 128
    R = 256
    kq = torch.randn(S, Hkv, D, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(H, D, device="cuda", dtype=torch.float32) * 0.05
    cl = torch.tensor([180, 150, 200, 120], device="cuda", dtype=torch.int64)
    cand = torch.randint(0, S, (Hkv, R), device="cuda", dtype=torch.int64)
    keep = torch.zeros(Hkv, R, device="cuda", dtype=torch.bool)
    for h in range(Hkv):
        keep[h, : cl[h].item()] = True
        if cl[h].item() > 5:
            cand[h, cl[h].item() - 2] = cand[h, 0]
    cand = torch.where(keep, cand, S - 1)
    return S, Hkv, G, H, D, R, kq, q, cand, keep, cl


def test_fused_probabilities_and_local_topk_match_reference():
    from sparse_attn.ops.nfa_fine_topk_triton import (
        fused_fine_probabilities,
        topk_selection_from_probabilities,
    )

    S, Hkv, G, H, D, R, kq, q, cand, keep, cl = _make_inputs()
    probs, unique = fused_fine_probabilities(
        kq, q, cand, keep, cl, S, G)
    torch.cuda.synchronize()

    kv = torch.arange(Hkv, device="cuda").view(-1, 1)
    gathered = repeat(kq[cand, kv, :], "kv r d -> (kv g) r d", g=G).float()
    scores = einsum(q, gathered, "h d, h r d -> h r")
    invalid = (torch.arange(R, device="cuda")[None] >= cl[:, None])
    invalid = invalid.unsqueeze(1).expand(Hkv, G, R).reshape(H, R)
    scores = scores.masked_fill(invalid, float("-inf"))
    reference = (
        torch.softmax(scores, dim=-1).nan_to_num(0).bfloat16()
        .view(Hkv, G, R).mean(dim=1).bfloat16()
    )
    torch.testing.assert_close(probs, reference, atol=0, rtol=0)

    owner = torch.full((Hkv, S), -1, dtype=torch.int32, device="cuda")
    for h in range(Hkv):
        for r in range(R):
            if keep[h, r] and 0 <= cand[h, r] < S:
                owner[h, cand[h, r]] = r
    unique_ref = torch.zeros_like(keep)
    for h in range(Hkv):
        for r in range(R):
            unique_ref[h, r] = (
                keep[h, r] and 0 <= cand[h, r] < S
                and owner[h, cand[h, r]] == r
            )
    assert torch.equal(unique, unique_ref)

    quota = 128
    logical, counts = topk_selection_from_probabilities(
        cand, probs, unique, 0, S, quota, S, quota)
    torch.cuda.synchronize()
    positive_ids = torch.where(unique_ref, cand, S)
    positive_p = torch.where(unique_ref, reference, 0)
    bits = positive_p.contiguous().view(torch.int16).to(torch.int32)
    key = (bits << 16) + (S - positive_ids.to(torch.int32))
    take = min(quota, int((positive_ids < S).sum(-1).max()))
    chosen = positive_ids.gather(-1, torch.topk(key, take, dim=-1).indices)
    chosen = chosen.sort(-1).values
    ref_ids = torch.where(chosen < S, chosen, 0)
    ref_counts = (chosen < S).sum(-1, dtype=torch.int32)
    assert torch.equal(logical, ref_ids)
    assert torch.equal(counts, ref_counts)


def test_logical_to_physical_pow2_and_nonpow2():
    from sparse_attn.ops.nfa_fine_topk_triton import logical_to_physical

    ids = torch.tensor(
        [
            [0, 1, 63, 64, 127, -1, 999, 128, 0, 1, 2, 3, 4, 5, 6, 7],
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160],
        ],
        dtype=torch.int64, device="cuda")
    block_table = torch.tensor(
        [[100, 101, 102, 103, 104, 105, 106, 107]],
        dtype=torch.int32, device="cuda")

    for page in (64, 60):
        out = logical_to_physical(ids, block_table, page)
        torch.cuda.synchronize()
        ref = torch.empty_like(out)
        for h in range(ids.shape[0]):
            for k in range(ids.shape[1]):
                x = int(ids[h, k])
                if x < 0:
                    ref[h, k] = -1
                else:
                    pg = min(x // page, block_table.shape[1] - 1)
                    ref[h, k] = int(block_table[0, pg]) * page + x % page
        assert torch.equal(out, ref)
