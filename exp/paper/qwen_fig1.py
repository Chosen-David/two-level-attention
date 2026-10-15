"""Fixed-cohort Qwen Fig1 CPU diagnostics, never scientific acceptance or plotting.

Default: audit only. --analyze consumes byte/schema-checked existing archives.
Model/input provenance remains unresolved, so paper_ready is always false.
"""
import argparse
import io
import json
import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exp.paper.audit_qwen_archive import audit



def bounds(position):
    if type(position) is not int or position < 0:
        raise ValueError("query position must be a nonnegative integer")
    length = position + 1
    lo, hi = min(128, length), max(min(128, length), length - 128)
    requested = max(64, int(.25 * (hi - lo))) if hi > lo else 0
    near = max(lo, min(hi, ((max(0, length - 128) - requested) // 64) * 64))
    return dict(length=length, mid_lo=lo, mid_hi=hi, near_lo=near,
                sink_hi=lo, swa_lo=max(0, length - 128))


def reference(q, k, position):
    """Post-RoPE Q[Hq,128], full K[S,Hkv,128]; FP32 causal per-head softmax."""
    bounds(position)
    q, k = np.asarray(q), np.asarray(k)
    if (q.ndim != 2 or k.ndim != 3 or q.shape[-1] != 128 or k.shape[-1] != 128
            or not q.shape[0] or not k.shape[1] or q.shape[0] % k.shape[1]
            or position >= k.shape[0]
            or not np.issubdtype(q.dtype, np.floating)
            or not np.issubdtype(k.dtype, np.floating)
            or not np.isfinite(q).all() or not np.isfinite(k).all()):
        raise ValueError("invalid finite D128 Q/K or GQA/causal geometry")
    q, k = q.astype(np.float32), k[:position + 1].astype(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        scores = np.einsum("hgd,thd->hgt", q.reshape(k.shape[1], -1, 128), k)
        scores *= np.float32(128 ** -.5)
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite FP32 logits")
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=-1, keepdims=True)
    return p.reshape(q.shape[0], -1), p.mean(axis=1)


def concentration(p, lo, hi):
    """Minimum token/page fractions covering 90% of a region's own mass."""
    p = np.asarray(p)
    if (p.ndim != 1 or not np.issubdtype(p.dtype, np.floating)
            or not np.isfinite(p).all() or (p < 0).any()
            or not np.isclose(p.sum(dtype=np.float64), 1, atol=1e-5, rtol=0)
            or type(lo) is not int or type(hi) is not int or not 0 <= lo <= hi <= len(p)):
        raise ValueError("invalid full-reference distribution or region")
    # Aggregate FP32 probabilities in float64; do not change the reference softmax.
    values = p[lo:hi].astype(np.float64)
    mass = float(values.sum())
    pages = list(range(lo // 64, (hi - 1) // 64 + 1)) if hi > lo else []
    counts = [min(hi, (page + 1) * 64) - max(lo, page * 64) for page in pages]
    result = dict(mass=mass, token_count=hi-lo, page_count=len(pages),
                  page_ids=pages, page_token_counts=counts,
                  token_count_90=None, page_count_90=None,
                  token_fraction_90=None, page_fraction_90=None,
                  undefined_reason="empty_region" if hi == lo else "zero_region_mass" if mass == 0 else None)
    if result["undefined_reason"]:
        return result
    def count90(weights):
        ordered = np.sort(weights)[::-1]
        return int(np.searchsorted(np.cumsum(ordered), .9 * mass, side="left")) + 1
    page_mass = np.array([p[max(lo, page*64):min(hi, (page+1)*64)].sum(dtype=np.float64) for page in pages])
    nt, npages = count90(values), count90(page_mass)
    result.update(token_count_90=nt, page_count_90=npages,
                  token_fraction_90=nt / len(values), page_fraction_90=npages / len(pages))
    return result


def query_rows(q, k, position):
    b = bounds(position)
    heads, groups = reference(q, k, position)
    group_size = len(heads) // len(groups)
    rows = []
    for level, probabilities in (("q_head", heads), ("kv_group_mean_probability", groups)):
        for head, p in enumerate(probabilities):
            sink = float(p[:b["sink_hi"]].sum(dtype=np.float64))
            swa = float(p[max(b["sink_hi"], b["swa_lo"]):].sum(dtype=np.float64))
            near = concentration(p, b["near_lo"], b["mid_hi"])
            far = concentration(p, b["mid_lo"], b["near_lo"])
            if not np.isclose(sink + swa + near["mass"] + far["mass"], 1, atol=1e-5, rtol=0):
                raise ValueError("protected/near/far mass conservation failed")
            # Keep only contributing bins; no absent-distance zero imputation.
            bins = []
            for region, lo, hi in (("near", b["near_lo"], b["mid_hi"]),
                                   ("far", b["mid_lo"], b["near_lo"])):
                ids = np.arange(lo, hi)
                bin_ids = (position - ids) // 64
                counts = np.bincount(bin_ids)
                masses = np.bincount(bin_ids, weights=p[ids].astype(np.float64))
                for index in np.flatnonzero(counts):
                    bins.append(dict(distance_lo=int(index)*64, distance_hi=(int(index)+1)*64,
                                     region=region, mass=float(masses[index]),
                                     token_count=int(counts[index]),
                                     density=float(masses[index]/counts[index])))
            rows.append(dict(query_position=position, level=level, head_index=head,
                             kv_group=head // group_size if level == "q_head" else head,
                             bounds=b, mass_sink=sink, mass_swa_excluding_sink=swa,
                             mass_protected_union=sink+swa, near=near, far=far,
                             distance_bins=bins))
    return rows


def analyze_archive(root, *, analyze=False):
    receipt = audit(root)
    result = dict(schema="qwen-fig1-diagnostic-v1", paper_ready=False,
                  status="blocked", phase="prefill-sampled-anchors", archive_audit=receipt,
                  protocol=dict(sink=128, swa=128, page_size=64, distance_bin_width=64,
                                alpha=.25, concentration_target=.9, scale=128 ** -.5,
                                probability="per-Q-head FP32 causal softmax, then KV-group mean",
                                independent_unit="source-sequence; document overlap unresolved",
                                source_sequences=4, layers=[8, 24], anchors_per_sequence=16),
                  scientific_claim=None, rows=[], distance_summary=[])
    if receipt["exit_code"] or not analyze:
        result["status"] = "blocked" if receipt["exit_code"] else "audit-only"
        return result
    import torch
    for entry in receipt["files"]:
        if not entry["relative_path"].endswith(".pt"):
            continue
        path = root / entry["relative_path"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("archive changed after audit")
        data = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        q, k = data["q"].float().numpy(), data["k"].float().numpy()
        for row_index, position in zip(entry["schema"]["selected_q_rows"], entry["schema"]["anchor_positions"]):
            for row in query_rows(q[row_index], k, position):
                row.update(source_sequence=path.parent.name, layer=int(path.stem[5:]),
                           trace_sha256=entry["sha256"])
                result["rows"].append(row)
    result["status"] = "diagnostic-unverified-provenance"
    result["distance_summary"] = summarize_distance(result["rows"])
    return result


def summarize_distance(rows):
    """Within-sequence/layer/head means over contributing queries only."""
    cells = {}
    names = ("source_sequence", "layer", "level", "head_index", "region", "distance_lo", "distance_hi")
    for row in rows:
        for b in row["distance_bins"]:
            key = tuple(row[n] for n in names[:4]) + tuple(b[n] for n in names[4:])
            cell = cells.setdefault(key, dict(positions=set(), mass=0., density=0.))
            if row["query_position"] in cell["positions"]:
                raise ValueError("duplicate query in distance summary")
            cell["positions"].add(row["query_position"])
            cell["mass"] += b["mass"]
            cell["density"] += b["density"]
    result = []
    for key, cell in sorted(cells.items()):
        n = len(cell["positions"])
        result.append(dict(zip(names, key), contributing_query_count=n,
                           query_positions=sorted(cell["positions"]),
                           mean_mass=cell["mass"]/n, mean_per_token_density=cell["density"]/n))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / ".archive/trace-dumps/qwen3-8b")
    parser.add_argument("--analyze", action="store_true", help="CPU Q/K diagnostics only; not identity acceptance")
    parser.add_argument("--out", type=Path, required=True, help="fresh JSON path")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists; choose a fresh path")
    result = analyze_archive(args.root, analyze=args.analyze)
    with args.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    return result["archive_audit"]["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
