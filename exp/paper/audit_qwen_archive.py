"""Read-only audit of the fixed Qwen Observation archive cohort; never collect.

Checksums establish bytes, not model/input provenance or scientific acceptance.
Exit 2 for missing files, 3 for identity/schema errors, 0 for byte/schema checks.
Even exit 0 leaves provenance unresolved. Requires torch only for present PTs.
"""
import argparse
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "exp/trace_archives/trace_dumps_md5/md5_qwen3-8b.txt"
NAMES = tuple(f"lb_{task}_{sample}/{name}" for task in ("hotpotqa", "musique")
              for sample in (0, 1) for name in ("meta.json", "layer08.pt", "layer24.pt"))


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_checksums(index):
    result = {}
    for line in index.read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        name = name.removeprefix("./")
        if name in NAMES:
            if name in result or len(checksum) != 32:
                raise ValueError("ambiguous or invalid archive checksum entry")
            int(checksum, 16)
            result[name] = checksum
    if set(result) != set(NAMES):
        raise ValueError("fixed cohort incomplete in checksum index")
    return result


def tensor_schema(path, meta):
    import torch
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or not {"k", "q", "qpos", "S"} <= data.keys():
        raise ValueError("missing Q/K archive fields")
    k, q, pos, length = (data[x] for x in ("k", "q", "qpos", "S"))
    if type(length) is not int or length < 16 or length > 32768 or length != meta["S"]:
        raise ValueError("invalid or mismatched sequence length")
    if (not all(isinstance(x, torch.Tensor) for x in (k, q, pos))
            or k.ndim != 3 or q.ndim != 3 or pos.ndim != 1
            or k.shape[0] != length or q.shape[0] != pos.numel()
            or k.shape[-1] != 128 or q.shape[-1] != 128
            or k.shape[1] < 1 or q.shape[1] < k.shape[1]
            or q.shape[1] % k.shape[1] or pos.dtype != torch.int64
            or not k.is_floating_point() or not q.is_floating_point()
            or not bool(torch.isfinite(k).all()) or not bool(torch.isfinite(q).all())):
        raise ValueError("invalid Q/K shape, dtype, GQA relation or finite values")
    anchors = [int(length * (i + 0.5) / 16) for i in range(16)]
    expected = sorted(set(range(length - min(256, length), length)) | set(anchors))
    if pos.tolist() != expected:
        raise ValueError("stored query positions do not match frozen collector")
    return {"S": length, "k_shape": list(k.shape), "q_shape": list(q.shape),
            "k_dtype": str(k.dtype), "q_dtype": str(q.dtype),
            "anchor_positions": anchors, "stored_qpos": expected,
            "selected_q_rows": [expected.index(p) for p in anchors]}


def audit(root, index=INDEX):
    checksums = expected_checksums(index)
    rows, metadata = [], {}
    for name in NAMES:
        path = root / name
        row = {"relative_path": name, "expected_md5": checksums[name]}
        rows.append(row)
        if not path.is_file():
            row["status"] = "missing"
            continue
        row.update(size_bytes=path.stat().st_size, md5=digest(path, "md5"),
                   sha256=digest(path, "sha256"))
        if row["md5"] != checksums[name]:
            row["status"] = "checksum-mismatch"
            continue
        try:
            if path.suffix == ".json":
                meta = json.loads(path.read_text())
                if (type(meta.get("S")) is not int or not 16 <= meta["S"] <= 32768
                        or meta.get("prompt") != path.parent.name
                        or type(meta.get("n_layers")) is not int or meta["n_layers"] <= 24):
                    raise ValueError("invalid source metadata")
                metadata[path.parent.name] = meta
                row["schema"] = {k: meta[k] for k in ("S", "n_layers", "prompt")}
            else:
                if path.parent.name not in metadata:
                    raise ValueError("verified metadata required before tensors")
                row["schema"] = tensor_schema(path, metadata[path.parent.name])
            row["status"] = "byte-and-schema-verified"
        except Exception as exc:
            row["status"] = "schema-unverified"
            row["error_type"] = type(exc).__name__
    bad = any(r["status"] not in ("missing", "byte-and-schema-verified") for r in rows)
    missing = any(r["status"] == "missing" for r in rows)
    return {"cohort": "qwen3-8b-archive-hotpotqa-musique-0-1-layers8-24",
            "phase": "prefill-sampled-anchors", "root": str(root),
            "index_sha256": digest(index, "sha256"), "files": rows,
            "provenance_status": "unresolved", "paper_ready": False,
            "missing_provenance": ["checkpoint/config", "tokenizer", "dataset rows/document overlap",
                                   "exact input IDs/template", "installed HF SDPA implementation/scaling"],
            "exit_code": 3 if bad else 2 if missing else 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / ".archive/trace-dumps/qwen3-8b")
    args = parser.parse_args()
    result = audit(args.root)
    print(json.dumps(result, indent=2))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
