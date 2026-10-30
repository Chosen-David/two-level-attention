"""Verify unchanged vendored sources; no imports, GPU or network."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "UPSTREAM.json").read_text())
expected = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
actual = {str(p.relative_to(root / "upstream")): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in (root / "upstream").rglob("*") if p.is_file() and "__pycache__" not in p.parts and not any(part.endswith(".egg-info") for part in p.parts)}
if actual != expected:
    raise SystemExit("Source inventory/hash mismatch")
print(f"Verified {len(actual)} upstream files at {manifest['commit']}")
