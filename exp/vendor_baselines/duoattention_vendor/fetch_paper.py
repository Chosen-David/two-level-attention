"""Fetch the fixed preprint into ignored local cache; verify before atomic save."""
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    info = json.loads((root / "PAPER.json").read_text())
    dest = root / ".cache" / "DuoAttention-2410.10819v1.pdf"
    if dest.exists():
        if hashlib.sha256(dest.read_bytes()).hexdigest() != info["sha256"]:
            raise SystemExit("Existing PDF differs; refusing overwrite")
        print(dest)
        return
    with urllib.request.urlopen(info["pdf_url"], timeout=60) as response:
        data = response.read(32 * 1024 * 1024 + 1)
    if len(data) > 32 * 1024 * 1024 or not data.startswith(b"%PDF-"):
        raise SystemExit("Invalid or oversized PDF")
    if hashlib.sha256(data).hexdigest() != info["sha256"]:
        raise SystemExit("PDF hash mismatch; no file written")
    dest.parent.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        # Hard-link fails if another writer has created the destination.
        os.link(tmp, dest)
    finally:
        os.unlink(tmp)
    print(dest)


if __name__ == "__main__":
    main()
