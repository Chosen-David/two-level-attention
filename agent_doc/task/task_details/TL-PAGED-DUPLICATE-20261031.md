# TL-PAGED-DUPLICATE-20261031

User-requested current-main audit; base dbbba689e19d0d1507d2005a7a4fb4227d281559.

write_decode validates all rows before writing, but duplicate active request IDs validate against the same original length and physical slot. Each duplicate overwrites the same token slot, then each increments seq_lens. Example length5, request_ids=[0,0]: only slot5 written, reported length7. This can expose an unwritten slot and misalign future positions. The cache API currently permits the input; this is not evidence the current B1 model runner triggers it.

Fix: reject duplicate active IDs before tensor admission, scalar slot reads, backups or writes. Repeated inactive padding IDs remain permitted. O(B) host validation, no GPU launch or synchronization added. No numerical behavior change for valid distinct requests.

Validation: actual main write_decode AST executed with NumPy tensor facade; three duplicate cases reproduce length corruption before fix and reject with zero copies after. Three distinct/reordered/inactive controls pass. This is host control-flow evidence, not native Torch/CUDA validation. Added native regression including CUDA when available, unchanged lengths/KV/write_bytes after rejection and valid retry with repeated -1 rows. Torch/Triton/pytest/GPU unavailable here; native tests pending.

Run: PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_paged_region_batch_cache.py tests/test_paged_region_cache.py

No GPU speedup claim. PR10 head2449f56983d7e97c5fe1c4df300c2b50b0c16b81 predates this cache API; no same-topic open PR. Preserve upstream request-metadata additions. Production mutation limited to admission; no main push or merge.

Next: native tests, then duplicate admission for future fused writers; allocation rollback, metadata lengths/block-table phase contract and pointer-table lifetime require separate coverage.
