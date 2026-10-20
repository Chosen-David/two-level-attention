# Independent review: frozen vendor manifest identity

Reviewed the bounded `sparse_attn/info.py` change against `f622c907` and the new host-only regression tests. No production edits or pushes performed by this reviewer.

## Conclusion

No blocking finding for the stated post-freeze source replacement/deletion defect. Manifest construction now consumes the same complete SHA256 identity as the frozen MInference pattern or staged FASA agreement. Capturing the resolved basename prevents a later source-to-symlink replacement from changing the manifest label. Unfrozen behavior remains on the existing file-identity path. The additional dataclass fields have trailing defaults, preserving existing constructor argument compatibility.

## Independently checked

- New regression suite: 7/7 methods pass normally and with `python -O`; production `info.py` is executed with the Torch import replaced by a shim that rejects tensor access.
- Same suite against base `f622c907`: 4 failing assertions (including both vendor symlink subtests) and 2 errors, across 7 test methods; unchanged-source and unfrozen-source controls pass.
- 17 base/current exact manifest comparisons passed: all registered vendor methods, nonvendor defaults, and unchanged frozen MInference/FASA content.
- A separately constructed original-symlink retarget probe retained both the original target basename and digest after retargeting to a differently named, different-content file.
- 32 fixed-quota configurations (four geometries crossed with D-prime on/off, boundary bypass on/off, and L2 quantization on/off) retained byte-identical manifest JSON and complete method names versus base.
- `git diff --check` passed. The reviewed production diff contains only host metadata consumption and snapshot identity capture; no fixed-quota selection, arithmetic, kernels, or timing paths changed.

## Scope and residual caveat

These are host metadata/source checks, not Torch integration, tensor selection, GPU correctness, model-quality, or performance evidence. The basename is resolved after reading source bytes, so a concurrent symlink retarget during the freeze operation itself can still mix an old-content digest with a newer target basename. That is outside the bounded post-freeze defect reviewed here; the digest and runtime content remain derived from the same bytes. Existing snapshots constructed with the new field left empty use the documented realpath-basename fallback.
