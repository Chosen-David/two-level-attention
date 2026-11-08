# GLM 4.7-Flash subspace sweep — same-host 33.253.189.125

- Run id: `glm_subspace_sweep_20261031`
- Host: `33.253.189.125` (8 H20-3e), same host for all eight variants.
- Source commit: `0a72672f3aa3d37f12545eeb8f397bbcc4c91d0b`
- Producer script SHA256: `cceabfcf82076008154caacb151721cd95e9b934d473236c9a4088b0cc969337`
- Dataset: RULER 65536 `niah_multikey_2.jsonl`, first 10 rows.
- Protocol: GLM chat template, thinking disabled; max64/min2; K1=128/K2=1024; gates off.
- Raw outputs: `raw125_full/` (new variants) + `raw125_prior/` (prior same-host variants).

Independent review: `validation/independent_review_same_host.json`.

## Selected-arm answer hits, recomputed independently

| Variant | Known 0–2 | Heldout 3–9 | Total |
|---|---:|---:|---:|
| `rope_low32` | 0/3 | 0/7 | 0/10 |
| `rope64` | 0/3 | 0/7 | 0/10 |
| `nope32` | 2/3 | 4/7 | 6/10 |
| `nope64` | 3/3 | 6/7 | 9/10 |
| `mixed32` | 2/3 | 3/7 | 5/10 |
| `mixed64` | 3/3 | 5/7 | 8/10 |
| `full256` | 3/3 | 7/7 | 10/10 |
| `alltoken_control` | 3/3 | 7/7 | 10/10 |

Dense arm is 10/10 for every variant; dense outputs are identical across variants. `alltoken_control` selected outputs exactly match dense on all 10 rows.

## Conclusion

This is a fixed-10-case recorded-output diagnostic, not a benchmark/paper-method acceptance. No tested 32-dimensional subspace matches dense on all 10 cases; the closest tested 32D variants are `nope32` (6/10) and `mixed32` (5/10). RoPE-only 32/64 variants collapse to 0/10. Therefore full task-dataset dispatch is not yet authorized by the user's “no accuracy drop 32D” condition.
