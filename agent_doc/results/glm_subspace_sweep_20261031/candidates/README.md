# Extended 32D candidate search — same-host 33.253.189.125

- Script: `sweep_candidates.py`, SHA256 `eb9e5aa2e4c1737293601290fda43fd8f52567035dfac6c1a320025ac92714ad`
- Source commit: `0a72672f3aa3d37f12545eeb8f397bbcc4c91d0b`
- Scope: first 10 RULER MultiKey2 64K rows, dense vs selected, fixed budgets; descriptive diagnostic only.
- Independent review: `validation/independent_review_candidates.json`.

Selected-arm answer hits:

| Candidate | Known 0–2 | Heldout 3–9 | Total |
|---|---:|---:|---:|
| `nope_last32` | 3/3 | 6/7 | 9/10 |
| `nope_even32` | 2/3 | 6/7 | 8/10 |
| `nope_mid32_64` | 3/3 | 5/7 | 8/10 |
| `mixed_nope16_rope_low8_high8` | 3/3 | 5/7 | 8/10 |
| `nope_mid32_96` | 2/3 | 5/7 | 7/10 |
| `nope_mid32_128` | 3/3 | 4/7 | 7/10 |
| `nope_first16_last16` | 3/3 | 4/7 | 7/10 |
| `nope_odd32` | 1/3 | 6/7 | 7/10 |
| `mixed_nope24_rope_low8` | 2/3 | 5/7 | 7/10 |
| `nope_first16_mid16` | 1/3 | 4/7 | 5/10 |
| `mixed_nope16_rope_high16` | 0/3 | 2/7 | 2/10 |
| `rope_high32` | 0/3 | 0/7 | 0/10 |
| `rope_mid32` | 0/3 | 0/7 | 0/10 |
| `rope_even32` | 0/3 | 0/7 | 0/10 |
| `tail32` | 0/3 | 0/7 | 0/10 |

Dense is 10/10 for every candidate and identical across candidates.

## Conclusion

No tested 32D candidate is accuracy-no-drop on these 10 fixed cases. Best is `nope_last32` at 9/10. Full task-dataset dispatch remains blocked by the user’s “no accuracy drop 32D” condition; next step is integration/debugging of the selected sparse consumer rather than blindly expanding the benchmark matrix.
