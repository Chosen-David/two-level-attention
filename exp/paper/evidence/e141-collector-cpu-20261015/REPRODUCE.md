# Reproduce the numerical counterexamples

Use the existing CPU torch environment and a local source file. Do not download
weights or run the collector's --execute. The tested alternative source is bound
by `identity-and-history.json`; the official default hash deliberately rejects it.
`runs.json` records ephemeral file paths; recreate those variants as follows.

1. Historical probe: `git show 13be23cb1da409a4a63c0d009af34c8fbf167f95:agent_doc/guide/paper_figs/trace/e141_mla_nope_probe.py`
   saved to a temporary `.py` file, without executing its main function.
2. Missing norm variant: copy the repaired probe; replace exactly
   `hs = model.model.layers[L].input_layernorm(out.hidden_states[L])`
   with `hs = out.hidden_states[L]`.
3. Missing layout variant: copy the repaired probe and remove exactly these lines:

```python
        q_pe = q_pe.reshape(T, H, ROPE_D // 2, 2).transpose(-1, -2).reshape(T, H, ROPE_D)
        k_pe = k_pe.reshape(T, 1, ROPE_D // 2, 2).transpose(-1, -2).reshape(T, 1, ROPE_D)
```

Pass each temporary path using `--probe-source` to
`exp/trace/verify_e141_native_forward.py --upstream-source SOURCE --require-sha256 HASH`.
Use normal Python and `python -O`. For the tested alternative file, HASH is
`040246d626f2bb94075177c9cc17999fc1641a96bdb4eaa45a9efd3cce4df0aa`.
All three variants must print `parity: false` and exit 1. The repaired probe must
print `parity: true` and exit 0. Without the explicit alternative hash, the
native verifier must reject this alternative source before executing definitions.

The unmodified reused `verify_e141_qk_reconstruction.py` also accepts
`--upstream-source SOURCE --probe-source PROBE`: historical probe exits 1, fixed
probe exits 0. It tests extracted norm/RoPE functions; the additional native
verifier executes actual Attention.forward. Neither is pretrained/model parity.
