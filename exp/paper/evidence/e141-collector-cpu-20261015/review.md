# Independent read-only E141 review

Reviewer `/root/e141_review`, sole writer `/root`.

No blocking code findings. Reviewer independently ran five collector tests in
normal and optimized Python; both passed. The existing TransMLA source's actual
Attention.forward produced exact Q/K/logit parity and probability error 5.96e-8.
Default official-source identity check rejected that alternative under -O,
exit 1. Input RMSNorm, rotary coordinate ordering, right near, exact SWA,
partial-page inclusion and skipped-query bookkeeping were inspected.

Release limitation: requested official 7d8e source was unavailable (HTTP403).
Alternative 040246 source is explicitly identified, never called that official
version. Tests cover random CPU fp32 eager attention and ordinary RoPE, not
YaRN, full decoder/model, bf16, pretrained identity or GPU. No real collection
is accepted or executed. Main and original E141 historical data are untouched.

Final review checked all three numerical counterexamples (historical collector,
missing RMSNorm, missing rotary rearrangement) in normal/-O logs and runs.json:
parity=false, exit 1; fixed parity=true, exit 0. The completed advice accurately
qualifies source identity and scope. Reviewer found no blocker to dev publication,
without accepting real collection. Mixed skip/saturation assertions were added
inside the existing test method; the final suite still contains five tests.
