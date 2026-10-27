# TL-SINGLE-OUTPUT-20261027

Status: implemented, CPU checked; GPU pending. Parent PR10ebacac0, main8f213e4.
Hypothesis: unconditional cat copies a sole attention output at B1.
Change: reuse that output; B>1 and EOS semantics unchanged.
Validation:16 CPU passed/8 CUDA skipped; exact paired logits and assembly counts.
Evidence: agent_doc/results/single-output-20261027/validation.md.
Next: CUDA old/new capture tests and paired full-model latency; no speed claim yet.
