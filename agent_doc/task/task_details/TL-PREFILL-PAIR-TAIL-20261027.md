# TL-PREFILL-PAIR-TAIL-20261027

Status: implementation and CPU regression complete; GPU acceptance pending.
Scope: main17e56e8 prefill observer, integrated on PR10 parent277c464. Integrated new main8f213e4 source-release repair and all four upstream files. No new dependency.
Hypothesis: converting unused full-prompt Q wastes traffic/allocation in static_pair mode.
Change: slice tail before FP32 conversion; exact same pair selection and request lifetime.
Evidence: agent_doc/results/prefill-tail-20261027/validation.md;61 CPU passed,30 CUDA skipped;
Post-merge68 CPU passed/55 CUDA skipped.
paired CPU cast storage16781312 ->1048576 bytes, not GPU peak or latency.
Next: remote CUDA observer matrix/paired bench and existing model eager/Graph gates.
Reject if selected IDs differ; do not relax tolerance. Dynamic gate contraction deferred.
