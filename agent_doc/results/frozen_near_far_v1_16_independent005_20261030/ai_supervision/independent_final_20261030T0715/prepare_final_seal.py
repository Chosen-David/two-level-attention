"""This run's explicit artifact list; delegates all sealing to the existing tools."""
from pathlib import Path
import json,subprocess,sys,shlex
B=Path(__file__).resolve().parent
V=B.parent.parent
r=json.loads((B/'v1_checked_continued/cpu_checks.json').read_text())
assert r['errors']==[] and r['completed_requests_checked']==3750
q=json.loads((V/'queue.json').read_text())
outputs=[V/'FINAL_REPORT_ZH.md',V/'FINISHED.json',V/'smoke_verification.md',V/'prelaunch_review.md']
outputs += [B/n for n in ['v1_checked_continued/cpu_checks.json','OFFICIAL_SCORES_VERIFIED.json','SCOPE_AUDIT.json','tools/results/v1_cache.json','tools/results/closeout/closeout_checks.json','SOURCE_REVIEW.md','CPU_SOURCE_RETRY.log','CPU_SOURCE_RETRY.command.json','CPU_TEST_SOURCE_BINDING.json','PRIOR_SMOKE_REUSE.json','SNAPSHOT_CACHE_BINDINGS.json','FULL_RECHECK_SNAPSHOT_BINDING.json','REPORT_PROSE_REVIEW.json','FOCUS_FACTS.json','CONTINUATION_REUSE_CHECK.json','REUSED_CHECK_BINDINGS.json','CAPTURE_BINDINGS.json','SNAPSHOT.json','FULL_RECHECK_CONTINUE.log','FULL_RECHECK_CONTINUE.command.json','OFFICIAL.command.json','CLOSEOUT.command.json','CACHE.command.json','AUDIT_SCOPE.command.json','TOOL_REUSE.md','prepare_review.py.diff','extra_checks.py.diff','FULL_V1_EXECUTION.md']]
outputs += sorted(B.glob('tools/**/*.py'))
outputs += [B/n for n in ['official_scores_verified.py','audit_scope.py','run_logged.py','prepare_review.py','prepare_final_seal.py']]
outputs += [Path(p) for p in json.loads((B/'PRIOR_SMOKE_REUSE.json').read_text())['artifact_hashes']]
outputs += [Path(p) for p in json.loads((B/'CPU_TEST_SOURCE_BINDING.json').read_text())['source_tests']]
outputs += sorted(V.glob('worker_rank*_attempt*.log'))
assert len(list(V.glob('worker_rank*_attempt*.log')))==128
cmd=[sys.executable,'-B',str(B/'tools/prepare_spec.py'),'--scope','v1']
for p in sorted(set(outputs)):cmd+=['--output-artifact',str(p)]
for p in [B/'CONTINUATION_ENVIRONMENT.json',V/'preflight/receipts.json',V/'deployment_receipt.json',V/'ai_supervision/round_20261030T071211/20261030T072130_health.json']:
 cmd+=['--environment-artifact',str(p)]
for p in [V/'queue.json',V/'dispatch_plan.json',B/'FULL_RECHECK_CONTINUE.command.json']:
 cmd+=['--command-evidence',str(p)]
for j in q['jobs']:cmd+=['--command',shlex.join(j['command'])]
cmd+=['--environment-description','Production128 shards on pc02/pc31/pc40 and8 GPUs, Qwen3-8B bfloat16, frozen08ce44d TLI eager quality. Actual host/software/preflight and historical resource receipts attached; no timing claim. Independent review CPU only/Python3.11.12/torch2.8.0/transformers4.56.2/numpy2.4.4. Post-launch disk hashes do not authenticate historical loaded process memory.']
subprocess.run(cmd,check=True)
