"""Idle-card quality relay with bounded restart; no performance experiment."""
import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
B=Path.home()/'tli-nfa-ablation';out=B/'C_low_threshold';out.mkdir(exist_ok=True)
lock=(out/'pool.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
plan=json.load((B/'C_LOW_THRESHOLD_PLAN.json').open());admit=json.load((B/'C_reuse_admission.json').open());assert admit['scope']=='HotpotQA/GovReport historical ring quality only; conditional Near-only .05'
assert admit['status']=='usable-with-scope' and admit['low_threshold_plan']=='design-admitted; results pending independent verification'
assert hashlib.sha256((B/'C_LOW_THRESHOLD_PLAN.json').read_bytes()).hexdigest()==admit['bindings']['C_LOW_THRESHOLD_PLAN.json']
identity={name:hashlib.sha256((B/name).read_bytes()).hexdigest() for name in ('C_LOW_THRESHOLD_PLAN.json','C_reuse_admission.json','low_threshold_worker.py','low_threshold_pool.py')}
identity_file=out/'execution_identity.json'
if identity_file.exists():assert json.load(identity_file.open())==identity,'execution plan/harness changed'
else:identity_file.write_text(json.dumps(identity,indent=2))
GPUS=[0,1,2,3,4,5];workers={};statefile=out/'attempts.json'
attempts=json.load(statefile.open()) if statefile.exists() else {str(i):0 for i in range(len(GPUS))}
def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def complete(rank):return (out/f'worker{rank}_done.json').exists()
while True:
 inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 idle={int(i):int(m)<1000 and int(u)<5 for i,m,u in (line.split(',') for line in inventory.splitlines())}
 stopped=(out/'STOP.json').exists()
 for rank,gpu in enumerate(GPUS):
  p=workers.get(rank)
  if stopped or complete(rank) or (p and p.poll() is None) or attempts[str(rank)]>=3 or not idle.get(gpu):continue
  # Child also takes the shared /tmp/nfa-gpu-N.lock and rechecks idleness.
  attempts[str(rank)]+=1;save(statefile,attempts)
  with (out/f'rank{rank}_attempt{attempts[str(rank)]}.log').open('w') as log:
   workers[rank]=subprocess.Popen([str(Path.home()/'tli-yarn-eval/venv/bin/python'),'-u',str(B/'low_threshold_worker.py'),str(rank),str(gpu),str(len(GPUS))],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
 states={i:dict(gpu=GPUS[i],pid=workers[i].pid if i in workers else None,returncode=workers[i].poll() if i in workers else None,attempts=attempts[str(i)],complete=complete(i)) for i in range(len(GPUS))}
 save(out/'health.json',dict(time=time.time(),workers=states,gpu=inventory,stop=stopped,scope='quality only'))
 active=any(p.poll() is None for p in workers.values())
 if all(complete(i) for i in range(len(GPUS))):
  records=[json.load((out/'requests'/f"{r['task']}_{r['row_index']:04d}.json").open()) for r in plan['requests']]
  assert len(records)==127 and all(r['status']=='complete-pending-independent-verification' for r in records)
  assert sum(len(r['arms']) for r in records)==139
  save(out/'RAW_COMPLETE.json',dict(status='pending independent result verification',requests=127,far_generations=139));break
 if stopped and not active:break
 if not active and all(complete(i) or attempts[str(i)]>=3 for i in range(len(GPUS))):
  save(out/'FAILED.json',dict(workers=states));break
 time.sleep(15)
