"""Hourly remote supervision for a fixed, disjoint frozen-region screen."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time


def ssh(host,program):
    result=subprocess.run(['sshpass','-f','/tmp/.sshpw','ssh','-o','ConnectTimeout=10',
        '-p','8416','wangyuanshuo02@'+host,'python -'],input=program,
        capture_output=True,text=True,timeout=60)
    if result.returncode:raise RuntimeError(result.stderr[-2000:])
    return json.loads(result.stdout)


def check(plan,worker,out):
    payload=dict(worker=worker,source=plan['source'],head=plan['head'],out=str(out),world=len(plan['workers']))
    program='payload='+repr(payload)+'\n'+r'''
import os,json,subprocess,time
from pathlib import Path
w=payload['worker'];root=Path(payload['source']);out=Path(payload['out']);rank=w['rank']
if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=payload['head']:
 raise RuntimeError('source HEAD mismatch')
statepath=out/f'pool_worker_{rank}.json'
state=json.loads(statepath.read_text()) if statepath.exists() else {'attempts':0}
if (out/f'worker_done_rank{rank}.json').exists():
 print(json.dumps(dict(rank=rank,status='done')));raise SystemExit
pid=state.get('pid');cmdpath=Path(f'/proc/{pid}/cmdline')
if pid and cmdpath.exists():
 args=cmdpath.read_bytes().decode().split(chr(0))
 if 'benchmark.LongBench.prefill_frozen_skip' in args and str(out) in args and args[args.index('--rank')+1]==str(rank):
  print(json.dumps(dict(rank=rank,status='alive',pid=pid)));raise SystemExit
if state['attempts']>=4:
 print(json.dumps(dict(rank=rank,status='restart_budget_exhausted')));raise SystemExit
free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
if int(free[w['gpu']])<40000:
 print(json.dumps(dict(rank=rank,status='waiting_for_gpu')));raise SystemExit
command=[w['python'],'-m','benchmark.LongBench.prefill_frozen_skip','--attention','tli','--thresholds','0.05','--samples','20','--world-size',str(payload['world']),'--rank',str(rank),'--device','cuda:'+str(w['gpu']),'--verify-native-full','--out',str(out)]
with (out/f'worker_rank{rank}_attempt{state["attempts"]}.log').open('a') as log:
 p=subprocess.Popen(command,cwd=root,env=dict(os.environ,PYTHONPATH=str(root),PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
state.update(pid=p.pid,attempts=state['attempts']+1,command=command,started=time.time())
statepath.write_text(json.dumps(state,indent=2))
print(json.dumps(dict(rank=rank,status='started',pid=p.pid,host=w['host'],gpu=w['gpu'])))
'''
    return ssh(worker['host'],program)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('plan');args=ap.parse_args()
    plan=json.loads(Path(args.plan).read_text());out=Path(plan['out']);out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'pool_supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures=[pool.submit(check,plan,w,out) for w in plan['workers']]
            events=[]
            for w,f in zip(plan['workers'],futures):
                try:events.append(f.result())
                except Exception as exc:events.append(dict(rank=w['rank'],status='check_failed',error=str(exc)))
        complete=all(e['status']=='done' for e in events)
        status=dict(pid=os.getpid(),checked_at=time.time(),next_check_at=time.time()+3600,
            status='production_complete_pending_independent_review' if complete else 'running',events=events)
        (out/'pool_status.json').write_text(json.dumps(status,indent=2)+'\n')
        with (out/'pool_history.jsonl').open('a') as log:log.write(json.dumps(status)+'\n')
        print(json.dumps(status),flush=True)
        if complete:return
        time.sleep(3600)

if __name__=='__main__':main()
