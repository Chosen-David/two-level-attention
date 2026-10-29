"""Hourly local supervisor: bounded resume, progress audit, score aggregation."""
import argparse
from datetime import datetime, timezone, timedelta
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmark.LongBench.analyze_frozen_skip import analyze
from benchmark.LongBench.prefill_frozen_skip import atomic_json


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--out',default='agent_doc/results/frozen_near_far_staging_20261029')
    ap.add_argument('--samples',type=int,default=20)
    ap.add_argument('--thresholds',default='0.10,0.15')
    ap.add_argument('--context-cap',type=int,default=31500)
    ap.add_argument('--interval',type=int,default=3600)
    ap.add_argument('--devices',default='0,1')
    ap.add_argument('--max-restarts',type=int,default=3)
    args=ap.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    lock=out/'supervisor.lock'
    import fcntl
    handle=lock.open('a+')
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('another supervisor holds this run lock')
    devices=args.devices.split(',');workers={};started={};attempts={rank:0 for rank in range(len(devices))}
    base=[sys.executable,'-m','benchmark.LongBench.prefill_frozen_skip','--samples',str(args.samples),
        '--thresholds',args.thresholds,'--context-cap',str(args.context_cap),'--world-size',str(len(devices)),
        '--out',str(out),'--verify-native-full']
    atomic_json(out/'supervisor_config.json',dict(pid=os.getpid(),interval_seconds=args.interval,worktree=str(ROOT),
        devices=devices,worker_command_base=base,max_restarts=args.max_restarts,
        policy='Only resume this run; no unrelated process termination; no automatic treatment changes.'))
    while True:
        now=datetime.now(timezone.utc);events=[]
        summary=analyze(out)
        if summary and summary['status']=='complete':
            atomic_json(out/'supervisor_status.json',dict(pid=os.getpid(),status='complete',checked_at=now.isoformat(),
                completed_requests=summary['completed_requests'],expected_requests=summary['expected_requests']))
            print('COMPLETE: paired screen and final summary saved',flush=True)
            return
        try:
            gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
            gpu_text=gpu.stdout
        except subprocess.TimeoutExpired:gpu_text=''
        free={line.split(',')[0].strip():int(line.split(',')[1].strip()) for line in gpu_text.splitlines() if line.strip()}
        exhausted=False
        for rank,device in enumerate(devices):
            if (out/f'worker_done_rank{rank}.json').exists():continue
            process=workers.get(rank)
            if process is not None and process.poll() is None:
                heartbeat=out/f'heartbeat_rank{rank}.json'
                last=json.loads(heartbeat.read_text()) if heartbeat.exists() else None
                last_time=max(started[rank],last['time'] if last else 0)
                if time.time()-last_time<=7200:
                    events.append(dict(rank=rank,pid=process.pid,state='alive',heartbeat=last))
                    continue
                # Only this supervisor's own process group may be terminated.
                events.append(dict(rank=rank,pid=process.pid,state='stalled_over_2h'))
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=15)
            if process is not None:events.append(dict(rank=rank,pid=process.pid,state='exited',exit_code=process.returncode))
            if attempts[rank]>args.max_restarts:
                events.append(dict(rank=rank,state='restart_budget_exhausted'));exhausted=True;continue
            if free.get(device,0)<40000:
                events.append(dict(rank=rank,state='waiting_for_gpu_memory',free_mib=free.get(device)));continue
            command=base+['--rank',str(rank),'--device','cuda:'+device]
            log=(out/f'worker_rank{rank}_attempt{attempts[rank]}.log').open('a')
            env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1')
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            log.close();workers[rank]=process;started[rank]=time.time();attempts[rank]+=1
            events.append(dict(rank=rank,pid=process.pid,state='started',command=command,attempt=attempts[rank]))
        status=dict(pid=os.getpid(),status='attention_required' if exhausted else 'running',checked_at=now.isoformat(),
            next_check_at=(now+timedelta(seconds=args.interval)).isoformat(),events=events,gpu_snapshot=gpu_text,
            completed_requests=summary['completed_requests'] if summary else 0,
            expected_requests=summary['expected_requests'] if summary else args.samples*5)
        atomic_json(out/'supervisor_status.json',status)
        with (out/'supervision.jsonl').open('a') as stream:stream.write(json.dumps(status)+'\n')
        print(json.dumps(status),flush=True)
        if exhausted and all((out/f'worker_done_rank{rank}.json').exists() or
                (attempts[rank]>args.max_restarts and (rank not in workers or workers[rank].poll() is not None))
                for rank in range(len(devices))):return
        time.sleep(args.interval)


if __name__=='__main__':main()
