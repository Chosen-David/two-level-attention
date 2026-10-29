"""Per-host durable pull queue. Each remote host schedules only its own GPUs."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def _lock_target(path):
    """Only accept this lock's uniquely named, sibling owner records."""
    target=Path(os.readlink(path))
    if target.parent!=path.parent or not target.name.startswith(path.name+'.owner.'):
        raise ValueError('unexpected shared-lock owner target')
    return target


def _recover_dead_lock(path,host):
    # Only this host may reap its PID. Serialize its reapers locally and reread
    # the current owner under that lock, preventing stale-owner ABA deletion.
    digest=hashlib.sha256(str(path).encode()).hexdigest()
    local=Path('/tmp')/('frozen-lock-recovery-'+digest+'.lock')
    with local.open('a') as recovery:
        fcntl.flock(recovery,fcntl.LOCK_EX)
        try:
            target=_lock_target(path)
            owner=json.loads(target.read_text())
            if owner['host']!=host:return False
            try:os.kill(owner['pid'],0)
            except ProcessLookupError:pass
            else:return False
            if _lock_target(path)!=target:return False
            path.unlink()
            target.unlink(missing_ok=True)
            return True
        except (FileNotFoundError,ValueError,KeyError,PermissionError):
            return False


@contextmanager
def shared_lock(path, blocking=True):
    """Atomically publish a symlink to a fully written unique owner record.

    A crash before publication leaves an unused file, never an ownerless lock.
    Recovery never steals another host's lock based on elapsed time.
    """
    path=Path(str(path)+'.d').absolute();host=os.uname().nodename
    owner=path.with_name(path.name+'.owner.'+uuid.uuid4().hex+'.json')
    atomic(owner,dict(host=host,pid=os.getpid()))
    deadline=time.monotonic()+30
    acquired=False
    try:
        while True:
            try:os.symlink(str(owner),path)
            except FileExistsError:
                if _recover_dead_lock(path,host):continue
                if not blocking or time.monotonic()>deadline:
                    raise BlockingIOError(str(path))
                time.sleep(.1);continue
            acquired=True
            break
        yield
    finally:
        if acquired:
            # Do not remove a successor if an external operator replaced us.
            try:
                if _lock_target(path)==owner:path.unlink()
            except FileNotFoundError:pass
        owner.unlink(missing_ok=True)


@contextmanager
def transaction(out,config):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    identity=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    with shared_lock(out/'queue.lock'):
        path=out/'queue.json'
        state=json.loads(path.read_text()) if path.exists() else dict(identity=identity,
            jobs=[dict(rank=r,status='pending',attempts=0) for r in range(config['shards'])])
        if state['identity']!=identity:raise ValueError('queue configuration identity changed')
        yield state
        atomic(path,state)


def claim(state,host,gpu,now):
    if any(j['status'] in ('claiming','running') and j.get('host')==host and j.get('gpu')==gpu for j in state['jobs']):
        return None
    job=next((j for j in state['jobs'] if j['status']=='pending'),None)
    if job is None:return None
    job.update(status='claiming',host=host,gpu=gpu,token=uuid.uuid4().hex,
               attempts=job['attempts']+1,started=now,pid=None)
    return dict(job)


def finish(state,rank,token,success,max_attempts,error=None):
    job=state['jobs'][rank]
    if job.get('token')!=token or job['status'] not in ('claiming','running'):
        raise ValueError('stale job owner cannot finalize a different lease')
    job.update(status='done' if success else ('failed' if job['attempts']>=max_attempts else 'pending'),
               finished=time.time(),error=error)


def command(config,node,job):
    return [node['python'],'-m','benchmark.LongBench.prefill_frozen_skip',
        '--attention','tli','--thresholds','0.05','--samples','0',
        '--datasets',','.join(config['datasets']),'--world-size',str(config['shards']),
        '--rank',str(job['rank']),'--device','cuda:'+str(job['gpu']),
        '--context-cap',str(config['context_cap']),'--verify-native-full','--out',config['out']]


def matches(pid,out,rank):
    try:
        args=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split(chr(0))
        return ('benchmark.LongBench.prefill_frozen_skip' in args
                and args[args.index('--out')+1]==str(out)
                and args[args.index('--rank')+1]==str(rank))
    except (OSError,ValueError,IndexError,UnicodeDecodeError):return False


def find_worker(job,out):
    if job.get('pid') and matches(job['pid'],out,job['rank']):return job['pid']
    # Recover an orphan if the daemon died between Popen and saving its PID.
    for p in Path('/proc').iterdir():
        if p.name.isdigit() and matches(int(p.name),out,job['rank']):return int(p.name)
    return None


def gpu_inventory():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=20)
    return {int(parts[0]):dict(memory_mib=int(parts[1]),utilization=int(parts[2]))
            for line in raw.splitlines() if (parts:=line.split(','))}


def idle(gpu,config):
    return gpu['memory_mib']<=config['idle_memory_mib'] and gpu['utilization']<=config['idle_utilization']


def stop_owned(pid,out,rank):
    if not matches(pid,out,rank) or os.getpgid(pid)!=pid:
        raise ValueError('refuse to stop an unrelated process')
    os.killpg(pid,signal.SIGTERM)
    deadline=time.monotonic()+15
    while matches(pid,out,rank) and time.monotonic()<deadline:time.sleep(.2)
    if matches(pid,out,rank):os.killpg(pid,signal.SIGKILL)


def reconcile(config,host):
    out=Path(config['out']);now=time.time()
    with transaction(out,config) as state:
        jobs=[dict(j) for j in state['jobs'] if j['status'] in ('claiming','running') and j.get('host')==host]
    for job in jobs:
        rank=job['rank'];done=out/f'worker_done_rank{rank}.json'
        success=False;error=None
        if done.exists():
            manifest=json.loads((out/f'manifest_rank{rank}.json').read_text())
            success=(json.loads(done.read_text())['identity']==manifest['identity']
                     and manifest['protocol']['git_head']==config['head'])
            if not success:raise ValueError('invalid completion identity')
        pid=find_worker(job,out)
        if pid and not success:
            heartbeat=out/f'heartbeat_rank{rank}.json'
            last=max(job['started'],json.loads(heartbeat.read_text())['time'] if heartbeat.exists() else 0)
            if now-last<=config['stall_seconds']:
                with transaction(out,config) as state:
                    current=state['jobs'][rank]
                    if current.get('token')==job['token']:current.update(status='running',pid=pid,last_alive=now)
                continue
            stop_owned(pid,out,rank);error='own worker stalled past heartbeat timeout'
        elif not success and now-job['started']<60:
            continue  # Do not race a Popen still being published.
        elif not success:error='worker exited before complete checkpoint'
        with transaction(out,config) as state:
            finish(state,rank,job['token'],success,config['max_attempts'],error)


def tick(config,host):
    out=Path(config['out']);node=config['hosts'][host]
    reconcile(config,host)
    if (out/'PAUSE').exists():return dict(status='paused_no_new_jobs')
    inventory=gpu_inventory();started=[]
    for gpu in node['gpus']:
        predecessor=config.get('after_run')
        if predecessor and not (Path(predecessor)/f'worker_done_rank{node["predecessor_ranks"][str(gpu)]}.json').exists():
            continue
        if gpu not in inventory or not idle(inventory[gpu],config):continue
        with transaction(out,config) as state:job=claim(state,host,gpu,time.time())
        if job is None:continue
        proc=None
        try:
            cmd=command(config,node,job)
            logpath=out/f'worker_rank{job["rank"]}_attempt{job["attempts"]}.log'
            with logpath.open('a') as log:
                proc=subprocess.Popen(cmd,cwd=config['source'],
                    env=dict(os.environ,PYTHONPATH=config['source'],PYTHONUNBUFFERED='1',TLI_SCORER_BACKEND='difflib'),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with transaction(out,config) as state:
                current=state['jobs'][job['rank']]
                if current['token']!=job['token']:raise ValueError('lease changed during launch')
                current.update(status='running',pid=proc.pid,command=cmd,log=str(logpath))
            started.append(dict(rank=job['rank'],gpu=gpu,pid=proc.pid))
        except Exception as exc:
            # Once Popen succeeds, keep the durable lease even if PID publication
            # fails. The next tick recovers it by matching this run and rank.
            if proc is None:
                with transaction(out,config) as state:finish(state,job['rank'],job['token'],False,config['max_attempts'],repr(exc))
            raise
    with transaction(out,config) as state:
        counts={status:sum(j['status']==status for j in state['jobs']) for status in ('pending','claiming','running','done','failed')}
    return dict(status='running',counts=counts,started=started,gpus=inventory)


def finalize(config):
    out=Path(config['out'])
    with shared_lock(out/'finalize.lock'):
        if (out/'FINISHED.json').exists():return True
        sys.path.insert(0,config['source'])
        from benchmark.LongBench.validate_frozen_skip import validate
        from benchmark.LongBench.analyze_frozen_skip import analyze
        from benchmark.LongBench.export_frozen_skip import export
        validate(out);analyze(out);export(out,plot=False)
        report=out/'REPORT.md'
        report.write_text('自动生成结果：机械核验已通过，独立结果审阅待完成。\n\n'+report.read_text())
        atomic(out/'FINISHED.json',dict(status='production_complete_pending_independent_review',
            time=time.time(),results_path=config['return_path'],note='Scores mechanically rechecked. Scientific acceptance requires independent result review.'))
        return True


def main():
    ap=argparse.ArgumentParser();ap.add_argument('plan');ap.add_argument('--host',required=True);args=ap.parse_args()
    os.environ['TLI_SCORER_BACKEND']='difflib'
    config=json.loads(Path(args.plan).read_text());host=args.host
    if host not in config['hosts']:raise ValueError('host is outside authorized GPU pool')
    if os.uname().nodename!=config['hosts'][host]['hostname']:
        raise ValueError('host key does not match this physical machine')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=config['source'],text=True).strip()!=config['head']:
        raise ValueError('immutable source HEAD mismatch')
    out=Path(config['out']);out.mkdir(parents=True,exist_ok=True)
    daemon_lock=(out/f'daemon_{host}.lock').open('a')
    try:fcntl.flock(daemon_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    failures=0;last_audit=0
    while True:
        try:
            status=tick(config,host);failures=0
            status.update(pid=os.getpid(),host=host,time=time.time())
            atomic(out/f'daemon_{host}.json',status)
            if time.time()-last_audit>=3600 or status.get('started'):
                with (out/f'daemon_{host}_history.jsonl').open('a') as log:log.write(json.dumps(status)+'\n')
                last_audit=time.time()
            counts=status.get('counts',{})
            if counts.get('done')==config['shards']:
                if finalize(config):return
            elif counts.get('failed') and not any(counts.get(s) for s in ('pending','claiming','running')):
                atomic(out/'NEEDS_ATTENTION.json',dict(time=time.time(),reason='retry budget exhausted',counts=counts));return
        except Exception as exc:
            failures+=1
            atomic(out/f'daemon_{host}.json',dict(pid=os.getpid(),host=host,time=time.time(),status='retrying_daemon_error',error=repr(exc),failures=failures))
            print(repr(exc),flush=True)
        time.sleep(config['poll_seconds'])

if __name__=='__main__':main()
