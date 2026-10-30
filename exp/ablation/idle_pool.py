"""Autonomous idle-GPU relay, adapted from the existing yarn125_pool.

Admission and job argv are explicit files. Holds a shared per-GPU flock for the
whole worker lifetime, never kills other jobs, persists bounded retry counts.
Completion requires the runner's done marker and every expected request.
"""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path


def save(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--plan',required=True);ap.add_argument('--admission',required=True)
    ap.add_argument('--gpus',default='0,1,2,3,4,5,7');args=ap.parse_args()
    plan=json.loads(Path(args.plan).read_text());admit=json.loads(Path(args.admission).read_text())
    sha=plan['code_revision'];code=Path(plan['code']);out=Path(plan['output']);out.mkdir(parents=True,exist_ok=True)
    assert admit['status']=='usable-with-scope' and admit['code_revision']==sha
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()==sha
    owner=(out/'pool.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=plan['jobs'];ids=[j['id'] for j in jobs];assert len(ids)==len(set(ids))
    statefile=out/'attempts.json';attempts=json.loads(statefile.read_text()) if statefile.exists() else {}
    live={};gpus=[int(i) for i in args.gpus.split(',')]
    def complete(job):
        marker=Path(job['done']);directory=Path(job['requests'])
        if not marker.exists():return False
        receipt=json.loads(marker.read_text())
        manifest=json.loads(Path(job['manifest']).read_text())
        if manifest['protocol']['git_head']!=sha or manifest['identity']!=receipt['identity']:
            raise ValueError('completed shard source/identity differs from plan')
        rows=[directory/f'{i:04d}.json' for i in range(job['start'],job['end'])]
        return all(p.exists() and json.loads(p.read_text()).get('status')=='complete' and json.loads(p.read_text()).get('identity')==receipt['identity'] for p in rows)
    while True:
        for gpu,(p,lease,job) in list(live.items()):
            if p.poll() is not None:
                lease.close();del live[gpu]
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        idle={int(i):int(m)<1000 and int(u)<5 for i,m,u in (s.split(',') for s in inventory.splitlines())}
        pending=[j for j in jobs if not complete(j) and j['id'] not in [x[2]['id'] for x in live.values()] and attempts.get(j['id'],0)<3]
        for gpu in gpus:
            if not pending or gpu in live or not idle.get(gpu):continue
            lease=open(f'/tmp/nfa-gpu-{gpu}.lock','a')
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:lease.close();continue
            # Recheck after acquiring the cooperative lock.
            current=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
            if int(current[0])>=1000 or int(current[1])>=5:lease.close();continue
            job=pending.pop(0);attempts[job['id']]=attempts.get(job['id'],0)+1;save(statefile,attempts)
            logpath=out/f"{job['id']}_attempt{attempts[job['id']]}.log"
            with logpath.open('w') as log:
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',TLI_SCORER_BACKEND='difflib')
                process=subprocess.Popen(job['argv'],cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            live[gpu]=(process,lease,job)
        done={j['id']:complete(j) for j in jobs}
        save(out/'health.json',dict(time=time.time(),code_revision=sha,complete=done,attempts=attempts,gpu=inventory,workers={g:dict(pid=p.pid,job=j['id']) for g,(p,lease,j) in live.items()}))
        if all(done.values()):save(out/'RAW_COMPLETE.json',dict(time=time.time(),status='pending independent result verification'));return
        if not live and all(done[j['id']] or attempts.get(j['id'],0)>=3 for j in jobs):save(out/'FAILED.json',dict(time=time.time(),incomplete=[i for i,v in done.items() if not v]));return
        time.sleep(15)

if __name__=='__main__':main()
