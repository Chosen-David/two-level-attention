"""One-shot read-only operational capture; never schedules or modifies workers."""
import collections, concurrent.futures, json, pathlib, subprocess, sys, time
root = pathlib.Path(__file__).resolve().parents[2]
label = sys.argv[1]
assert label.isidentifier()
now = time.time()
plan = json.loads((root / 'dispatch_plan.json').read_text())
queue = json.loads((root / 'queue.json').read_text())
records, errors, seen = [], [], set()
for path in sorted((root / 'requests').glob('*/*.json')):
    try:
        data = json.loads(path.read_text())
        key = (data['dataset'], data['row_index'])
        if key in seen: errors.append('duplicate:' + repr(key))
        seen.add(key)
        records.append(dict(dataset=key[0], row_index=key[1], status=data['status'], arms=list(data['arms'])))
    except Exception as exc: errors.append(str(path) + ':' + repr(exc))
coverage = {ds: dict(complete=sum(x['dataset']==ds and x['status']=='complete' for x in records),
    partial=sum(x['dataset']==ds and x['status']!='complete' for x in records),
    generations=sum(len(x['arms']) for x in records if x['dataset']==ds)) for ds in plan['datasets']}
daemons, workers = {}, []
for host in plan['hosts']:
    data = json.loads((root / ('daemon_'+host+'.json')).read_text())
    daemons[host] = dict(status=data['status'], age_seconds=now-data['time'], pid=data['pid'])
for job in queue['jobs']:
    if job['status'] != 'running': continue
    path = root / ('heartbeat_rank'+str(job['rank'])+'.json')
    hb = json.loads(path.read_text()) if path.exists() else {}
    workers.append({k:job.get(k) for k in ('rank','host','gpu','pid','attempts')} | dict(
        heartbeat=hb, heartbeat_age_seconds=time.time()-hb['time'] if hb else None,
        log_tail=(root/pathlib.Path(job['log']).name).read_text()[-1500:]))
receipt = json.loads((root/'deployment_receipt.json').read_text())
def remote(item):
    host, cfg = item
    pids = [daemons[host]['pid'],receipt['hosts'][host]['watchdog_pid']] + [w['pid'] for w in workers if w['host']==host]
    cmd = 'ps -p '+','.join(map(str,pids))+' -o pid,ppid,etime,stat,args; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; git -C '+plan['source']+' rev-parse HEAD'
    try:
        proc = subprocess.run(['sshpass','-f','/tmp/.sshpw','ssh','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=no','-p','8416','wangyuanshuo02@'+cfg['address'],cmd], capture_output=True,text=True,timeout=40)
        return host,dict(returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr)
    except Exception as exc: return host,dict(error=repr(exc))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    remote_checks = dict(pool.map(remote,plan['hosts'].items()))
snapshot = dict(time=now, counts=dict(collections.Counter(j['status'] for j in queue['jobs'])), coverage=coverage,
    complete_pairs=sum(x['complete'] for x in coverage.values()),generations=sum(x['generations'] for x in coverage.values()),
    daemon=daemons, workers=workers,errors=errors,done_markers=len(list(root.glob('worker_done_rank*.json'))),
    records=records,remote_checks=remote_checks)
output = pathlib.Path(__file__).parent/('snapshot_'+label+'.json')
output.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in snapshot.items() if k not in ('workers','records','remote_checks')},ensure_ascii=False,indent=2))
print('saved',output)
