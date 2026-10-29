"""Durable scheduler tests require no GPU, SSH, or real worker processes."""
import multiprocessing
from pathlib import Path
import pytest
from scripts.remote_frozen_queue import transaction,claim,finish,idle,command,stop_owned


def consume(out,config,host,result):
    ranks=[]
    while True:
        with transaction(out,config) as state:
            job=claim(state,host,0,10.)
            if job is None:break
            ranks.append(job['rank'])
            finish(state,job['rank'],job['token'],True,4)
    result.put(ranks)


def test_shared_queue_has_no_duplicate_claims_across_processes(tmp_path):
    config={'shards':64,'head':'immutable'}
    ctx=multiprocessing.get_context('fork');result=ctx.Queue()
    procs=[ctx.Process(target=consume,args=(str(tmp_path),config,str(i),result)) for i in range(3)]
    for p in procs:p.start()
    lists=[result.get(timeout=15) for _ in procs]
    for p in procs:p.join(timeout=15);assert p.exitcode==0
    all_ranks=[r for group in lists for r in group]
    assert sorted(all_ranks)==list(range(64))
    with transaction(tmp_path,config) as state:assert all(j['status']=='done' for j in state['jobs'])


def test_durable_slot_lease_retry_limit_and_stale_owner(tmp_path):
    config={'shards':2}
    with transaction(tmp_path,config) as state:
        first=claim(state,'pc02',0,1.)
        assert claim(state,'pc02',0,2.) is None
        assert claim(state,'pc02',1,2.)['rank']==1
    # Re-open on disk as after daemon restart; a live lease is still occupied.
    with transaction(tmp_path,config) as state:
        assert claim(state,'pc02',0,3.) is None
        finish(state,0,first['token'],False,2,'synthetic failure')
        second=claim(state,'pc31',0,4.)
        assert second['rank']==0 and second['attempts']==2
        with pytest.raises(ValueError,match='stale'):finish(state,0,first['token'],True,2)
        finish(state,0,second['token'],False,2,'synthetic failure')
        assert state['jobs'][0]['status']=='failed'
        assert claim(state,'pc40',0,5.) is None
    with pytest.raises(ValueError,match='identity'):
        with transaction(tmp_path,{'shards':3}):pass


def test_busy_card_is_never_considered_idle():
    config={'idle_memory_mib':64,'idle_utilization':5}
    assert idle({'memory_mib':4,'utilization':0},config)
    assert not idle({'memory_mib':18000,'utilization':0},config)
    assert not idle({'memory_mib':4,'utilization':30},config)


def test_full_worker_command_preserves_protocol_and_explicit_gpu():
    config={'datasets':['qasper','hotpotqa','musique','gov_report','repobench-p'],
            'shards':64,'context_cap':31500,'out':'/shared/run'}
    cmd=command(config,{'python':'/user/python'},{'rank':12,'gpu':3})
    for flag,value in {'--samples':'0','--thresholds':'0.05','--world-size':'64','--rank':'12','--device':'cuda:3'}.items():
        assert cmd[cmd.index(flag)+1]==value
    assert '--verify-native-full' in cmd


def test_stop_refuses_unrelated_process(monkeypatch):
    monkeypatch.setattr('scripts.remote_frozen_queue.matches',lambda *args:False)
    with pytest.raises(ValueError,match='unrelated'):stop_owned(123,'/out',0)


def test_pid_publication_failure_keeps_live_worker_lease(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from scripts import remote_frozen_queue as queue
    config=dict(out=str(tmp_path),source='/source',shards=1,max_attempts=4,stall_seconds=7200,
        idle_memory_mib=64,idle_utilization=5,context_cap=31500,datasets=['qasper'],
        hosts={'pc02':{'python':'python','gpus':[0]}})
    original=queue.transaction;calls=0
    @contextmanager
    def fail_publication(out,cfg):
        nonlocal calls
        calls+=1
        if calls==3:raise OSError('synthetic PID publication failure')
        with original(out,cfg) as state:yield state
    monkeypatch.setattr(queue,'transaction',fail_publication)
    monkeypatch.setattr(queue,'gpu_inventory',lambda:{0:dict(memory_mib=4,utilization=0)})
    monkeypatch.setattr(queue.subprocess,'Popen',lambda *a,**kw:SimpleNamespace(pid=123))
    with pytest.raises(OSError,match='publication'):queue.tick(config,'pc02')
    with original(tmp_path,config) as state:
        assert state['jobs'][0]['status']=='claiming'
        assert state['jobs'][0]['attempts']==1
        assert queue.claim(state,'pc31',0,100.) is None
    monkeypatch.setattr(queue,'transaction',original)
    monkeypatch.setattr(queue,'find_worker',lambda *a:123)
    queue.reconcile(config,'pc02')
    with original(tmp_path,config) as state:
        assert state['jobs'][0]['status']=='running'
        assert state['jobs'][0]['pid']==123
