"""Verify real CLI construction and frozen inputs without launching a GPU job."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

path=Path(__file__).resolve().parents[1]/'exp/bench/operator_model_paired.py'
spec=importlib.util.spec_from_file_location('paired_benchmark',path)
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)

@pytest.mark.parametrize('arms,modes,backends',[
    (['graph_ring_legacy','graph_ring_simt'],['off','serial'],['simt','simt']),
    (['graph_ring_simt','graph_ring_tc'],['serial','serial'],['simt','tc']),
    (['graph_ring_simt','graph_ring_tc3'],['serial','serial'],['simt','tc']),
    (['graph_ring_tc3','graph_ring_tc3_overlap'],['serial','overlap'],['tc','tc']),
    (['graph_ring_simt','graph_ring_overlap'],['serial','overlap'],['simt','simt']),
])
def test_ring_pairs_freeze_mask_and_construct_graph_commands(monkeypatch,tmp_path,arms,modes,backends):
    mask=tmp_path/'input.json';mask.write_text('{"skip":[0,2]}')
    out=tmp_path/'out';calls=[]
    monkeypatch.setattr(sys,'argv',['bench','--arms',*arms,'--ring-mask',str(mask),'--out-dir',str(out),'--cycles','1','--repeats','1'])
    monkeypatch.setattr(bench,'require_lease',lambda:None)
    monkeypatch.setattr(bench,'snapshot',lambda:dict(gpu='uuid, 0, 0, 40, 1500, 50',processes=[]))
    monkeypatch.setattr(bench.time,'sleep',lambda _:None)
    monkeypatch.setattr(bench.subprocess,'check_output',lambda *a,**kw:'test-commit\n')
    def launch(command,**kwargs):
        calls.append((command,kwargs['env']))
        # Original input can change after dispatch; every arm must use frozen bytes.
        mask.write_text('{"skip":[1]}')
        return SimpleNamespace(pid=123,returncode=0,poll=lambda:0)
    monkeypatch.setattr(bench.subprocess,'Popen',launch)
    bench.main()
    assert len(calls)==4
    for (cmd,env),arm_idx in zip(calls,[0,1,1,0]):
        assert cmd[1].endswith('e152v4_graph_latency.py')
        assert env['TLI_OPERATOR_MODE']==modes[arm_idx]
        assert env['TLI_OPERATOR_BACKEND']==backends[arm_idx]
        assert env['TLI_BENCH_REQUIRE_RING']=='1'
        assert env['TLI_OPERATOR_SELECTED_STAGES']==('3' if arms[arm_idx] in ('graph_ring_tc3','graph_ring_tc3_overlap') else '1')
        for flag,value in [('--tli_enable_layer_skip','true'),('--tli_far_skip_backend','compact'),('--tli_far_skip_cache','evict')]:
            assert cmd.count(flag)==1 and cmd[cmd.index(flag)+1]==value
        frozen=Path(cmd[cmd.index('--tli_layer_skip_path')+1])
        assert frozen.is_absolute() and json.loads(frozen.read_text())=={'skip':[0,2]}
    receipt=json.loads((out/'receipt.json').read_text())
    assert receipt['ring_skip_ids']==[0,2]

@pytest.mark.parametrize('extra',[
    ['--arms','graph_ring_simt','fa2','--ring-mask','unused'],
    ['--arms','graph_ring_simt','graph_ring_tc'],
    ['--arms','fa2','serial','--ring-mask','unused'],
])
def test_mismatched_protocol_rejected_before_lease(monkeypatch,tmp_path,extra):
    monkeypatch.setattr(sys,'argv',['bench','--out-dir',str(tmp_path/'out'),*extra])
    def lease():raise AssertionError('must reject before acquiring GPU')
    monkeypatch.setattr(bench,'require_lease',lease)
    with pytest.raises(SystemExit) as exc:bench.main()
    assert exc.value.code==2
