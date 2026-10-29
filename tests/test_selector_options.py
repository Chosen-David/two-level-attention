import importlib.util
import json
from pathlib import Path
import sys
import pytest
from sparse_attn.ops.selector_options import validate_selector_options


@pytest.mark.parametrize('name',['TLI_SPARSE_SORT_INT16','TLI_COMPACT_SCAN_TILED'])
@pytest.mark.parametrize('value',['','bad','2'])
def test_qwen_rejects_invalid_selector_flag_before_cache_update(monkeypatch,name,value):
    from sparse_attn.patches.qwen3_attn_patch import qwen3_attn_forward
    for key in ['TLI_SPARSE_SORT_INT16','TLI_COMPACT_SCAN_TILED']:monkeypatch.delenv(key,raising=False)
    monkeypatch.setenv(name,value)
    class Cache:
        def get_max_cache_shape(self):return 32840
        def update(self,*a,**kw):raise AssertionError('cache mutated')
    with pytest.raises(ValueError,match=name):
        qwen3_attn_forward(object(),None,(None,None),None,past_key_values=Cache())


def test_packed_capacity_rejected_before_projection_or_cache_update(monkeypatch):
    from sparse_attn.patches.qwen3_attn_patch import qwen3_attn_forward
    monkeypatch.setenv('TLI_SPARSE_SORT_INT16','1');monkeypatch.setenv('TLI_COMPACT_SCAN_TILED','1')
    class Cache:
        def get_max_cache_shape(self):return 131072
        def update(self,*a,**kw):raise AssertionError('cache mutated')
    with pytest.raises(ValueError,match='integer capacity'):
        qwen3_attn_forward(object(),None,(None,None),None,past_key_values=Cache())
    assert validate_selector_options(32840)==(True,True)


def test_actual_paired_commands_keep_baseline_and_optimized_treatments_separate(monkeypatch,tmp_path):
    path=Path(__file__).parents[1]/'exp/bench/operator_model_paired.py'
    spec=importlib.util.spec_from_file_location('paired_selector_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'require_lease',lambda:None)
    monkeypatch.setattr(module,'snapshot',lambda:{'gpu':'fake, 1, 0','processes':[]})
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    launches=[]
    class Child:
        returncode=0;pid=999999
        def __init__(self,command,**kwargs):launches.append((command,kwargs['env']))
        def poll(self):return 0
    monkeypatch.setattr(module.subprocess,'Popen',Child)
    # Git identity lookup is unrelated to the launch treatment being tested.
    monkeypatch.setattr(module.subprocess,'check_output',lambda *a,**kw:'test-revision\n')
    out=tmp_path/'pair'
    monkeypatch.setattr(sys,'argv',['paired','--out-dir',str(out),'--arms','graph_sparse','graph_sparse_index_optimized','--cycles','1','--warmup','1','--repeats','1'])
    monkeypatch.setenv('TLI_SPARSE_SORT_INT16','invalid-inherited-value')
    module.main()
    assert len(launches)==4
    for (command,env),enabled in zip(launches,['0','1','1','0']):
        assert command[1].endswith('e152v4_graph_latency.py')
        assert env['TLI_OPERATOR_MODE']=='serial' and env['TLI_SPARSE_SELECTION']=='1'
        assert env['TLI_SPARSE_SORT_INT16']==env['TLI_COMPACT_SCAN_TILED']==enabled
    assert json.loads((out/'receipt.json').read_text())['execution_complete']


def test_count_actual_kernel_specialization_and_capture():
    from exp.bench.selector_route_observer import CountedKernel,checked_delta
    launches=[]
    class Kernel:
        def __getitem__(self,grid):
            return lambda *args,**kwargs:launches.append((grid,kwargs))
    counts={'capture':0,'eager':0}
    kernel=CountedKernel(Kernel(),counts,lambda:True,{'PACKED':True})
    with pytest.raises(RuntimeError,match='specialization'):kernel[(1,)](PACKED=False)
    assert not launches
    before={'packed':counts.copy()};kernel[(1,)](PACKED=True)
    assert checked_delta({'packed':counts},before,1)['packed']=={'capture':1,'eager':0}
    with pytest.raises(RuntimeError,match='missing'):checked_delta({'packed':counts},before,36)
