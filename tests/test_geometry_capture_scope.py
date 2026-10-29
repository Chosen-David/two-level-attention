"""Capture geometry must share within one graph and invalidate across graphs."""
import ast
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
import torch
from sparse_attn.ops.cuda_graph_state import capture_graph, geometry_cache_key

source=Path(__file__).resolve().parents[1]/'sparse_attn/indexer/tli_indexer.py'
functions=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('_graph_geo_lookup','_graph_geo_store')]
ns={'_GRAPH_GEO_CACHE':{}};exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),ns)
lookup,store=ns['_graph_geo_lookup'],ns['_graph_geo_store']


def test_capture_namespace_and_mutation_invalidation(monkeypatch):
    monkeypatch.setattr(torch.cuda,'graph',lambda *a,**kw:nullcontext())
    ns['_GRAPH_GEO_CACHE'].clear();length=torch.tensor([31]);key=('geometry',)
    eager={'value':length*2};store(length,key,eager)
    assert geometry_cache_key(key,capturing=False)==key
    assert geometry_cache_key(key,capturing=True) is None
    with capture_graph(object()):
        first=geometry_cache_key(key,capturing=True)
        assert lookup(length,first) is None
        value={'value':length*3};store(length,first,value)
        assert lookup(length.view(1),geometry_cache_key(key,capturing=True)) is value
        with ThreadPoolExecutor(1) as pool:
            assert pool.submit(geometry_cache_key,key,capturing=True).result() is None
        length.add_(1)
        assert lookup(length,first) is None
        store(length,first,value)
    with capture_graph(object()):
        second=geometry_cache_key(key,capturing=True)
        assert second!=first and lookup(length,second) is None
    assert geometry_cache_key(key,capturing=True) is None
    assert lookup(length,key) is None


def test_capture_exception_and_inference_tensor(monkeypatch):
    monkeypatch.setattr(torch.cuda,'graph',lambda *a,**kw:nullcontext())
    with pytest.raises(RuntimeError,match='abort'):
        with capture_graph(object()):raise RuntimeError('abort')
    assert geometry_cache_key(('geometry',),capturing=True) is None
    with torch.inference_mode():length=torch.tensor([9])
    ns['_GRAPH_GEO_CACHE'].clear()
    value={'x':1}
    assert store(length,('test',),value) is value
    assert lookup(length,('test',)) is None and not ns['_GRAPH_GEO_CACHE']

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_two_graphs_replay_live_geometry_and_raw_capture():
    ns['_GRAPH_GEO_CACHE'].clear()
    length=torch.tensor([31],device='cuda');key=('gpu-geometry',)
    counts=[];graphs=[];outputs=[]
    # Eager cache from unchanged length must not leak into either capture.
    store(length,key,{'x':length*999})
    for scoped in (True,True,False):
        built=[0];graph=torch.cuda.CUDAGraph()
        context=capture_graph(graph) if scoped else torch.cuda.graph(graph)
        with context:
            out=[]
            for layer in range(3):
                cache_key=geometry_cache_key(key,capturing=True)
                value=None if cache_key is None else lookup(length,cache_key)
                if value is None:
                    value=store(length,cache_key,{'x':length*2});built[0]+=1
                out.append(value['x']+layer)
        counts.append(built[0]);graphs.append(graph);outputs.append(out)
    assert counts==[1,1,3]
    for value in (17,32,1,95):
        length.fill_(value)
        for graph,results in zip(graphs,outputs):
            graph.replay()
            assert [int(x.item()) for x in results]==[value*2+i for i in range(3)]
