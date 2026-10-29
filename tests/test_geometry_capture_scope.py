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
    assert geometry_cache_key(key,capturing=True,stream=11) is None
    with capture_graph(object()):
        first=geometry_cache_key(key,capturing=True,stream=11)
        assert lookup(length,first) is None
        assert geometry_cache_key(key,capturing=True,stream=12)!=first
        assert geometry_cache_key(key,capturing=True) is None
        value={'value':length*3};store(length,first,value)
        assert lookup(length.view(1),geometry_cache_key(key,capturing=True,stream=11)) is value
        with ThreadPoolExecutor(1) as pool:
            assert pool.submit(geometry_cache_key,key,capturing=True,stream=11).result() is None
        length.add_(1)
        assert lookup(length,first) is None
        store(length,first,value)
    with capture_graph(object()):
        second=geometry_cache_key(key,capturing=True,stream=11)
        assert second!=first and lookup(length,second) is None
    assert geometry_cache_key(key,capturing=True,stream=11) is None
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
    _=length*2+1
    torch.cuda.synchronize()
    for scoped in (True,True,False):
        built=[0];graph=torch.cuda.CUDAGraph()
        context=capture_graph(graph) if scoped else torch.cuda.graph(graph)
        with context:
            out=[]
            for layer in range(3):
                cache_key=geometry_cache_key(key,capturing=True,stream=torch.cuda.current_stream().cuda_stream)
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

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('sparse', ['0', '1'])
def test_real_selector_cross_graph_geometry(monkeypatch, sparse):
    """Real selectors share only equal geometry and replay independent live inputs."""
    import importlib.util
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    spec = importlib.util.spec_from_file_location(
        'geometry_prod_args', source.parents[2] / 'exp/bench/test_tli_graph_gates.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('TLI_SPARSE_SELECTION', sparse)
    torch.manual_seed(827)
    capacity = 32840
    k = torch.randn(1, capacity, 8, 128, device='cuda', dtype=torch.bfloat16)
    q = torch.randn(1, 1, 32, 128, device='cuda', dtype=k.dtype)
    length = torch.tensor([capacity], device='cuda')
    cu = torch.tensor([0, capacity], device='cuda', dtype=torch.int32)
    selectors = []
    for swa in (128, 128, 192, 128):
        args = module._make_prod_args()
        args.tli_swa_size = swa
        idx = TLIIndexer(args)
        idx.layer_idx = 2
        idx.clear()
        selectors.append((idx, idx.prepare_index(k, cu)))

    builds = [0]
    original = TLIIndexer._build_graph_geometry
    def counted(self, *args, **kwargs):
        builds[0] += 1
        return original(self, *args, **kwargs)
    monkeypatch.setattr(TLIIndexer, '_build_graph_geometry', counted)
    def run():
        return [idx.compute_selection_graph(q, index, 128 ** -.5, length)
                for idx, index in selectors]

    # Compile before capture; leave eager cache populated to test isolation.
    run()
    torch.cuda.synchronize()
    graphs, results = [], []
    for scoped in (True, True, False):
        graph = torch.cuda.CUDAGraph()
        builds[0] = 0
        with capture_graph(graph) if scoped else torch.cuda.graph(graph):
            output = run()
        # Equal adjacent layers share; different geometry evicts this pointer's
        # single cache entry. The final layer must rebuild its original geometry.
        assert builds[0] == (3 if scoped else 4)
        graphs.append(graph)
        results.append(output)
    for valid in (132, 4097, capacity, 512, 32768, capacity):
        length.fill_(valid)
        q.normal_()
        # Alternate replay order, so no graph may depend on another's last run.
        for i in (1, 0, 2, 0):
            graphs[i].replay()
            expected = run()
            for actual, reference in zip(results[i], expected):
                assert torch.equal(actual.indices, reference.indices), (sparse, valid, i, 'IDs')
                assert torch.equal(actual.counts, reference.counts), (sparse, valid, i, 'counts')


def test_capture_ablation_and_invalid_flag(monkeypatch):
    entered = []
    def graph(*args, **kwargs):
        entered.append(True)
        return nullcontext()
    monkeypatch.setattr(torch.cuda, 'graph', graph)
    monkeypatch.setenv('TLI_GRAPH_GEOMETRY_SHARED', '0')
    with capture_graph(object()):
        assert geometry_cache_key(('geometry',), capturing=True, stream=11) is None
    assert len(entered) == 1
    monkeypatch.setenv('TLI_GRAPH_GEOMETRY_SHARED', 'invalid')
    with pytest.raises(ValueError, match='TLI_GRAPH_GEOMETRY_SHARED'):
        with capture_graph(object()):
            pass
    assert len(entered) == 1  # Invalid configuration rejected before capture.
