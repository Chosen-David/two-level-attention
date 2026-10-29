"""Fail closed on mislabeled benchmark routes, without initializing GPU imports."""
import ast
from pathlib import Path
import pytest

source=Path(__file__).resolve().parents[1]/'exp/bench/e152v4_graph_latency.py'
node=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='_check_ring_routes')
namespace={}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
check=namespace['_check_ring_routes']

def rows():
    return [dict(role='ring',layer=i,physical_tokens_per_row=16,logical_tokens=32768,
                 keys_ptr=10+i,values_ptr=20+i) for i in (0,2)]

def test_exact_capture_and_legacy_routes():
    assert check(rows(),{0,2},{(10,20),(12,22)},True)==rows()
    assert check(rows(),{0,2},set(),False)==rows()

@pytest.mark.parametrize('ids,ptrs,enabled',[
    ({0,1},{(10,20),(12,22)},True),
    ({0,2},{(10,20),(11,21)},True), # same count, wrong cache
    ({0,2},{(10,20)},True),
    ({0,2},{(10,20)},False),
])
def test_mislabeled_routes_rejected(ids,ptrs,enabled):
    with pytest.raises(RuntimeError):check(rows(),ids,ptrs,enabled)

def test_unbounded_or_absent_ring_rejected():
    with pytest.raises(RuntimeError):check([],set(),set(),False)
    data=rows();data[0]['physical_tokens_per_row']=32768
    with pytest.raises(RuntimeError):check(data,{0,2},set(),False)
