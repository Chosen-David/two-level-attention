import ast
from pathlib import Path


def test_submission_counter_forwards_grid_and_arguments():
    source=Path(__file__).resolve().parents[1]/'exp/bench/e152v4_graph_latency.py'
    cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='_CountedKernel')
    ns={};exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),ns)
    seen=[]
    class Kernel:
        def __getitem__(self,grid):
            def run(*args,**kw):seen.append((grid,args,kw));return 17
            return run
    counts=dict(capture=0,eager=0);state=[False]
    kernel=ns['_CountedKernel'](Kernel(),counts,lambda:state[0])
    assert kernel[(3,)](4,x=5)==17
    state[0]=True
    assert kernel[(7,)](8,x=9)==17
    assert counts==dict(capture=1,eager=1)
    assert seen==[((3,),(4,),dict(x=5)),((7,),(8,),dict(x=9))]
