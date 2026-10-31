"""Paired CPU-only timing of actual Python _reclaim; no tensor/GPU surrogate timing."""
import ast
import argparse
from dataclasses import dataclass
import heapq
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline',type=Path,required=True)
parser.add_argument('--candidate',type=Path,default=Path('sparse_attn/ops/paged_region_cache.py'))
parser.add_argument('--geometry',type=Path,default=Path('sparse_attn/ops/frozen_region_cache.py'))
args=parser.parse_args()
def load(path):
    ns={'dataclass':dataclass,'CacheLayerMixin':object,'heapq':heapq}
    for file,name in ((args.geometry,'RegionGeometry'),(path,'PagedRegionLayer')):
        nodes=[n for n in ast.parse(file.read_text()).body if isinstance(n,ast.ClassDef) and n.name==name]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(file),'exec'),ns)
    return ns['PagedRegionLayer'],ns['RegionGeometry']

classes=[load(args.baseline),load(args.candidate)]
rows=[]
for length in (8192,32768,131072):
    for near in (False,True):
        layers=[]
        for L,G in classes:
            layer=L(G());layer.logical_length=length;layer.skip_near=near
            layer.pages={i:(i,None,None) for i in range(length//64)}
            layers.append(layer)
        def measure(layer):
            start=perf_counter_ns()
            for _ in range(100):layer._reclaim()
            return (perf_counter_ns()-start)/100
        for _ in range(3):
            for layer in layers:measure(layer)
        times=[[],[]]
        for rep in range(15):
            for which in ((0,1,1,0) if rep%2==0 else (1,0,0,1)):
                times[which].append(measure(layers[which]))
        rows.append(dict(length=length,skip_near=near,pages=length//64,
            baseline_ns=times[0],candidate_ns=times[1],
            baseline_median_ns=median(times[0]),candidate_median_ns=median(times[1])))
print(json.dumps(dict(scope='CPU metadata _reclaim only; far retained; no GPU/E2E/allocator timing',rows=rows),indent=2))
