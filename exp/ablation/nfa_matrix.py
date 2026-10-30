"""Canonical quality cells; paired timing is scheduled separately."""
import hashlib
import json

BASE=dict(subspace='tail',dimensions=32,sink=4,swa=128,near=4096,
          k1=128,k2=1024,near_pages=32,partition=True,far_score='minmax',
          near_score='avg',boundary=True,mode='full',epsilon=.05,reclaim=True)


def matrix():
    cells={}
    def add(group, **updates):
        spec={**BASE,**updates}
        key=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:12]
        if key not in cells: cells[key]=dict(id=key,spec=spec,groups=[])
        if group not in cells[key]['groups']: cells[key]['groups'].append(group)
    for subspace in ('full','tail','highfreq','random'):
        add('A_coordinates',subspace=subspace,dimensions=128 if subspace=='full' else 32)
    for dim in (8,16,32,64): add('A_dimensions',dimensions=dim)
    for partition,far,near in ((False,'minmax','minmax'),(True,'minmax','minmax'),(True,'avg','avg'),(True,'minmax','avg')):
        for boundary in (False,True):
            add('B_selection',partition=partition,far_score=far,near_score=near,boundary=boundary)
    add('C_gates')
    for mode in ('near','far','dual'):
        for epsilon in (.01,.025,.05,.1): add('C_gates',mode=mode,epsilon=epsilon)
        for reclaim in (False,True): add('C_cache',mode=mode,reclaim=reclaim)
    for near in (1024,2048,4096,8192): add('E_near_length',near=near)
    for sink in (0,4,16,64):
        for swa in (0,64,128,256): add('F_protection',sink=sink,swa=swa)
    return list(cells.values())

if __name__=='__main__':
    print(json.dumps(matrix(),indent=2))
