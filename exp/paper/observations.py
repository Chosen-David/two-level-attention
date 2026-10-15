"""Offline full-reference mass accounting. Not a production selector."""
import numpy as np


def protected_ids(position, sink, swa):
    if any(type(x) is not int or x < 0 for x in (position,sink,swa)):
        raise ValueError('invalid frozen protection geometry')
    return sorted(set(range(min(position+1,sink))) | set(range(max(0,position-swa+1),position+1)))


def mass_partition(probability, selected, protected):
    p=np.asarray(probability)
    if p.ndim!=1 or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1,atol=1e-5):
        raise ValueError('need full-reference probability distribution')
    sets=[]
    for ids in (selected,protected):
        values=list(ids)
        if any(type(x) is not int or not 0<=x<len(p) for x in values) or len(set(values))!=len(values):
            raise ValueError('invalid or duplicate IDs')
        sets.append(set(values))
    sel,prot=sets
    return dict(selected_protected_mass=float(p[sorted(sel&prot)].sum()),
                selected_nonprotected_mass=float(p[sorted(sel-prot)].sum()),
                all_protected_reference_mass=float(p[sorted(prot)].sum()))


def o1_mass(probability,position,near_start,protected,edges):
    p=np.asarray(probability)
    mass_partition(p,[],protected)
    if len(p)!=position+1 or type(near_start) is not int or not 0<=near_start<=position+1:
        raise ValueError('near boundary must be frozen per query')
    if not isinstance(edges,list) or len(edges)<2 or edges[0]!=0 or edges[-1]<=position or any(type(x) is not int for x in edges) or any(a>=b for a,b in zip(edges,edges[1:])):
        raise ValueError('distance bins must cover every causal position exactly')
    prot=set(protected);out=[]
    for lo,hi in zip(edges,edges[1:]):
        for region in ('near','far'):
            ids=[i for i in range(position+1) if i not in prot and lo<=position-i<hi and ((i>=near_start)==(region=='near'))]
            mass=float(p[ids].sum())
            out.append(dict(distance_lo=lo,distance_hi=hi,region=region,mass=mass,token_count=len(ids),density=mass/len(ids) if ids else None))
    if not np.isclose(sum(r['mass'] for r in out)+p[sorted(prot)].sum(),1,atol=1e-5):
        raise ValueError('mass conservation failed')
    return out
