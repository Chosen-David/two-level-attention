"""Bounded CPU verification; captures each process's actual returncode."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
ENV=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',MPLCONFIGDIR='/tmp/observation-mpl',XDG_CACHE_HOME='/tmp/observation-cache')
PYTHON='/workspace/twolevel-cpu-env/bin/python'

def main():
    phase=sys.argv[1]
    if phase=='render':
        commands=[('render',['python','agent_doc/guide/paper_figs/figs/fig_o_observation.py','--diagnostic'])]
    elif phase=='regression':
        commands=[]
        for mode,flag in [('normal',[]),('optimized',['-O'])]:
            for name,interpreter,pattern in [('bins','python','test_observation_bins.py'),('assets','python','test_e140_asset_provenance.py'),('collector',PYTHON,'test_e140_o*.py')]:
                commands.append((name+'-'+mode,[interpreter,*flag,'-m','unittest','discover','-s','tests','-p',pattern,'-v']))
    else:raise ValueError('unknown phase')
    record=OUT/(phase+'-execution.json')
    if record.exists():raise FileExistsError('preserve existing run evidence; use a fresh version')
    rows=[]
    for name,command in commands:
        path=OUT/(phase+'-'+name+'.log')
        if path.exists():raise FileExistsError(path)
        result=subprocess.run(command,cwd=ROOT,env=ENV,capture_output=True,text=True)
        path.write_text(result.stdout+result.stderr)
        rows.append(dict(name=name,command=command,exit_code=result.returncode,log=path.name))
        print(name,result.returncode)
    record.write_text(json.dumps(rows,indent=2)+'\n')
    return int(any(r['exit_code']!=0 for r in rows))

if __name__=='__main__':sys.exit(main())
