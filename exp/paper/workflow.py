"""Paper entry point. Default plan; never launches GPU or generation jobs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT / "exp" / "figures"))
from exp.paper.contract import protocol_errors, summarize, digest

FIGURES={'Fig1':{'stable_ids':['O1','O3'],'entry':'exp/figures/o3_retrieval.py','needs':['trace identities','RoPE receipt','frozen protection/distance bins']},
         'Fig2':{'stable_ids':['architecture'],'entry':'architecture.mmd','needs':[]},
         'Fig3':{'stable_ids':['E2'],'entry':'timing','needs':['direct T3 raw repetitions','paired quality identity','independent acceptance']},
         'Fig4':{'stable_ids':['E1','E3'],'entry':'exp/figures/paired_quality.py','needs':['FullKV/SS sample/document/input identity','inline run caps','scorer binding']}}


def write(path,obj):
    with path.open('x') as f: json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')


def plan(protocol):
    errors=protocol_errors(protocol)
    return dict(schema='paper-plan-v1',execution='disabled-no-host-snapshot',gpu_started=False,
                protocol_status='blocked' if errors else 'ready-for-host-review', blockers=errors,
                figures=FIGURES,launch_commands=[],
                dynamic_research='paused',performance_claim=None)


def inventory(root):
    """Only the design's named existing artifact families; no raw text export."""
    families={'dense':'exp/trace/results/e123_trial_raw/pred_fullkv',
              'FS':'exp/trace/results/e123_trial_raw/pred_mavg',
              'SS':'exp/trace/results/e134_l1sub/pred_mavg_l1sub',
              'SF':'exp/trace/results/e134_l2full/pred_mavg_l2full'}
    rows=[]
    for arm,directory in families.items():
        files=sorted(p for p in (root/directory).glob('*') if p.is_file() and p.suffix in ('.json','.jsonl'))
        artifacts=[dict(path=str(p.relative_to(root)),sha256=digest(p),bytes=p.stat().st_size) for p in files]
        rows.append(dict(arm=arm,status='identity_missing' if files else 'absent_in_reviewed_artifacts',
                         artifacts=artifacts,scientific_acceptance=None,
                         blockers=['historical producer/config/sample/document/input/scorer/phase correspondence not independently established']))
    rows.append(dict(arm='FF',status='absent_in_reviewed_artifacts',artifacts=[],
                     blockers=['no accepted FF source in the design ledger; not a global absence claim']))
    import subprocess
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    return dict(schema='paper-reuse-ledger-v1',source_commit=source_commit,arms=rows,raw_predictions_exported=False)


def architecture(out):
    import shutil
    shutil.copyfile(Path(__file__).with_name('architecture.mmd'),out/'Fig2.mmd')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,4));ax.set(xlim=(0,11),ylim=(0,4));ax.axis('off')
    nodes=[(1,2,'Full-D Q/K/V'),(3,2,'Shared low-D coordinates\nper layer / KV head'),
           (5.3,3,'Near: average\nL1 pages → L2 tokens'),(5.3,1,'Far: minmax\nL1 pages → L2 tokens'),
           (8,2,'Union + dedup\ncausal sink/SWA protection'),(10,2,'Full-D\nattention')]
    for x,y,label in nodes: ax.text(x,y,label,ha='center',va='center',fontsize=9,bbox=dict(boxstyle='round',fc='white',ec='black'))
    for a,b in [(0,1),(1,2),(1,3),(2,4),(3,4),(4,5)]:
        ax.annotate('',xy=(nodes[b][0]-.7,nodes[b][1]),xytext=(nodes[a][0]+.7,nodes[a][1]),arrowprops=dict(arrowstyle='->'))
    ax.text(5.3,.2,"D′: far-only bypass; no new dynamic policy",ha='center',fontsize=9)
    ax.text(5.5,3.8,'Fig2 semantic architecture · implementation support is version-specific',ha='center',fontsize=10)
    fig.savefig(out/'Fig2.svg',bbox_inches='tight');fig.savefig(out/'Fig2.pdf',bbox_inches='tight');plt.close(fig)


def render_timing(result,out):
    if result['status']=='blocked': return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pairs=result['pairs'];fig,ax=plt.subplots(figsize=(7,4))
    # Show actual observations only; repetitions are not independent documents.
    ax.scatter(range(len(pairs)),[r['dense_over_method'] for r in pairs])
    ax.axhline(1,color='gray',ls='--');ax.set(xlabel='Paired request/repetition',ylabel='Dense / method (TPOT for T2; wall otherwise)',
        title=result['timing_level']+' descriptive diagnostic · not quality-matched acceptance')
    fig.tight_layout();fig.savefig(out/'timing_diagnostic.svg');fig.savefig(out/'timing_diagnostic.pdf');plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',nargs='?',choices=('plan','inventory','architecture','timing','observations','quality'),default='plan')
    ap.add_argument('--protocol',type=Path,default=Path(__file__).with_name('protocol.unfrozen.json'))
    ap.add_argument('--rows',type=Path);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--plot',action='store_true');args=ap.parse_args()
    if args.out.exists(): ap.error('output exists; choose a fresh directory')
    if args.action in ('observations','quality'):
        import subprocess
        script=ROOT/'exp'/'figures'/('o3_retrieval.py' if args.action=='observations' else 'paired_quality.py')
        command=[sys.executable,str(script),'--protocol',str(args.protocol),'--out',str(args.out)]
        if args.plot: command.append('--plot')
        raise SystemExit(subprocess.call(command))
    # Compute/validate before publication. Failures leave no successful summary.
    p=json.loads(args.protocol.read_text())
    if args.action=='timing':
        if args.rows is None: ap.error('--rows required')
        rows=[json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
        result=summarize(rows,p,digest(args.protocol))
    elif args.action=='inventory': result=inventory(ROOT)
    else: result=plan(p)
    import tempfile
    from exp.figures.o3_retrieval import publish_directory
    args.out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.paper-pending-',dir=args.out.parent))
    if args.action=='architecture': architecture(stage)
    if args.action=='timing' and args.plot: render_timing(result,stage)
    write(stage/'summary.json',result)
    write(stage/'artifact_manifest.json',dict(protocol_sha256=digest(args.protocol),
        script_sha256=digest(Path(__file__)),artifacts=[dict(path=f.name,sha256=digest(f)) for f in sorted(stage.iterdir())]))
    publish_directory(stage,args.out)
    print(json.dumps({'action':args.action,'status':result.get('status',result.get('protocol_status','inventory-only')),'gpu_started':False}))

if __name__=='__main__':
    try: main()
    except (ValueError,KeyError,TypeError,OSError) as exc:
        print('paper workflow blocked: '+str(exc),file=sys.stderr);sys.exit(2)
