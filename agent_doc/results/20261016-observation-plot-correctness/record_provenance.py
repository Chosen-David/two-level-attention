"""Bind this one diagnostic revision and retain recoverable baseline identities."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
BASE='47e4ab2f004d1eabce85c63849a2651123f6f45c'
FIG='agent_doc/guide/paper_figs/figs/'
MP='agent_doc/guide/paper_figs/observation/e140_validity.json'

def digest(b):return hashlib.sha256(b).hexdigest()
def current(path):return dict(path=path,sha256=digest((ROOT/path).read_bytes()))
def main():
    old=json.loads((ROOT/MP).read_text())
    if old['schema']!='e140-historical-validity-v3':raise ValueError('expected unmodified baseline manifest')
    # Preserve unchanged raw bytes; generated TSV has a new role below.
    raw=[e for e in old['unchanged_inputs'] if e['path'].startswith('exp/')]
    for entry in raw:
        if entry['status']!='missing' and digest((ROOT/entry['path']).read_bytes())!=entry['sha256']:
            raise ValueError('raw input changed')
    from exp.paper import e140_asset_provenance as a
    previous=dict(git_commit=BASE,files=[])
    for path in [FIG+'O_observation.'+ext for ext in ('png','pdf','svg')]+[FIG+'O_observation_figure_data.tsv',FIG+'fig_o_observation.py',MP]:
        b=subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT)
        previous['files'].append(dict(path=path,sha256=digest(b),git_blob=subprocess.check_output(['git','rev-parse',BASE+':'+path],cwd=ROOT,text=True).strip()))
    new=dict(old)
    new['prior_asset_sets']=old['prior_asset_sets']+[dict(git_commit=BASE,current_assets=old['current_assets'],unchanged_inputs=old['unchanged_inputs'],redesign=old.get('redesign'),retired_reason='Whole-bin budget overshoot and unsupported oracle/gate claims; raw preserved.')]
    new['historical_snapshots']=old.get('historical_snapshots',[])+[previous]
    new.update(schema=a.SCHEMA,status=a.STATUS,unchanged_inputs=raw,
               current_assets=[dict(**current(FIG+'O_observation.'+ext),status=a.STATUS) for ext in ('svg','png','pdf')],
               derived_outputs=[current(FIG+'O_observation_figure_data.tsv'),current(FIG+'O_observation_diagnostic.json')],
               renderer_sources=[current(p) for p in (FIG+'fig_o_observation.py','exp/paper/observation_bins.py','exp/paper/e140_asset_provenance.py')],
               annotation=dict(warning=a.WARNING,detail=a.DETAIL,reference=a.REFERENCE),
               source_provenance='Existing O1/O3 aggregates only. Whole-bin heuristic is budget-feasible, not token TopK or a bound. E6 blocked. CPU validation does not accept scientific data.',
               redesign=dict(protocol='feasible-whole-bin-mass-rank-v1',token_budget_cap=2048,raw_source='exp/trace/results/e140_o_trace_v2.json',e6='blocked; no profile or skip data consumed',o4='not used; prior partial-page quarantine retained',scientific_acceptance=False))
    (ROOT/MP).write_text(json.dumps(new,indent=2)+'\n')
    a.verify(ROOT)

if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    main()
