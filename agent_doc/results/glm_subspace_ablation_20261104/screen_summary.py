import json, glob, os, statistics
base='agent_doc/results/glm_subspace_scan_20261101'
rows=[]
for f in sorted(glob.glob(base+'/summary_*.json')):
    d=json.load(open(f))
    sel=d.get('selector')
    if not sel: continue
    scores=d.get('scores',{})
    row={'selector':sel}
    for k in ['hotpotqa','gov_report','niah_multikey_2']:
        s=scores.get(k,{})
        row[k+'_dense']=s.get('dense')
        row[k+'_sel']=s.get('selected')
        if isinstance(row[k+'_dense'],(int,float)) and isinstance(row[k+'_sel'],(int,float)):
            row[k+'_delta']=row[k+'_sel']-row[k+'_dense']
    rows.append(row)
# print markdown table
print('| selector | hotpot dense | hotpot sel | hotpot delta | gov dense | gov sel | gov delta | multikey2 dense | multikey2 sel | multikey2 delta |')
print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for r in rows:
    def num(x): return f'{x:.2f}' if isinstance(x,(int,float)) else 'NA'
    print(f"| {r['selector']} | {num(r.get('hotpotqa_dense'))} | {num(r.get('hotpotqa_sel'))} | {num(r.get('hotpotqa_delta'))} | {num(r.get('gov_report_dense'))} | {num(r.get('gov_report_sel'))} | {num(r.get('gov_report_delta'))} | {num(r.get('niah_multikey_2_dense'))} | {num(r.get('niah_multikey_2_sel'))} | {num(r.get('niah_multikey_2_delta'))} |")
