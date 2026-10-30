"""Fail-closed complete matrix scoring; raw data remain pending independent review."""
import argparse
import json
import os
from pathlib import Path
os.environ['TLI_SCORER_BACKEND']='difflib'
import numpy as np
from benchmark.LongBench.eval import sample_score
from benchmark.RULER.score_ruler import string_match_all
from benchmark.LongBench.glm_region_matrix import ARMS
from benchmark.LongBench.prefill_frozen_skip import atomic_json,SCREEN_TASKS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run');args=ap.parse_args();root=Path(args.run)
    protocol=json.loads((root/'protocol.json').read_text());identity=protocol['identity'];expected=protocol['protocol']['expected']
    scores={task:{arm:[] for arm in ARMS} for task in expected}
    all_files=list((root/'requests').glob('*/*.json'))
    assert len(all_files)==sum(expected.values()),'missing/extra inputs'
    for task,n in expected.items():
        paths=sorted((root/'requests'/task).glob('*.json'))
        assert {p.stem for p in paths}=={f'{i:04d}' for i in range(n)},f'{task}: row closure'
        for p in paths:
            record=json.loads(p.read_text())
            assert record['identity']==identity and record['status']=='complete',str(p)
            assert record['task']==task and record['row']==int(p.stem),str(p)
            assert set(record['arms'])==set(ARMS),str(p)
            for arm in ARMS:
                result=record['arms'][arm]
                # These five tasks do not consume all_classes in scoring.
                score=(string_match_all([result['prediction']],[record['answers']]) if task=='niah_multikey_2' else
                       100*sample_score('repobench' if task=='repobench-p' else task,result['prediction'],record['answers'],[]))
                assert np.isfinite(score) and 0<=score<=100
                assert abs(score-result['score_pct'])<1e-6,(p,arm,'stored metric mismatch')
                scores[task][arm].append(score)
    means={t:{a:float(np.mean(v)) for a,v in arms.items()} for t,arms in scores.items()}
    if set(SCREEN_TASKS)<=set(means):means['AVG5']={a:float(np.mean([means[t][a] for t in SCREEN_TASKS])) for a in ARMS}
    rng=np.random.default_rng(42);draws={};deltas={}
    for task,arms in scores.items():
        ix=rng.integers(0,len(arms['dense']),size=(5000,len(arms['dense'])))
        draws[task]={arm:np.asarray(values)[ix].mean(1) for arm,values in arms.items()}
    if set(SCREEN_TASKS)<=set(draws):draws['AVG5']={a:np.mean([draws[t][a] for t in SCREEN_TASKS],axis=0) for a in ARMS}
    for task,values in draws.items():
        deltas[task]={}
        for baseline in ('dense','full'):
            for arm in ARMS:
                if arm==baseline:continue
                delta=values[arm]-values[baseline]
                deltas[task][f'{arm}-vs-{baseline}']=dict(delta_pp=means[task][arm]-means[task][baseline],
                    paired_bootstrap_95ci_pp=np.quantile(delta,[.025,.975]).tolist())
    atomic_json(root/'summary_pending_validation.json',dict(status='pending_independent_verification',
        identity=identity,expected=expected,means_pct=means,deltas=deltas,
        bootstrap=dict(replicates=5000,seed=42,AVG5='equal task means; task-stratified paired resampling'),
        interpretation='No equivalence/noninferiority claim from an interval containing zero.'))

if __name__=='__main__':main()
