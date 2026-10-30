import json
import pytest
from benchmark.LongBench.analyze_frozen_skip import analyze


def build_run(tmp_path, tasks=None):
    tasks=tasks or ['qasper','hotpotqa','musique','gov_report','repobench-p']
    protocol=dict(selected_rows={t:[0,1] for t in tasks},datasets=tasks,thresholds=[.1])
    (tmp_path/'manifest_rank0.json').write_text(json.dumps(dict(identity='fixture',protocol=protocol)))
    for task in tasks:
        directory=tmp_path/'requests'/task;directory.mkdir(parents=True)
        for i in [0,1]:
            arms={}
            for name,score in [('full',1.),('far_0.1',1.),('dual_0.1',.5)]:
                arms[name]=dict(score=score,gate_counts=dict(keep=36,near=0,far=0,both=0),
                    decisions={'0':dict(skip_near=False,skip_far=False)},
                    frozen_kv=dict(kv_bytes=100),prefill_full_kv=dict(kv_bytes=100),generated_ids=[int(score*10)])
            (directory/f'{i}.json').write_text(json.dumps(dict(identity='fixture',dataset=task,row_index=i,
                status='complete',arms=arms)))


def test_task_macro_pairing_and_bootstrap_known_constant_drop(tmp_path):
    build_run(tmp_path)
    report=analyze(tmp_path)
    assert report['status']=='complete' and report['completed_requests']==10
    assert report['arms']['full']['avg5']==100
    assert report['arms']['dual_0.1']['avg5']==50
    comparison=report['comparisons']['dual_0.1-vs-far_0.1']['avg5']
    assert comparison['delta_pp']==-50 and comparison['ci95_pp']==[-50,-50]


def test_duplicate_rows_cannot_inflate_completion(tmp_path):
    build_run(tmp_path)
    original=tmp_path/'requests/qasper/0.json'
    original.with_name('duplicate.json').write_text(original.read_text())
    with pytest.raises(ValueError,match='duplicate request identity'):
        analyze(tmp_path)


def test_sixteen_task_macro_is_not_mislabeled_avg5(tmp_path):
    from benchmark.LongBench.prefill_frozen_skip import V1_TASKS
    build_run(tmp_path, V1_TASKS)
    result=analyze(tmp_path)
    assert result['completed_requests']==32
    assert result['arms']['full']['avg16']==100
    assert result['arms']['dual_0.1']['avg16']==50
    assert 'avg5' not in result['arms']['full']
    assert result['comparisons']['dual_0.1-vs-full']['avg16']['ci95_pp']==[-50,-50]
    assert 'AVG16' in (tmp_path/'REPORT.md').read_text()
