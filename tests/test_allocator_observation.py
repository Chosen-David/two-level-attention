import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('allocator_observation',
    Path(__file__).parents[1]/'exp/bench/allocator_observation.py')
observation=importlib.util.module_from_spec(spec);spec.loader.exec_module(observation)


@pytest.mark.parametrize('snapshot,expected',[
    ({'segments':[{'is_expandable':True}], 'allocator_settings':{'expandable_segments':True}},1),
    ({'segments':[{}]},None),
    ({'segments':[], 'allocator_settings':{'expandable_segments':True}},None),
])
def test_requested_flag_does_not_substitute_for_observed_mode(monkeypatch,snapshot,expected):
    monkeypatch.setenv('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    monkeypatch.setattr(observation.torch.cuda.memory,'_snapshot',lambda:snapshot)
    monkeypatch.setattr(observation.torch.cuda.memory,'get_allocator_backend',lambda:'native')
    monkeypatch.setattr(observation.torch.cuda,'memory_allocated',lambda:64)
    monkeypatch.setattr(observation.torch.cuda,'memory_reserved',lambda:128)
    monkeypatch.setattr(observation.torch.cuda,'memory_stats',lambda:{})
    result=observation.observe_allocator('startup')
    assert result['settings']==snapshot.get('allocator_settings')
    assert result['expandable_segment_count']==expected
    assert result['scope'].startswith('startup')


def test_diagnostic_io_error_preserves_original_failure(monkeypatch,tmp_path,capsys):
    def fail(*args,**kwargs):raise OSError('injected full disk')
    monkeypatch.setattr(observation,'observe_allocator',lambda phase:{})
    monkeypatch.setattr(Path,'write_text',fail)
    original=RuntimeError('original CUDA failure')
    with pytest.raises(RuntimeError) as caught:
        try:
            raise original
        except Exception as exc:
            observation.write_failure_diagnostic(tmp_path/'failure.json',exc)
            raise
    assert caught.value is original
    assert 'injected full disk' in capsys.readouterr().err
