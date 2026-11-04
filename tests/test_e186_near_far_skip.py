import torch
from exp.trace.e186_near_far_skip import region_masks, measure_regions


def test_region_partition_and_equal_token_control():
    masks=region_masks(16400)
    assert not (masks['near']&masks['far']).any()
    assert masks['near'].sum()==masks['far_equal_near_tokens'].sum()
    assert not (masks['far_equal_near_tokens']&~masks['far']).any()
    assert not masks['near'][:4].any() and not masks['far'][-128:].any()


def test_region_outputs_match_explicit_removed_softmax():
    torch.manual_seed(186)
    q=torch.randn(1,4,1,8);k=torch.randn(1,2,9000,8);v=torch.randn_like(k)
    actual=measure_regions(q,k,v)
    kk=k.repeat_interleave(2,1);vv=v.repeat_interleave(2,1)
    scores=q@kk.transpose(-1,-2)/8**.5;probs=scores.softmax(-1);full=probs@vv
    for name,mask in region_masks(9000).items():
        expected=scores.masked_fill(mask,-torch.inf).softmax(-1)@vv
        assert abs(actual[name]['rel_l2']-float((expected-full).norm()/full.norm()))<1e-6
    assert abs(actual['far']['mass']+actual['near']['mass']+actual['protected_mass']-1)<1e-6


def test_query_relative_prefill_regions_and_mass_conservation():
    # Query positions 8998/8999: the near window must move with each query.
    q=torch.zeros(1,4,2,8);k=torch.zeros(1,2,9000,8)
    r=measure_regions(q,k,prefill=True,start=8998)
    expected={name:sum(float(region_masks(n)[name].sum())/n for n in [8999,9000])/2
              for name in ['near','far']}
    for name in ['near','far']:assert abs(r['moving_'+name]-expected[name])<1e-6
    assert abs(r['moving_near']+r['moving_far']+r['moving_protected']-1)<1e-6
