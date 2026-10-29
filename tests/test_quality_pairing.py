import pytest

from exp.bench.quality_pairing import compare_predictions


def row(key='a', tokens=None):
    return dict(_id=key, pred='answer', answers=['answer'],
                generated_token_ids=[7, 2] if tokens is None else tokens)


def test_special_token_difference_is_not_hidden_by_decoding():
    check, = compare_predictions([row()], [row(tokens=[7, 3])])
    assert check['text_equal'] and not check['tokens_equal']


def test_pair_by_identity_not_output_order():
    checks = compare_predictions([row('a'), row('b')], [row('b'), row('a')])
    assert all(c['tokens_equal'] for c in checks)


@pytest.mark.parametrize('left,right', [
    ([], []), ([row(), row()], [row()]), ([row()], [row('b')]),
    ([row()], [row(tokens=[])]), ([row()], [row(tokens=[True])]),
    ([row()], [row(tokens=[-1])]),
    ([row()], [dict(_id='a', pred='answer', answers=['answer'])]),
    ([row()], [dict(row(), answers=['different'])]),
])
def test_missing_duplicate_or_incompatible_evidence_rejected(left, right):
    with pytest.raises(ValueError):
        compare_predictions(left, right)
