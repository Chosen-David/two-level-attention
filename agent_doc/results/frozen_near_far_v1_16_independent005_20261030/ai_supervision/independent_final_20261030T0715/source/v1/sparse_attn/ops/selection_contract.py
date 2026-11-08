"""CPU support accounting shared by paper receipts; no selector policy changes.

Accounting is per actual query row / head or KV group. Protection is causal and
unioned before counting. This is not a GPU value inspector in the hot path.
"""
def integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name + ' must be a nonnegative integer')
    return value


def support_receipt(selected_ids, *, query_position, length, sink, swa, k_cap):
    for name, value in locals().copy().items():
        if name != 'selected_ids': integer(value, name)
    if query_position >= length:
        raise ValueError('query outside sequence')
    if not isinstance(selected_ids, (list, tuple)) or any(
            type(i) is not int or not 0 <= i <= query_position for i in selected_ids):
        raise ValueError('selected IDs must be causal integer positions')
    selected = set(selected_ids)
    protected = set(range(min(sink, query_position+1))) | set(
        range(max(0, query_position-swa+1), query_position+1))
    missing = protected - selected
    duplicates = len(selected_ids)-len(selected)
    return dict(query_position=query_position,length=length,sink=sink,swa=swa,k_cap=k_cap, k_real=len(selected), protected_real=len(protected),
                selected_protected=len(selected & protected),
                selected_middle=len(selected-protected), duplicate_count=duplicates,
                missing_protected_count=len(missing),
                cap_excess=max(0, len(selected)-max(k_cap, len(protected))),
                support_ids=sorted(selected))


def require_support(receipt):
    if any(receipt[k] for k in ('duplicate_count', 'missing_protected_count', 'cap_excess')):
        raise ValueError('invalid selection support/protection/budget')
    return receipt


def match_support(left, right):
    """Different methods may use different K1. Match final K at every unit."""
    if set(left) != set(right) or not left:
        raise ValueError('support comparison units differ or are empty')
    for value in list(left.values())+list(right.values()): require_support(value)
    return ('realized-support-matched' if all(left[k]['k_real'] == right[k]['k_real'] for k in left)
            else 'cap-matched' if all(left[k]['k_cap'] == right[k]['k_cap'] for k in left)
            else 'unmatched')


def row_support_receipt(shared_ids, *, query_position, length, sink, swa, k_cap):
    """Mirror row-causal/protection union for audit, never change runtime selection.

    Chunk/group end-Q selection can include future IDs relative to an earlier
    row; filtering these prevents future reads, not prefix dependence on Q/K.
    """
    for name, value in dict(query_position=query_position,length=length,sink=sink,swa=swa,k_cap=k_cap).items():
        integer(value,name)
    if query_position >= length or not isinstance(shared_ids,(list,tuple)):
        raise ValueError('invalid query or group selection')
    if any(type(x) is not int or not 0<=x<length for x in shared_ids):
        raise ValueError('invalid group selection IDs')
    ids={x for x in shared_ids if x<=query_position}
    ids.update(range(min(sink,query_position+1)))
    ids.update(range(max(0,query_position-swa+1),query_position+1))
    if not ids: ids.add(query_position)  # existing diagonal empty-row fallback
    return support_receipt(sorted(ids),query_position=query_position,length=length,sink=sink,swa=swa,k_cap=k_cap)
