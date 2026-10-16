"""Exact whole-bin diagnostic arithmetic; never a token TopK oracle.

A selected bin means selecting ALL its tokens. Greedy mass order skips bins
that do not fit. This is feasible but neither knapsack-optimal nor a bound.
No within-bin distribution is assumed. Regions must align to whole bins.
"""
import math


def integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name + ' must be a nonnegative integer')
    return value


def finite_nonnegative(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('mass must be finite and nonnegative')
    return float(value)


def validated_bins(observation):
    q = integer(observation['q_pos'], 'q_pos')
    rb = observation['region_bounds']
    lo, hi, near = [integer(rb[k], k) for k in ('mid_lo', 'mid_hi', 'near_lo')]
    if not (0 <= lo <= near <= hi <= q + 1) or rb['length'] != q + 1:
        raise ValueError('invalid region bounds')
    if lo != rb['sink_hi'] or hi != rb['swa_lo']:
        raise ValueError('mid must exclude protected sink and SWA')
    result = []
    for raw in observation['bins']:
        dl, dh = integer(raw['d_lo'], 'd_lo'), integer(raw['d_hi'], 'd_hi')
        start, end = q - dh, q - dl + 1
        n = integer(raw['n_tok'], 'n_tok')
        if not (lo <= start < end <= hi) or end - start != n:
            raise ValueError('bin width/causal/region mismatch')
        if start < near < end:
            raise ValueError('bin straddles near boundary; token masses unavailable')
        result.append(dict(start=start, end=end, n_tok=n,
                           mass=finite_nonnegative(raw['mass']),
                           region='far' if end <= near else 'near'))
    result.sort(key=lambda b: b['start'])
    cursor = lo
    for b in result:
        if b['start'] != cursor:
            raise ValueError('bins must be disjoint and contiguous')
        cursor = b['end']
    if cursor != hi:
        raise ValueError('bins do not cover complete mid region')
    # Independently stored region totals provide an integrity cross-check.
    for region in ('far', 'near'):
        expected = finite_nonnegative(observation['mass_' + region])
        actual = math.fsum(b['mass'] for b in result if b['region'] == region)
        if not math.isclose(actual, expected, rel_tol=2e-5, abs_tol=2e-7):
            raise ValueError('bin mass disagrees with recorded region total')
    return result


def select_whole_bins(bins, budget):
    integer(budget, 'budget')
    for b in bins:
        if integer(b['n_tok'], 'n_tok') == 0:
            raise ValueError('nonempty bin required')
        finite_nonnegative(b['mass'])
    chosen, used = set(), 0
    # Deterministic tie order is input order (validated bins are position-sorted).
    for i in sorted(range(len(bins)), key=lambda i: (-bins[i]['mass'], i)):
        if bins[i]['n_tok'] <= budget - used:
            chosen.add(i)
            used += bins[i]['n_tok']
    return chosen, used


def ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def diagnose(observation, budget=2048):
    bins = validated_bins(observation)
    selected, used = select_whole_bins(bins, budget)
    masses = {r: math.fsum(b['mass'] for b in bins if b['region'] == r)
              for r in ('far', 'near')}
    capture = {r: ratio(math.fsum(b['mass'] for i,b in enumerate(bins)
                                    if i in selected and b['region'] == r), masses[r])
               for r in masses}
    return dict(bins=bins, selected=sorted(selected), selected_tokens=used,
                budget=budget, available_tokens=sum(b['n_tok'] for b in bins),
                far_share=ratio(masses['far'],sum(masses.values())),
                far_capture=capture['far'], near_capture=capture['near'])


def finite_summary(values):
    valid = [v for v in values if v is not None and math.isfinite(v)]
    return dict(mean=sum(valid)/len(valid) if valid else None,
                n_valid=len(valid), n_total=len(values))


def paired_comparison(near, far):
    if len(near) != len(far):
        raise ValueError('paired length mismatch')
    pairs=[(n,f) for n,f in zip(near,far)
           if n is not None and f is not None and math.isfinite(n) and math.isfinite(f)]
    wins=sum(n > f for n,f in pairs)
    return dict(near_gt_far=wins, n_pairs=len(pairs), n_total=len(near),
                fraction=wins/len(pairs) if pairs else None)
