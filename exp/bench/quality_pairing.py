"""Strict per-request parity checks; decoded text alone can hide EOS changes."""


def require_graph_capacity(prompt_lengths, max_gen, minimum=16384):
    """Preflight actual tokenized lengths, never the dataset's nominal length."""
    if type(max_gen) is not int or max_gen < 2:
        raise ValueError('parity run requires at least two generated tokens')
    if not prompt_lengths:
        raise ValueError('no prompt lengths')
    for length in prompt_lengths:
        if type(length) is not int or length <= 0:
            raise ValueError('invalid tokenized prompt length')
        if length + max_gen < minimum:
            raise ValueError(f'capacity_below_{minimum}: {length} + {max_gen}')


def compare_predictions(reference, candidate):
    def index(rows):
        result = {}
        for row in rows:
            key = row['_id']
            if key in result:
                raise ValueError(f'duplicate request: {key}')
            tokens = row.get('generated_token_ids')
            if (not isinstance(tokens, list) or not tokens
                    or any(type(t) is not int or t < 0 for t in tokens)):
                raise ValueError(f'missing or invalid token evidence: {key}')
            result[key] = row
        if not result:
            raise ValueError('empty prediction set')
        return result

    left, right = index(reference), index(candidate)
    if left.keys() != right.keys():
        raise ValueError('request sets differ')
    checks = []
    for key, ref in left.items():
        got = right[key]
        if ref['answers'] != got['answers']:
            raise ValueError(f'answer binding differs: {key}')
        checks.append(dict(
            request_id=key,
            tokens_equal=ref['generated_token_ids'] == got['generated_token_ids'],
            text_equal=ref['pred'] == got['pred'],
        ))
    return checks
