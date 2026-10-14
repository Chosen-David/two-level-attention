"""Thin E0.3 figure adapter: bind numeric JSON leaves to explicit figure cells.

Does not rescore, bootstrap, infer run identity or replace existing run-cap /
RULER producer validators. Bindings are reviewed figure inputs, not GPU dispatch.
Only standard library is required. Outputs remain pending independent review.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

COLUMNS = ('figure_id panel_id arm_id reference_id run_mode metric_id unit estimate '
           'ci_low ci_high n_independent_seq aggregation transform source_run_id source_field').split()
IDENTITY = ('figure_id', 'panel_id', 'arm_id', 'metric_id')
NUMBERS = ('estimate', 'ci_low', 'ci_high', 'n_independent_seq')
FIGURES = {'E1', 'E2', 'E3', 'A1', 'A2a', 'A2b', 'A3', 'A4', 'A5', 'A6', 'O1', 'O2', 'O3'}
MODES = {'natural', 'teacher-forced', 'fixed-QK'}
FORBIDDEN = {'UNFROZEN', 'BLOCKED', 'PENDING', 'UNKNOWN', 'NONE', 'NULL', 'TODO', 'TBD', 'MAIN', 'MASTER', 'HEAD', 'LATEST', ''}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def label(value, name):
    need(isinstance(value, str) and value.strip() == value and
         value.upper() not in FORBIDDEN and not any(ord(c) < 32 for c in value),
         'missing/floating ' + name)
    return value


def snapshot(path, expected=None):
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    need(expected is None or sha == expected, f'hash mismatch: {Path(path).name}')
    # Hash and parse the SAME bytes, including when a producer replaces its file.
    return json.loads(raw), sha


def pointer(data, path):
    need(isinstance(path, str) and path.startswith('/'), 'source pointer must be absolute JSON pointer')
    for part in path[1:].split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        if isinstance(data, list):
            need(part.isdigit(), 'invalid list pointer')
            data = data[int(part)]
        else:
            need(isinstance(data, dict), 'pointer traverses scalar')
            data = data[part]
    return data


def key(row):
    return tuple(row[k] for k in IDENTITY)


def numeric(value, name):
    need(type(value) in (int, float) and math.isfinite(value), f'invalid numeric {name}')
    if name == 'n_independent_seq':
        need(type(value) is int and value >= 2, 'CI requires >=2 independent sequences/documents')
    return value


def build_rows(binding_path):
    """No globs/best-file selection: owner explicitly pins each source and cell."""
    spec, binding_sha = snapshot(binding_path)
    need(spec.get('schema') == 'figure-bindings-v1', 'unsupported binding schema')
    need(spec.get('status') == 'frozen', 'UNFROZEN binding')
    need(isinstance(spec.get('sources'), dict) and spec['sources'], 'no sources')
    sources, hashes = {}, {}
    for source_id, source in spec['sources'].items():
        label(source_id, 'source ID')
        # These files are existing verifier/manifest outputs; no new scorer logic.
        label(source.get('run_id'), 'source run_id')
        source_hash = source.get('sha256')
        need(isinstance(source_hash, str) and len(source_hash) == 64 and
             all(c in '0123456789abcdef' for c in source_hash), 'missing/invalid source hash')
        data, sha = snapshot(binding_path.parent / source['path'], source_hash)
        sources[source_id] = (source, data)
        hashes[source_id] = sha
    cells = spec.get('cells')
    need(isinstance(cells, list) and cells, 'no planned cells')
    rows, seen, panels = [], set(), {}
    for cell in cells:
        row = {k: cell[k] for k in COLUMNS if k not in NUMBERS + ('source_field', 'source_run_id')}
        for field, value in row.items():
            if field != 'transform':
                label(value, field)
        need(row['figure_id'] in FIGURES and row['run_mode'] in MODES, 'unknown figure/run mode')
        need(row['transform'] == 'none', 'transform upstream with a verified artifact; no silent scaling')
        need(key(row) not in seen, 'duplicate figure cell')
        seen.add(key(row))
        panel = (row['figure_id'], row['panel_id'])
        signature = (row['run_mode'], row['metric_id'], row['unit'], row['aggregation'], row['reference_id'])
        need(panel not in panels or panels[panel] == signature, 'mixed mode/metric/unit/reference/aggregation in panel')
        panels[panel] = signature
        source, data = sources[cell['source_id']]
        pointers = cell['fields']
        need(set(pointers) == set(NUMBERS), 'all estimate/CI/N source pointers required')
        for field in NUMBERS:
            row[field] = numeric(pointer(data, pointers[field]), field)
        need(row['ci_low'] <= row['ci_high'], 'inverted CI')
        # Bootstrap intervals need not contain the point estimate; do not invent that constraint.
        row['source_run_id'] = source['run_id']
        row['source_field'] = json.dumps({'source_id': cell['source_id'], 'pointers': pointers}, sort_keys=True, separators=(',', ':'))
        rows.append(row)
    return rows, {'binding_sha256': binding_sha, 'source_sha256': hashes,
                  'status': 'structural-pass-pending-independent-review',
                  'scope': 'exact JSON-to-TSV values and declared cell closure only; not E0 provenance/scientific acceptance'}


def tsv_bytes(rows):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=COLUMNS, delimiter='\t', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode()


def validate_tsv(path, expected):
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        need(reader.fieldnames == COLUMNS, 'missing/reordered/extra TSV columns')
        rows = list(reader)
    need(len(rows) == len(expected), 'missing/extra arm or cell')
    actual = {}
    for row in rows:
        need(None not in row and all(v is not None for v in row.values()), 'malformed TSV row')
        need(key(row) not in actual, 'duplicate TSV cell')
        actual[key(row)] = row
    for row in expected:
        need(key(row) in actual, 'missing planned cell')
        candidate = actual[key(row)]
        for field in COLUMNS:
            if field in NUMBERS:
                try:
                    value = float(candidate[field])
                except ValueError as exc:
                    raise ValueError('invalid TSV number: ' + field) from exc
                need(math.isfinite(value) and value == row[field], 'source value differs: ' + field)
            else:
                need(candidate[field] == str(row[field]), 'binding differs: ' + field)
    return len(rows)


def write_new(path, raw):
    """Publish complete bytes without overwriting an existing candidate, even on races."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.figure-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.link(temp, path)  # exclusive destination, no replacement of others' work
    finally:
        os.unlink(temp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bindings', required=True, type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--export', type=Path)
    action.add_argument('--check', type=Path)
    args = parser.parse_args()
    try:
        rows, receipt = build_rows(args.bindings)
        if args.export:
            write_new(args.export, tsv_bytes(rows))
        else:
            validate_tsv(args.check, rows)
        print(json.dumps(dict(receipt, cells=len(rows)), indent=2))
    except (ValueError, KeyError, IndexError, TypeError, OSError) as exc:
        parser.exit(2, f'FIGURE DATA BLOCKED: {exc}\n')


if __name__ == '__main__':
    main()
