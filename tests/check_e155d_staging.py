"""Direct-parent T2 differential; synthetic scores, only snapshot location changes."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import types
from unittest.mock import patch

from test_e155d_snapshot import SnapshotTests, grid

ROOT = Path(__file__).resolve().parents[1]
BASE = '143ff2cf39fe5012d929afe81d71229f054c3f4d'
source = subprocess.check_output(
    ['git', 'show', BASE + ':exp/trace/analyze_e155d_grid.py'], cwd=ROOT)
baseline = types.ModuleType('staging_baseline')
baseline.__file__ = str(ROOT / 'exp/trace/analyze_e155d_grid.py')
exec(compile(source, '<staging-baseline>', 'exec'), baseline.__dict__)


def normalized(verdict):
    verdict['meta'].pop('generated')
    for row in verdict['ranking'] + list(verdict['ctrl_singlepool'].values()):
        for task, path in row['files'].items():
            snapshot = Path(path)
            if grid.file_sha(snapshot) != row['file_sha256'][task]:
                raise AssertionError('published snapshot hash mismatch')
            row['files'][task] = '/'.join(snapshot.parts[-3:])
    return verdict


digest = hashlib.sha256()
for seed in range(24):
    case = SnapshotTests()
    case.setUp()
    try:
        rng = random.Random(seed)
        for path in case.paths.values():
            rows = [dict(_id=i, pred=rng.choice([0, .125, .5, .75, 1]), answers=['yes'])
                    for i in range(2 + seed % 5)]
            rng.shuffle(rows)
            raw = ''.join(json.dumps(row) + '\n' for row in rows).encode()
            digest.update(raw)
            path.write_bytes(raw)
        with contextlib.ExitStack() as stack:
            for name in ('COLLECT_ROOT', 'STAGING', 'OUT', 'HOSTS', 'MIN_ROWS'):
                stack.enter_context(patch.object(baseline, name, getattr(grid, name)))
            stack.enter_context(patch.object(baseline.subprocess, 'run', side_effect=case.eval_stub))
            stack.enter_context(patch.object(sys, 'argv', ['grid', '--bootstrap', '40']))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            baseline.main()
        old = normalized(json.loads(Path(grid.OUT).read_text()))
        new = normalized(case.run_main())
        if old != new:
            raise AssertionError(f'full-entry difference at seed {seed}')
    finally:
        case.doCleanups()
print('PASS: 24 direct-parent T2 comparisons; only generated time/staging prefix '
      'normalized; input_sha256=' + digest.hexdigest())
