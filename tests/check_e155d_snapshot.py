"""Full-entry differential against frozen pre-fix code; synthetic scorer only."""
import contextlib
import hashlib
import io
import json
import random
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

from test_e155d_snapshot import SnapshotTests, grid

ROOT = Path(__file__).resolve().parents[1]
BASE = 'b418e220aae5cd86d4ba4ce1833ea79d4ad6bcde'
source = subprocess.check_output(['git', 'show', BASE + ':exp/trace/analyze_e155d_grid.py'], cwd=ROOT)
baseline = types.ModuleType('snapshot_baseline')
baseline.__file__ = str(ROOT / 'exp/trace/analyze_e155d_grid.py')
exec(compile(source, '<snapshot-baseline>', 'exec'), baseline.__dict__)


def normalized(verdict):
    verdict['meta'].pop('generated')
    for row in verdict['ranking'] + list(verdict['ctrl_singlepool'].values()):
        row.pop('files', None)
        row.pop('file_sha256', None)
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
            data = ''.join(json.dumps(row) + '\n' for row in rows).encode()
            digest.update(data)
            path.write_bytes(data)
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
print('PASS: 24 full-entry baseline differential cases; input_sha256=' + digest.hexdigest())
