"""Constructor differential evidence; run with python -O, no tensor claims."""
import itertools
import json
import hashlib
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tests'))
from test_explicit_near_page_budget import config, load_constructors
from test_tli_enum_contract import CHOICES

BASE = 'bb173c7300a01cbe720adbc705795a578da5ae75'
SOURCE = ROOT / 'sparse_attn/indexer/tli_indexer.py'
old = subprocess.check_output(['git', 'show', BASE + ':sparse_attn/indexer/tli_indexer.py'], cwd=ROOT).decode()
read_text = Path.read_text

def baseline_read(path, *args, **kwargs):
    return old if path == SOURCE else read_text(path, *args, **kwargs)

def main():
    if sys.flags.optimize < 1:
        raise RuntimeError('run with python -O to reproduce stripped assertions')
    with patch.object(Path, 'read_text', baseline_read):
        before = load_constructors()
    after = load_constructors()
    controls = []
    rejected = []
    with patch.dict('os.environ', TLI_E151C_ORACLE_QUOTA=''):
        for fixed in (False, True):
            geometry = {} if fixed else dict(tli_near_pages=None, tli_near_tokens=None, tli_near_len=None)
            for values in itertools.product(*CHOICES.values()):
                changes = dict(zip(CHOICES, values))
                args = config(**geometry, **changes)
                if vars(before(args)) != vars(after(args)):
                    raise AssertionError((fixed, changes))
                controls.append([fixed, changes])
            for name in CHOICES:
                for value in ('invalid', '', None, 0):
                    args = config(**geometry, **{name:value})
                    before(args)  # Defect: optimized baseline accepts it.
                    try:
                        after(args)
                    except ValueError as exc:
                        if name not in str(exc):
                            raise
                    else:
                        raise AssertionError((fixed,name,value))
                    rejected.append([fixed,name,value])
    matrix = json.dumps([controls,rejected],sort_keys=True,separators=(',',':')).encode()
    print(json.dumps(dict(base=BASE, identical_controls=len(controls),
        accepted_before_rejected_after=len(rejected), input_sha256=hashlib.sha256(matrix).hexdigest(),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest()),indent=2))

if __name__ == '__main__':
    main()
