"""Exercise real frozen treatment producer -> T2 consumer, without tensors."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from test_e155d_geometry import DEFAULTS, grid
from test_vendor_frozen_manifest import load_info


def main():
    info = load_info()  # Only torch import removed; any tensor access raises.
    count, rejected = 0, 0
    digest = hashlib.sha256()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for nb in sorted(grid.NB_AXIS):
            for nt in sorted(grid.NT_AXIS, key=str):
                for nl in sorted(grid.NL_AXIS):
                    tag = f'pred_nb{nb}nt{nt}nl{nl}'
                    if grid.parse_tag(tag) is None:
                        continue
                    args = dict(DEFAULTS, method='tli', tli_enable_layer_skip=False,
                                tli_alpha=None, tli_beta=None, tli_gamma=None,
                                tli_near_pages=nb, tli_near_tokens=nt, tli_near_len=nl)
                    args = SimpleNamespace(**args)
                    snap = info.resolve_treatment_snapshot(args)
                    name = info.get_method_name_with_info(args, snap)
                    text = info.get_treatment_manifest_json(args, snap)
                    digest.update(text.encode('utf-8'))
                    path = root / f'qasper-{name}-fixed.jsonl'
                    # Only metadata is consumed by this helper, no scoring.
                    sidecar = Path(str(path) + '.tli_manifest.json')
                    sidecar.write_text(text)
                    result = grid.cell_treatment_hashes(tag, {'qasper': str(path)})
                    if result != {'qasper': hashlib.sha256(text.encode()).hexdigest()}:
                        raise AssertionError('producer/consumer binding mismatch')
                    count += 1
                    wrong = f'pred_nb{4 if nb != 4 else 8}nt{nt}nl{nl}'
                    # Deliberately choose only another valid registered label.
                    if grid.parse_tag(wrong):
                        try:
                            grid.cell_treatment_hashes(wrong, {'qasper': str(path)})
                        except ValueError:
                            rejected += 1
                        else:
                            raise AssertionError('mislabeled producer accepted')
        args = SimpleNamespace(**dict(DEFAULTS, method='tli', tli_enable_layer_skip=False,
                                      tli_alpha=0., tli_beta=0., tli_gamma=None))
        snap = info.resolve_treatment_snapshot(args)
        name = info.get_method_name_with_info(args, snap)
        text = info.get_treatment_manifest_json(args, snap)
        digest.update(text.encode('utf-8'))
        path = root / f'qasper-{name}-fixed.jsonl'
        Path(str(path) + '.tli_manifest.json').write_text(text)
        grid.cell_treatment_hashes('pred_ctrl_singlepool', {'qasper': str(path)})
        count += 1
    if count != 87 or rejected == 0:
        raise AssertionError((count, rejected))
    print(f'PASS: {count} real producer geometry manifests; {rejected} wrong labels rejected; '
          f'manifest_input_sha256={digest.hexdigest()}')


if __name__ == '__main__':
    main()
