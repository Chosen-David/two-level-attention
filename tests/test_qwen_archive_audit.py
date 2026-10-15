import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

spec = importlib.util.spec_from_file_location(
    "archive_audit", Path(__file__).resolve().parents[1] / "exp/paper/audit_qwen_archive.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ArchiveAuditTest(unittest.TestCase):
    def test_real_index_missing_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = audit.audit(Path(tmp))
        self.assertEqual(result["exit_code"], 2)
        self.assertEqual(len(result["files"]), 12)
        self.assertFalse(result["paper_ready"])

    def fixture(self, root):
        for name in audit.NAMES:
            path = root / name
            path.parent.mkdir(exist_ok=True)
            if path.suffix == ".json":
                path.write_text(json.dumps({"S": 320, "n_layers": 36, "prompt": path.parent.name}))
            else:
                positions = sorted(set(range(64, 320)) | {int(320 * (i + .5) / 16) for i in range(16)})
                torch.save({"k": torch.zeros(320, 2, 128),
                            "q": torch.zeros(len(positions), 4, 128),
                            "qpos": torch.tensor(positions), "S": 320}, path)
        index = root / "index.txt"
        index.write_text("\n".join(f"{audit.digest(root / n, 'md5')}  ./{n}" for n in audit.NAMES))
        return index

    def test_valid_synthetic_files_do_not_prove_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = audit.audit(root, self.fixture(root))
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["paper_ready"])
        self.assertEqual(result["provenance_status"], "unresolved")
        self.assertEqual(result["files"][1]["schema"]["anchor_positions"], list(range(10, 320, 20)))

    def test_checksum_corruption_blocks_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = self.fixture(root)
            (root / audit.NAMES[1]).write_bytes(b"bad tensor")
            result = audit.audit(root, index)
        self.assertEqual(result["exit_code"], 3)
        self.assertEqual(result["files"][1]["status"], "checksum-mismatch")

    def test_schema_and_query_identity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            path = root / audit.NAMES[1]
            original = torch.load(path, weights_only=True)
            for field, value in (("qpos", original["qpos"].flip(0)),
                                 ("k", torch.full_like(original["k"], float("nan"))),
                                 ("q", torch.zeros(len(original["qpos"]), 3, 128)),
                                 ("S", 319)):
                with self.subTest(field=field):
                    torch.save(dict(original, **{field: value}), path)
                    with self.assertRaises(ValueError):
                        audit.tensor_schema(path, {"S": 320})

    def test_duplicate_or_missing_checksum_entry_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = self.fixture(root)
            lines = index.read_text().splitlines()
            for content in (lines[:-1], lines + lines[:1]):
                index.write_text("\n".join(content))
                with self.assertRaises(ValueError):
                    audit.expected_checksums(index)


if __name__ == "__main__":
    unittest.main()
