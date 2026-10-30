"""CPU-only download contract tests; network is mocked."""
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("fetch_paper", Path(__file__).with_name("fetch_paper.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FetchPaperTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = b"%PDF-1.7\nsynthetic test, not a paper\n"
        (self.root / "PAPER.json").write_text(json.dumps({"pdf_url": "https://example.invalid/paper", "sha256": hashlib.sha256(self.data).hexdigest()}))
        self.dest = self.root / ".cache" / "DuoAttention-2410.10819v1.pdf"
        self.file_patch = patch.object(module, "__file__", str(self.root / "fetch_paper.py"))
        self.file_patch.start()
        self.addCleanup(self.file_patch.stop)

    def test_success_then_cached_no_network(self):
        with patch.object(module.urllib.request, "urlopen", return_value=io.BytesIO(self.data)):
            module.main()
        self.assertEqual(self.dest.read_bytes(), self.data)
        with patch.object(module.urllib.request, "urlopen", side_effect=AssertionError("network")):
            module.main()

    def test_hash_mismatch_not_saved(self):
        with patch.object(module.urllib.request, "urlopen", return_value=io.BytesIO(b"%PDF-other")):
            with self.assertRaises(SystemExit): module.main()
        self.assertFalse(self.dest.exists())

    def test_existing_different_not_overwritten(self):
        self.dest.parent.mkdir()
        self.dest.write_bytes(b"author file")
        with self.assertRaises(SystemExit): module.main()
        self.assertEqual(self.dest.read_bytes(), b"author file")

    def test_non_pdf_not_saved(self):
        with patch.object(module.urllib.request, "urlopen", return_value=io.BytesIO(b"<html>error")):
            with self.assertRaises(SystemExit): module.main()
        self.assertFalse(self.dest.exists())

if __name__ == "__main__": unittest.main()
