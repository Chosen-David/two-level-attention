"""Load the vendored, unmodified official Quest attention module.

The official ``evaluation/quest_attention.py`` is imported by path so adapters
never copy or rewrite the selection algorithm. Keep this module small and
side-effect free: it only resolves the file and executes it once.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_OFFICIAL_PATH = Path(__file__).resolve().parent / "upstream" / "evaluation" / "quest_attention.py"
_MODULE = None


def load_official_module():
    """Return the official ``quest_attention`` module (cached)."""
    global _MODULE
    if _MODULE is None:
        if not _OFFICIAL_PATH.is_file():
            raise RuntimeError(f"official Quest source missing: {_OFFICIAL_PATH}")
        spec = importlib.util.spec_from_file_location("quest_official_attention", _OFFICIAL_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE = module
    return _MODULE
