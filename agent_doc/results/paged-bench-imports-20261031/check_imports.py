"""Dependency-free check of the migrated benchmark imports; no CUDA claim."""
import ast
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[3]
FILES = ("exp/bench/check_paged_region_kernels.py",
         "exp/bench/check_paged_region_arithmetic_boundary.py")
failures = []
for name in FILES:
    source = ROOT / name
    tree = ast.parse(source.read_text(), filename=name)
    imports = [node for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom)
               and any(alias.name == "PagedRegionLayer" for alias in node.names)]
    if len(imports) != 1:
        failures.append(f"{name}: expected one cache import")
        continue
    module = imports[0].module
    target = ROOT.joinpath(*module.split(".")).with_suffix(".py")
    if not target.is_file():
        failures.append(f"{name}: missing {module}")
    elif not any(isinstance(node, ast.ClassDef) and node.name == "PagedRegionLayer"
                 for node in ast.parse(target.read_text()).body):
        failures.append(f"{name}: missing PagedRegionLayer export")
    else:
        print(f"PASS {name} -> {module}.PagedRegionLayer")
for failure in failures:
    print("FAIL", failure)
raise SystemExit(bool(failures))
