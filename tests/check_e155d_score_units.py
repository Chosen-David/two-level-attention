"""Differential check against the published pre-unit-conversion implementation.

Run from any directory: python tests/check_e155d_score_units.py
Uses git history and the standard library; does not import model dependencies.
"""
import ast
import importlib.util
from pathlib import Path
import random
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "exp/trace/analyze_e155d_grid.py"
BASELINE = "4ea101c9eca2ef9c4a852db7a5dfbd04db061c6b"
spec = importlib.util.spec_from_file_location("e155d_unit_candidate", ROOT / SCRIPT)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


def main():
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:{SCRIPT}"], cwd=ROOT, text=True)
    namespace = {"__file__": str(ROOT / SCRIPT), "__name__": "unit_reference"}
    exec(compile(source, "published_baseline", "exec"), namespace)
    # Observe pre-rounding outputs of the unchanged baseline algorithm.
    namespace["round"] = lambda value, digits: value
    rng = random.Random(90)
    count = 0
    for size in (1, 3, 17, 41):
        for seed in (0, 7, 20261020):
            for string_ids in (False, True):
                a, b = {}, {}
                for index, task in enumerate(candidate.AVG5_TASKS):
                    ids = [str(i) if string_ids else i for i in range(size + index)]
                    a[task] = {key: rng.random() for key in ids}
                    b[task] = {key: rng.random() for key in reversed(ids)}
                old = namespace["paired_bootstrap"](a, b, B=200, seed=seed)
                new = candidate.paired_bootstrap(a, b, B=200, seed=seed)
                expected = dict(old, point_diff=round(100 * old["point_diff"], 4),
                                ci95=[round(100 * v, 4) for v in old["ci95"]],
                                unit="score_points")
                if new != expected:
                    raise AssertionError((size, seed, string_ids, old, new))
                count += 1
    # Execute the actual aggregate scorer's AST without unavailable dependencies.
    # Only the per-example metric is replaced by numeric fixture values.
    tree = ast.parse((ROOT / "benchmark/LongBench/eval.py").read_text())
    scorer = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "scorer")
    env = {"sample_score": lambda dataset, prediction, answers, classes: prediction}
    exec(compile(ast.Module(body=[scorer], type_ignores=[]), "actual_eval_scorer", "exec"), env)
    expected = env["scorer"]("hotpotqa", [.75], [[]], []) - env["scorer"](
        "hotpotqa", [.5], [[]], [])
    values = lambda x: {t: {"sample": x} for t in candidate.AVG5_TASKS}
    actual = candidate.paired_bootstrap(values(.75), values(.5), B=40)
    if actual["point_diff"] != expected or actual["ci95"] != [expected, expected]:
        raise AssertionError((actual, expected))
    print(f"PASS: {count} baseline differential cases; actual aggregate scorer scale")


if __name__ == "__main__":
    main()
