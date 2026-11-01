"""One-shot static gate calibration for frozen-region NFA.

Runs the existing request-time frozen prefill on one calibration sample and
writes the per-layer near/far skip decisions as a tracked static-gate artifact.
The main pred_ruler entrypoint then reads this artifact via --nfa_static_gate.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig, AutoModelForCausalLM, AutoTokenizer)

from benchmark.RULER import pred_ruler  # noqa: E402
from sparse_attn.info import resolve_treatment_snapshot  # noqa: E402
from sparse_attn.patches import register_patch  # noqa: E402
from sparse_attn.policies.static_gate import (  # noqa: E402
    default_static_gate_path, save_static_gate, static_gate_from_audit)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate a static near/far skip gate from one sample")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-num", type=int, default=1)
    parser.add_argument("--output", type=str, default=None,
                        help="override default static gate path")
    parser = pred_ruler.add_sparse_attn_args(parser)
    parser.add_argument("--nfa_prefill_frozen_regions", action="store_true")
    parser.add_argument("--nfa_frozen_region_mode",
                        choices=["full", "far", "near", "dual"], default="dual")
    parser.add_argument("--nfa_frozen_region_epsilon", type=float, default=0.05)
    parser.add_argument("--nfa_frozen_region_statistic",
                        choices=["mean", "p95"], default="mean")
    parser.add_argument("--nfa_frozen_region_rows", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    frozen_runtime = pred_ruler._configure_frozen_regions(args, config)
    if frozen_runtime is None:
        raise SystemExit("calibration requires --nfa_prefill_frozen_regions "
                         "or a supported GLM NFA configuration")
    snapshot = resolve_treatment_snapshot(args) if args.method == "nfa" else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto").eval()
    n_patched = register_patch(model, args, snapshot)
    pred_ruler._validate_frozen_model(model)
    if n_patched != model.config.num_hidden_layers:
        raise ValueError("calibration requires NFA patches on every layer")
    model._frozen_nfa = True

    data_file = os.path.join(args.data_root, str(args.context_length),
                             f"{args.task}.jsonl")
    with open(data_file, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if not rows:
        raise SystemExit(f"empty calibration data: {data_file}")
    if args.max_num and len(rows) > args.max_num:
        rows = rows[:args.max_num]
    if args.sample_index >= len(rows):
        raise SystemExit(
            f"sample-index={args.sample_index} out of range {len(rows)}")
    row = rows[args.sample_index]

    input_ids = tokenizer(row["input"], truncation=False,
                          return_tensors="pt").input_ids.to(next(model.parameters()).device)
    gate, geometry = frozen_runtime
    with torch.no_grad():
        output, audit = pred_ruler.prefill_frozen_request(model, input_ids, gate, geometry)
    del output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_type = config.model_type
    path = args.output or default_static_gate_path(model_type)
    source = (f"task={args.task} sample_index={args.sample_index} "
              f"context_length={args.context_length}")
    decision = static_gate_from_audit(
        model_type, model.config.num_hidden_layers, audit,
        {"mode": args.nfa_frozen_region_mode,
         "epsilon": args.nfa_frozen_region_epsilon,
         "rows": args.nfa_frozen_region_rows,
         "statistic": args.nfa_frozen_region_statistic},
        source=source)
    save_static_gate(decision, path=path)
    print(f"[static-gate] calibrated from {source}")
    print(f"[static-gate] wrote {path}")
    print(f"[static-gate] skip_far={sum(decision.skip_far)} "
          f"skip_near={sum(decision.skip_near)}")


if __name__ == "__main__":
    main()
