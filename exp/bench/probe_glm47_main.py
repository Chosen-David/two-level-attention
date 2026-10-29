"""Real-weight dense/full-budget/sparse diagnostic, separate from formal RULER."""
import gc
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import MethodType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparse_attn.info import resolve_treatment_snapshot
from sparse_attn.patches import register_patch
from sparse_attn.ops.far_skip_runtime import resolve_far_skip_modes
from sparse_attn.ops.eager_decoding import eager_decoding_request_guard
from benchmark.RULER.score_ruler import string_match_all


def main():
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    helpers = runpy.run_path(str(ROOT / "exp/bench/test_glm47_d256_gates.py"))
    model_path = "/home/wangyuanshuo02/models/huggingface.co/zai-org/GLM-4.7-Flash"
    data = Path("/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER/32768")
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto").eval()
    native = [layer.self_attn.forward for layer in model.model.layers]
    checks = []

    def generate(ids, steps, teacher=None):
        logits, generated = [], []
        with torch.inference_mode(), eager_decoding_request_guard():
            output = model(input_ids=ids, use_cache=True, logits_to_keep=1)
            for step in range(steps):
                last = output.logits[:, -1].detach().float().cpu()
                logits.append(last)
                generated.append(last.argmax(-1).item())
                if step + 1 < steps:
                    token = teacher[step] if teacher is not None else generated[-1]
                    output = model(input_ids=torch.tensor([[token]], device=ids.device),
                                   past_key_values=output.past_key_values, use_cache=True)
        del output
        gc.collect()
        torch.cuda.empty_cache()
        return generated, logits

    with (outdir / "predictions.jsonl").open("x") as fout:
        for task in ("niah_multikey_2", "niah_multikey_3"):
            row = json.loads((data / f"{task}.jsonl").read_text().splitlines()[0])
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["input"]}], tokenize=False,
                enable_thinking=False, add_generation_prompt=True)
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            for layer, forward in zip(model.model.layers, native):
                layer.self_attn.forward = forward
            reference, ref_logits = generate(ids, 64)
            for mode in ("native", "full_budget", "k128", "k128_far_skip"):
                audit = []
                # Full-budget audit deliberately exceeds the production 2048
                # gather bound. Keep the normal bound for both sparse arms.
                os.environ["TLI_E127_W_UPPER"] = str(ids.shape[-1] + 64) if mode == "full_budget" else "2048"
                if mode == "native":
                    tokens, logits = reference, ref_logits
                else:
                    args = helpers["make_prod_args"]()
                    args.tli_far_skip_backend = "legacy"
                    args.tli_far_skip_cache = "retain"
                    args.tli_dyn_dgate = mode == "k128_far_skip"
                    if mode == "full_budget":
                        args.tia_level1_topk = args.tia_level2_topk = 999999
                        args.tli_sink_tokens = 0
                    resolve_far_skip_modes(args, config=model.config)
                    assert register_patch(model, args, resolve_treatment_snapshot(args)) == 47
                    if mode == "full_budget":
                        for layer in model.model.layers:
                            indexer = layer.self_attn.indexer
                            original = indexer.prepare_mask
                            def checked(self, *a, _original=original, **kw):
                                mask, block = _original(*a, **kw)
                                audit.append(bool(mask.all()))
                                return mask, block
                            indexer.prepare_mask = MethodType(checked, indexer)
                    tokens, logits = generate(ids, 8 if mode == "full_budget" else 64,
                                              teacher=reference if mode == "full_budget" else None)
                pred = tokenizer.decode(tokens, skip_special_tokens=True)
                record = {"task": task, "index": row["index"], "mode": mode,
                          "pred": pred, "answers": row["outputs"],
                          "score": string_match_all([pred], [row["outputs"]]),
                          "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
                if mode == "full_budget":
                    record.update(mask_checks=len(audit), all_masks_full=all(audit),
                                  teacher_forced_argmax_equal=tokens == reference[:8],
                                  max_abs_logits=max((a-b).abs().max().item()
                                                     for a,b in zip(logits, ref_logits)),
                                  prefill_logits_equal=torch.equal(logits[0], ref_logits[0]))
                    checks.append(record["all_masks_full"] and
                                  record["teacher_forced_argmax_equal"] and
                                  record["prefill_logits_equal"] and len(audit) == 47*7)
                if mode == "k128_far_skip":
                    record["skip_far_layers"] = [i for i,l in enumerate(model.model.layers)
                                                 if l.self_attn.indexer.skip_far]
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                print(json.dumps(record, ensure_ascii=False), flush=True)
            del ids, ref_logits
    metadata = {"status": "pending_independent_validation", "base_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "full_budget_checks_pass": all(checks), "full_budget_cases": len(checks),
        "scope": "32K first sample of multikey_2/3, native chat, greedy, 64 tokens; full budget teacher-forced 8 steps",
        "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            "sparse_attn/indexer/tli_indexer.py", "sparse_attn/patches/glm4_moe_lite_attn_patch.py",
            "exp/bench/probe_glm47_main.py")}}
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("MODEL_PROBE_DONE", json.dumps(metadata), flush=True)
    return 0 if all(checks) and len(checks) == 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
