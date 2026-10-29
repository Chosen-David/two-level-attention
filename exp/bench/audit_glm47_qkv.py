"""Compare attention consumers on identical real GLM decode Q/K/V tensors."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from sparse_attn.ops.eager_decoding import (
    _eager_gather_triton, _eager_decoding_attn_dense,
    eager_decoding_attn, eager_decoding_request_guard,
)
import sparse_attn.ops.eager_decoding as consumer


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    model_path = "/home/wangyuanshuo02/models/huggingface.co/zai-org/GLM-4.7-Flash"
    source = Path("/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER/32768/niah_multikey_2.jsonl")
    row = json.loads(source.read_text().splitlines()[0])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": row["input"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                                                device_map="auto").eval()
    helpers = runpy.run_path(str(ROOT / "exp/bench/test_glm47_d256_gates.py"))
    backend = model.config._attn_implementation
    original = ALL_ATTENTION_FUNCTIONS[backend]
    records = []
    torch.backends.cuda.matmul.allow_tf32 = False

    def error(a, b):
        a, b = a.float(), b.float()
        return {"max_abs": (a-b).abs().max().item(),
                "relative_l2": ((a-b).norm()/b.norm().clamp_min(1e-12)).item(),
                "allclose_atol_rtol_1e_2": torch.allclose(a,b,atol=1e-2,rtol=1e-2),
                "finite": bool(torch.isfinite(a).all())}

    def audited(module, q, k, v, mask, **kwargs):
        result = original(module, q, k, v, mask, **kwargs)
        if q.shape[2] != 1 or module.layer_idx not in (0, 23, 46):
            return result
        scale = kwargs["scaling"]
        assert q.shape[1] == k.shape[1] == 20 and q.shape[-1] == 256
        qt, kt, vt = (x.transpose(1,2) for x in (q,k,v))
        length = k.shape[2]
        cu = torch.tensor([0,length], device=q.device, dtype=torch.int32)
        logits = (q.float() @ k.float().transpose(-1,-2)) * scale
        ref = (logits.softmax(-1) @ v.float()).transpose(1,2)
        all_mask = torch.ones(1,1,20,length,device=q.device,dtype=torch.bool)
        dense = _eager_decoding_attn_dense(qt,kt,vt,all_mask,1,cu,softmax_scale=scale)
        # Direct kernel wrapper deliberately bypasses the dispatcher's
        # high-budget dense fallback, so this actually tests all-token gather.
        gathered = _eager_gather_triton(qt[0],kt[0],vt[0],all_mask[0,0],
                                       False,1,length,cu,scale)
        record = {"layer":module.layer_idx,"tokens":length,"scale":scale,
                  "native_backend":backend,"native_vs_fp32":error(result[0],ref),
                  "dense_vs_fp32":error(dense,ref),
                  "direct_full_triton_vs_fp32":error(gathered,ref)}
        indexer = helpers["make_indexer"]()
        pos = torch.tensor([length-1],device=q.device)
        chosen, block = indexer.prepare_mask(qt,pos,kt,cu,softmax_scale=scale)
        token_mask = chosen.transpose(1,2)  # [B,H,1,S], matching native QK scores
        sparse_ref = (logits.masked_fill(~token_mask,-torch.inf).softmax(-1) @ v.float()).transpose(1,2)
        calls = []
        saved = consumer._eager_gather_triton
        def traced(*a,**kw):
            calls.append(True)
            return saved(*a,**kw)
        consumer._eager_gather_triton = traced
        try:
            actual = eager_decoding_attn(qt,kt,vt,chosen,block,cu,softmax_scale=scale)
        finally:
            consumer._eager_gather_triton = saved
        record.update(production_triton_calls=len(calls),
                      selected_min=int(chosen.sum(-1).min()),
                      selected_max=int(chosen.sum(-1).max()),
                      production_sparse_vs_same_mask_fp32=error(actual,sparse_ref))
        records.append(record)
        print(json.dumps(record),flush=True)
        return result

    ALL_ATTENTION_FUNCTIONS.register(backend,audited)
    try:
        ids = tokenizer(prompt,return_tensors="pt").input_ids.to("cuda")
        with torch.inference_mode(), eager_decoding_request_guard():
            first = model(input_ids=ids,use_cache=True,logits_to_keep=1)
            token = first.logits[:,-1].argmax(-1).unsqueeze(1)
            model(input_ids=token,past_key_values=first.past_key_values,use_cache=True)
    finally:
        ALL_ATTENTION_FUNCTIONS.register(backend,original)
    report = {"status":"pending_independent_validation", "scope":"same actual GLM decode QKV, layers 0/23/46; full and TLI selected-token consumers; no end-to-end accuracy claim",
              "oracle":"FP32 QK matmul -> scale -> masked softmax -> FP32 V matmul; TF32 disabled",
              "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
              "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "records":records}
    (out/"qkv_audit.json").write_text(json.dumps(report,indent=2)+'\n')
    ok = len(records)==3 and all(r["production_triton_calls"]==1 and
        all(r[key]["allclose_atol_rtol_1e_2"] and r[key]["finite"] for key in (
            "native_vs_fp32","dense_vs_fp32","direct_full_triton_vs_fp32",
            "production_sparse_vs_same_mask_fp32")) for r in records)
    print("QKV_AUDIT_DONE",ok,flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
