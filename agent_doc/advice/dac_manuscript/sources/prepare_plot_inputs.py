"""Derive privacy-minimal plot inputs from immutable repository results.
Run with --repository PATH; original files are read only. Rebuild uses the bundled derived inputs.
"""
from pathlib import Path
import argparse,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--repository',type=Path,required=True);args=p.parse_args()
OUT=Path(__file__).resolve().parent/'data'
EXPECTED_SHA256={'exp/trace/results/e152v2_e2e_latency_verdict.json': 'cbae41ef66aeab661ccf64b5548cc683cd596f7dedb1948df556e6e3c250bbc3', 'exp/trace/results/e140_o_trace_v2.json': 'f0db60364a0e516b5ad28affeb895dadb2b50e0eecd583ee21b0a993b5029b2b'}
def source(path):
 b=(args.repository/path).read_bytes()
 if hashlib.sha256(b).hexdigest()!=EXPECTED_SHA256[path]:raise ValueError('Source differs from the frozen commit: '+path)
 return json.loads(b),{'source_path':path,'source_sha256':hashlib.sha256(b).hexdigest(),'source_commit':'4067e6c25d5110322ffcf48f3beb77f50b82778a'}
x,ref=source('exp/trace/results/e152v2_e2e_latency_verdict.json')
y={'provenance':{**ref,'transformation':'Retain only three dynamic-cache lengths and dense/sparse TPOT values; transcribe relevant model/hardware/software fields from caliber; omit static points, attribution prose and all host/network/user paths.'},'measurement':{'model':'Qwen3-8B','dtype':'bf16','attention_backend_declared':'sdpa','hardware':'NVIDIA H20','transformers':'4.56.2','torch':'2.8.0','generation':'greedy, fixed128 output tokens','tpot_denominator':'127 after-first output intervals','instrumentation':'per-step device synchronized; runtime route not independently verified','cache':'dynamic','alpha':.125,'beta':.375,'gamma':.625,'K1':128,'K':1024,'protection':'not explicitly bound'},'results_median':{L:{arm:{'tpot_ms':x['results_median'][L][arm]['tpot_ms']} for arm in ['dense_dynamic','tli_dynamic']} for L in ['L32768','L65536','L131072']}}
(OUT/'decode_tpot_plot_input.json').write_text(json.dumps(y,indent=2)+'\n')
x,ref=source('exp/trace/results/e140_o_trace_v2.json')
y={'provenance':{**ref,'transformation':'Retain O1 bins/region totals and O3 cell summaries plus task/sample/layer identity; retain protocol fields; omit private snapshot/sidecar locations, receipts and unneeded revisions. No numerical values changed.'},'meta':{k:x['meta'][k] for k in ['schema','status','independent_unit','n_source_sequences','tasks','n_sample','layers','alpha','bs','K1','K2_mid','sink','swa','subspaces','o3_rule','o1_rule']},'cells':[{k:c[k] for k in ['task','sample','layer','o1','o3']} for c in x['cells']]}
(OUT/'observation_plot_input.json').write_text(json.dumps(y,indent=2)+'\n')
