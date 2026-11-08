"""CPU-only shared-storage artifact sealer. NEVER a scientific admission provider.
Only writes a new directory below this script. No producer modules are imported.
CLI uses fixed production locations; tests call the pure completeness primitives.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import stat
import subprocess
import sys

BASE = Path(__file__).resolve().parent
PROJECT = Path("/home/wangyuanshuo02/two-level-attention")
V1 = PROJECT/"agent_doc/results/frozen_near_far_v1_16_independent005_20261030"
SPECIAL = PROJECT/"agent_doc/results/frozen_qasper_mk2_64k_20261030"
SOURCE = Path("/home/wangyuanshuo02/two-level-attention-frozen-main")
DEPLOYED = Path("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/tla/frozen005_v1_16_source_20261030")
MK_SOURCE = Path("/home/wangyuanshuo02/two-level-attention-frozen-mk2")
V1_ID = "628463dbc68fcb474cd9c77b48f553d832c23983795748689b9748f63575f7e2"
MK_ID = "dd0e84bc766e7131d0ef12f935535693eaf28821a05fc09bc84af7b4dfea9214"
V1_SHA = "08ce44d6027a235080e7ff4127754af6b1ae91e0"
MK_SHA = "de2a1c1670b82bebcde6c0c3aa02fb8ee1431568"
ARMS = {"full", "far_0.05", "dual_0.05"}
TASKS = "narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p".split()
COUNTS = {d:150 if d=="multifieldqa_en" else 500 if d in ("lcc","repobench-p") else 200 for d in TASKS}
SCOPES = {
 "v1": "Qwen3-8B current08ce44d LongBench V1 English/code16;3750 paired/11250 arms/128done;Full TLI gates-off,Far .05,Dual independent .05;AVG16 task equal weight;paired bootstrap5000 seed42 order Far-Full,Dual-Full,Dual-Far;eager quality only;no latency/equivalence claim",
 "qasper": "Qwen3-8B current08ce44d Qasper only;200 paired/600 arms;Full TLI gates-off,Far .05,Dual independent .05;paired bootstrap10000 seed42 order Dual-Far,Far-Full,Dual-Full;primary Dual-Far;no requirement other V1 tasks done;no latency/equivalence claim",
 "mk2": "Qwen3-8B de2a1c1 RULER niah_multikey_2 65536 YaRN2 only;100 paired/300 arms;Full TLI gates-off,Far .05,Dual independent .05;paired bootstrap10000 seed42 order Dual-Far,Far-Full,Dual-Full;primary Dual-Far;no latency/equivalence claim"
}
CRITERIA = {"implementation","reference_boundary","data_integrity","numerical_sanity","measurement_validity","reproducibility"}
LIMITATIONS = [
 "Manifest integrity is not scientific acceptance; real independent six-criterion review and report prose review are required.",
 "No independent GPU regeneration; no new GPU task required by this sealer.",
 "Q/K not retained: real attention mass cannot be independently reconstructed.",
 "Post-launch model/tokenizer hashes do not prove historical in-memory loading.",
 "Local/shared source bytes and historical remote environment receipts do not prove every remote process used those bytes.",
 "Sequential before/after hashes are not an atomic filesystem snapshot or protection against adversarial change-and-revert.",
 "Shared-storage adaptation: absolute canonical references and streamed large files; generic project-relative no-symlink/16MiB validator is NOT claimed to pass.",
 "Scoped plan copies original criteria without weakening them; specialty parent contains both tasks. Trusted reviewer must resolve criterion applicability using the original protocol; neither sub-result closes the combined task."
]
class Refusal(ValueError): pass
def need(ok, why):
    if not ok: raise Refusal(why)
def digest(b): return hashlib.sha256(b).hexdigest()
def encode(x): return (json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode()
def no_duplicate(pairs):
    d={}
    for k,v in pairs:
        need(k not in d,"duplicate JSON key "+k); d[k]=v
    return d
def decode(b):
    return json.loads(b,object_pairs_hook=no_duplicate,
                      parse_constant=lambda s: (_ for _ in ()).throw(Refusal("nonfinite JSON "+s)))
def regular(path):
    p=Path(path)
    need(p.is_absolute(),"absolute path required")
    # Canonical roots are explicitly resolved once; all descendants must be normal.
    need(p.resolve(strict=True)==p,"noncanonical/symlink path "+str(p))
    need(stat.S_ISREG(p.stat().st_mode),"not regular "+str(p))
    return p
def fingerprint(path):
    p=regular(path); before=p.stat()
    h=hashlib.sha256()
    with p.open("rb") as f:
        need(stat.S_ISREG(os.fstat(f.fileno()).st_mode),"nonregular opened file")
        for block in iter(lambda:f.read(4*1024*1024),b""): h.update(block)
        opened=os.fstat(f.fileno())
    after=p.stat()
    sig=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    need(sig(before)==sig(opened)==sig(after),"file changed while hashing "+str(p))
    return dict(path=str(p),sha256=h.hexdigest(),bytes=after.st_size,
                device=after.st_dev,inode=after.st_ino,mtime_ns=after.st_mtime_ns,ctime_ns=after.st_ctime_ns)
class Bindings:
    def __init__(self): self.files={}; self.groups={k:[] for k in ("code","inputs","config","raw_data","outputs","environment")}; self.roots={}; self.memberships=[]
    def root(self,logical):
        logical=Path(logical).absolute(); canonical=logical.resolve(strict=True)
        self.roots[str(logical)]=str(canonical); return canonical
    def add(self,path,group,expected=None):
        p=Path(path); ref=fingerprint(p)
        if expected is not None: need(ref["sha256"]==expected,"hash differs "+str(p))
        key=str(p)
        if key in self.files: need(self.files[key]==ref,"changed since first read "+key)
        else: self.files[key]=ref
        if key not in self.groups[group]: self.groups[group].append(key)
        return ref
    def read(self,path,group):
        ref=self.add(path,group); b=Path(path).read_bytes()
        need(digest(b)==ref["sha256"],"read changed "+str(path));return decode(b)
    def listing(self,root,pattern):
        paths=sorted(Path(root).glob(pattern))
        self.memberships.append((Path(root),pattern,[str(p) for p in paths]))
        return paths
    def finish(self):
        for logical,canonical in self.roots.items():
            need(str(Path(logical).resolve(strict=True))==canonical,"root alias retargeted "+logical)
        for root,pattern,names in self.memberships:
            need([str(p) for p in sorted(root.glob(pattern))]==names,"file set changed "+str(root/pattern))
        for key,old in self.files.items():
            current=fingerprint(key)
            need(current==old,"binding changed "+key+" "+str({k:(old[k],current[k]) for k in old if old[k]!=current[k]}))
    def artifacts(self):
        return {g:[self.files[p] for p in sorted(paths)] for g,paths in self.groups.items()}

def coverage(kind,run,b):
    """Structural only. No scoring, tokenization, cache checks or partial statistics."""
    run=b.root(run); mk=kind=="mk2"
    manifest=run/("manifest.json" if mk else "manifest_rank0.json")
    m=b.read(manifest,"config"); cfg=m["protocol"]; ident=m["identity"]
    need(ident==(MK_ID if mk else V1_ID),"wrong frozen identity")
    need(digest(json.dumps(cfg,sort_keys=True).encode())==ident,"protocol digest")
    need(cfg["git_head"]==(MK_SHA if mk else V1_SHA),"wrong code revision")
    if mk:
        need(cfg["samples"]==100 and cfg["selected_indices"]==list(range(100)),"MK2 input ID contract")
        expected={("niah_multikey_2",i) for i in range(100)}
        paths=b.listing(run,"requests/*.json")
    else:
        need(cfg["datasets"]==TASKS and cfg["selected_counts"]==COUNTS and cfg["dataset_counts"]==COUNTS,"16-task contract")
        need(cfg["world_size"]==128 and cfg["seed"]==42 and cfg["full_dataset"],"shard/seed/full contract")
        for d in TASKS: need(len(cfg["selected_rows"][d])==COUNTS[d] and set(cfg["selected_rows"][d])==set(range(COUNTS[d])),"selected IDs "+d)
        datasets=TASKS if kind=="v1" else ["qasper"]
        expected={(d,i) for d in datasets for i in range(COUNTS[d])}
        paths=b.listing(run,"requests/*/*.json" if kind=="v1" else "requests/qasper/*.json")
    need(len(paths)==len(expected),"partial or extra request files: exact full scope required")
    seen=set()
    for p in paths:
        r=b.read(p,"raw_data"); key=(r["dataset"],r["index" if mk else "row_index"])
        need(key in expected and key not in seen,"duplicate/foreign request")
        need(p==run/"requests"/(f"{key[1]:04d}.json" if mk else f"{key[0]}/{key[1]:04d}.json"),"noncanonical raw filename")
        need(r["identity"]==ident and r["status"]=="complete" and set(r["arms"])==ARMS,"partial arms/status/identity")
        for arm in r["arms"].values():
            need(isinstance(arm,dict) and isinstance(arm.get("generated_ids"),list) and len(arm["generated_ids"])>0,"missing generation")
            need(isinstance(arm.get("score"),(int,float)) and not isinstance(arm["score"],bool) and math.isfinite(arm["score"]),"missing/nonfinite score")
        seen.add(key)
    need(seen==expected,"missing scope IDs")
    if mk:
        done=b.read(run/"GENERATED.json","raw_data")
        need(done["identity"]==ident and done["samples"]==100,"MK2 GENERATED")
    else:
        mapping={};offset=0
        for d in TASKS:
            for i in cfg["selected_rows"][d]:
                if (d,i) in expected: mapping[(d,i)]=offset%128
                offset+=1
        ranks=set(mapping.values())
        for rank in sorted(ranks):
            rm=b.read(run/f"manifest_rank{rank}.json","config")
            need(rm["rank"]==rank and rm["identity"]==ident and rm["protocol"]==cfg,"covering rank manifest")
        if kind=="v1":
            need(len(b.listing(run,"manifest_rank*.json"))==128,"extra/missing rank manifests")
            need(len(b.listing(run,"worker_done_rank*.json"))==128,"128 done required")
            queue=b.read(run/"queue.json","raw_data"); jobs=queue["jobs"]
            need(len(jobs)==128 and {j["rank"] for j in jobs}==set(range(128)) and all(j["status"]=="done" for j in jobs),"queue not fully done")
            for j in jobs:
                rank=j["rank"];d=b.read(run/f"worker_done_rank{rank}.json","raw_data")
                rm=b.read(run/f"manifest_rank{rank}.json","config")
                need(d["identity"]==ident and d["pid"]==rm["pid"]==j["pid"],"done identity/PID")
    return cfg,ident,len(expected)

def git_head(root):
    return subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
def bind_source(b,root,cfg):
    root=b.root(root);need(git_head(root)==cfg["git_head"],"source HEAD mismatch")
    for name,h in cfg["source_sha256"].items():
        p=root/name;need(p.is_relative_to(root),"source escape")
        b.add(p,"code",h)
        need(digest(subprocess.check_output(["git","show",cfg["git_head"]+":"+name],cwd=root))==h,"commit object differs "+name)
    # Record dependency directory membership too; added runtime modules require review.
    for p in b.listing(root/"sparse_attn","**/*.py"):
        need(str(p.relative_to(root)) in cfg["source_sha256"],"unbound source module "+str(p))
    return root

def bind_model(b,cfg):
    root=b.root(cfg["model"])
    index=b.read(root/"model.safetensors.index.json","inputs")
    shards=set(index["weight_map"].values())
    need(shards and all(Path(s).name==s for s in shards),"invalid weight shard index")
    need({p.name for p in b.listing(root,"*.safetensors")}==shards,"unindexed/missing weights")
    for p in b.listing(root,"*"):
        if p.is_file():b.add(p,"inputs")
        else:need(False,"unexpected model subdirectory; explicit adaptation required")
    for f,k in [("config.json","model_config_sha256"),("tokenizer_config.json","tokenizer_config_sha256"),("generation_config.json","generation_config_sha256")]:
        if k in cfg:b.add(root/f,"config",cfg[k])
    for f in ("tokenizer.json","generation_config.json"):b.add(root/f,"config")
    return root

def check_plan(plan,scope):
    need(plan.get("schema_version")=="experiment-validation-plan/v1","validation plan schema")
    need(plan.get("scope")==scope,"scope differs across plan/contract/manifest")
    need(set(plan.get("criteria",{}))==CRITERIA,"six criteria required")
    for k,c in plan["criteria"].items():
        need(isinstance(c.get("procedure"),str) and c["procedure"] and isinstance(c.get("acceptance"),str) and c["acceptance"] and isinstance(c.get("allow_not_applicable"),bool),"criterion "+k)

def seal(kind,spec_path,out):
    b=Bindings();run= SPECIAL/"ruler64k" if kind=="mk2" else V1
    cfg,ident,n=coverage(kind,run,b) # FULL gate BEFORE external spec/model reads
    spec_path=Path(spec_path).resolve(strict=True);spec=b.read(spec_path,"config")
    need(spec["scope"]==SCOPES[kind],"spec scope mismatch")
    need(spec["bootstrap"]==(5000 if kind=="v1" else 10000) and spec["seed"]==42,"statistics mismatch")
    plan_path=Path(spec["validation_plan"]).resolve(strict=True)
    plan=b.read(plan_path,"config");check_plan(plan,SCOPES[kind])
    parent=SPECIAL if kind!="v1" else V1
    parent=b.root(parent)
    b.add(parent/"run_contract.json","config");b.add(parent/"validation_plan.json","config")
    sources=[MK_SOURCE] if kind=="mk2" else [SOURCE,DEPLOYED]
    roots=[bind_source(b,s,cfg) for s in sources]
    run=b.root(run)
    if kind=="mk2":
        data=b.root(Path(cfg["data"]).parent)/Path(cfg["data"]).name
        b.add(data,"inputs",cfg["data_sha256"])
        rows=decode_jsonl(data);need([r["index"] for r in rows]==cfg["selected_indices"],"input ID order")
        for name in ("summary.json","validation.json"):b.add(run/name,"outputs")
        b.add(parent/"run_local.py","code")
        b.add(parent/"formal_launch.json","environment")
        # Existing smoke evidence is necessary context, not fresh GPU verification.
        for name in ("manifest.json","requests/0000.json","validation.json","summary.json","GENERATED.json"):
            b.add(parent/"smoke"/name,"raw_data")
        b.add(parent/"smoke_verification.md","outputs")
    else:
        data_root=b.root(Path("/home/wangyuanshuo02/datasets/LongBench/data"))
        for d in TASKS if kind=="v1" else ["qasper"]:
            b.add(data_root/(d+".jsonl"),"inputs",cfg["dataset_sha256"][d])
            need(len(decode_jsonl(data_root/(d+".jsonl")))==COUNTS[d],"input count")
        for source in roots:
            for name in ("benchmark/LongBench/config/dataset2prompt.json","benchmark/LongBench/config/dataset2maxlen.json"):
                b.add(source/name,"config",cfg["source_sha256"][name])
        b.add(run/"dispatch_plan.json","config") # provenance only, NEVER result manifest
        if kind=="v1":
            for name in ("summary.json","validation.json","predictions.jsonl","raw_index.json","scores.tsv","paired_differences.tsv","activation_storage.tsv","per_task_activation.tsv","per_request_scores.tsv","REPORT.md","FINAL_REPORT_ZH.md"):
                b.add(run/name,"outputs")
        # Historical per-host evidence bound as receipts, not local re-measurement.
        historical=run/"ai_supervision/round_20261030T014152"
        for host in ("pc02","pc31","pc40"):
            for suffix in ("model_identity","software_identity"):
                b.add(historical/f"{host}_{suffix}.json","environment")
    bind_model(b,cfg)
    # Caller explicitly enumerates final reviewer snapshots and actual environment receipts.
    # Existence/hashes do not authenticate reviewer identity or claims.
    for group in ("outputs","environment"):
        need(isinstance(spec[group],list) and spec[group],"nonempty explicit "+group)
        for item in spec[group]:
            logical=Path(item);canonical=b.root(logical.parent)/logical.name
            b.add(canonical,group)
    for directory in spec.get("review_directories",[]):
        root=b.root(directory)
        for item in b.listing(root,"**/*"):
            need(not item.is_symlink(),"review snapshot contains symlink")
            if item.is_file():
                need(item.name not in ("session.log","FINAL_RESPONSE.md") and ".git" not in item.parts and "SYNTHETIC_NOT_QUALITY" not in item.parts,"live/unscoped review directory")
                b.add(item,"outputs")
    need(isinstance(spec["execution"]["command"],list) and spec["execution"]["command"] and all(isinstance(x,str) and x for x in spec["execution"]["command"]),"actual command provenance required")
    need(spec["execution"].get("environment_description") and spec["execution"].get("command_evidence"),"execution environment/command evidence required")
    for item in spec["execution"]["command_evidence"]:b.add(b.root(Path(item).parent)/Path(item).name,"environment")
    if kind=="qasper":
        need(any(Path(p).name=="QASPER_REPORT_FACTS.json" for p in b.groups["outputs"]),"full Qasper checker facts required")
        need(any(Path(p).name=="QASPER_CURRENT_REPORT.md" for p in b.groups["outputs"]),"Qasper final report required")
    else:
        need(any(Path(p).name=="FINAL_REPORT_ZH.md" for p in b.groups["outputs"]),"final report required")
    b.add(Path(__file__).resolve(),"code")
    b.finish()
    need(all(b.groups.values()),"empty artifact category")
    manifest=dict(schema_version="shared-storage-experiment-result/v1",
        compatibility="explicit project adaptation; NOT accepted by generic result validator",
        project_root=str(PROJECT),result_id=kind+"-sealed-"+out.name,run_id=run.name,
        scope=SCOPES[kind],source_identity=ident,code_revision=cfg["git_head"],
        status="sealed-pending-independent-review",full_acceptance="pending",
        validation_plan=b.files[str(plan_path)],bootstrap=spec["bootstrap"],seed=42,
        execution=spec["execution"],canonical_roots=b.roots,artifacts=b.artifacts(),
        file_sets=[dict(root=str(root),pattern=pattern,paths=names) for root,pattern,names in b.memberships],
        metrics=[dict(name="paired_requests",value=n,unit="count"),dict(name="generation_arms",value=3*n,unit="count")],
        limitations=LIMITATIONS,independent_validation=None,
        closure="Never write COMPLETE or infer acceptance from exit0. Independent reviewer must bind this exact manifest and all six checks.",
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    out.mkdir() # fail if an existing seal would be overwritten
    (out/"artifact_manifest.json").write_bytes(encode(manifest))
    contract=dict(result_id=manifest["result_id"],producer_task_id="TL-FROZEN-V1-16-20261030" if kind=="v1" else "TL-FROZEN-QASPER-MK2-20261030",
        producer_actor="/root",scope=SCOPES[kind],manifest_path=str(out/"artifact_manifest.json"),
        manifest_sha256=digest((out/"artifact_manifest.json").read_bytes()),
        validation_plan=b.files[str(plan_path)],status="pending",adapter="shared-storage-explicit; no trusted verifier installed")
    (out/"result_contract.json").write_bytes(encode(contract))
    return manifest

def decode_jsonl(path):return [decode(s) for s in Path(path).read_bytes().splitlines() if s.strip()]

def verify_seal(path,kind):
    """Rehash immutable seal references. Returns integrity only, NEVER admission."""
    path=Path(path).resolve(strict=True)
    before=fingerprint(path); m=decode(path.read_bytes())
    need(m["schema_version"]=="shared-storage-experiment-result/v1","adapter schema")
    need(m["scope"]==SCOPES[kind] and m["full_acceptance"]=="pending" and m["independent_validation"] is None,"manifest scope/state")
    need(m["bootstrap"]==(5000 if kind=="v1" else 10000) and m["seed"]==42,"manifest bootstrap/seed")
    b=Bindings()
    for logical,canonical in m["canonical_roots"].items():
        need(str(b.root(logical))==canonical,"alias changed")
    need(set(m["artifacts"])==set(b.groups),"six artifact categories")
    for group,refs in m["artifacts"].items():
        need(refs,"empty artifact group")
        for ref in refs:
            actual=b.add(Path(ref["path"]),group)
            need(actual==ref,"artifact changed "+ref["path"])
    for fs in m["file_sets"]:
        need([str(p) for p in b.listing(Path(fs["root"]),fs["pattern"])]==fs["paths"],"membership changed")
    plan_ref=m["validation_plan"]
    need(plan_ref["path"] in b.files and b.files[plan_ref["path"]]==plan_ref,"plan not bound")
    check_plan(decode(Path(plan_ref["path"]).read_bytes()),m["scope"])
    cp=path.parent/"result_contract.json";contract_before=fingerprint(cp)
    c=decode(cp.read_bytes())
    need(c["manifest_sha256"]==before["sha256"] and c["manifest_path"]==str(path),"contract manifest binding")
    need(c["scope"]==m["scope"] and c["validation_plan"]==plan_ref and c["status"]=="pending","contract scope/plan/state")
    b.finish()
    need(fingerprint(path)==before and fingerprint(cp)==contract_before,"seal changed during recheck")
    return dict(status="integrity rechecked only",full_acceptance="pending",manifest_sha256=before["sha256"],scope=m["scope"])

def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument("--scope",choices=SCOPES,required=True)
    ap.add_argument("--verify",type=Path,help="rehash an existing seal; no scientific admission")
    ap.add_argument("--spec",type=Path,help="explicit final plan/output/environment/execution bindings")
    ap.add_argument("--preflight",action="store_true",help="structural full-scope gate only; always pending")
    ap.add_argument("--name",help="new directory basename below this script")
    a=ap.parse_args()
    if a.verify:
        try:print(json.dumps(verify_seal(a.verify,a.scope)));return 0
        except (Refusal,OSError,KeyError,TypeError) as e:
            print(json.dumps(dict(status="invalid or pending",reason=str(e),full_acceptance="pending")));return 2
    if a.preflight:
        b=Bindings()
        try:
            _,_,n=coverage(a.scope,SPECIAL/"ruler64k" if a.scope=="mk2" else V1,b);b.finish()
            print(json.dumps(dict(status="complete structure; pending",scope=a.scope,rows=n,quality_statistics="not computed")))
        except (Refusal,OSError,KeyError,TypeError) as e: print(json.dumps(dict(status="pending or invalid",scope=a.scope,reason=str(e),quality_statistics="not computed")))
        return 2
    need(a.spec is not None,"--spec required for seal")
    name=a.name or "seal_"+a.scope+"_"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    need(Path(name).name==name and name not in (".",".."),"basename required")
    try:
        m=seal(a.scope,a.spec,BASE/name)
        print(json.dumps(dict(status=m["status"],full_acceptance="pending",manifest=str(BASE/name/"artifact_manifest.json"))))
        return 0
    except (Refusal,OSError,KeyError,TypeError,subprocess.CalledProcessError) as e:
        print(json.dumps(dict(status="pending or invalid",reason=str(e),full_acceptance="pending")));return 2
if __name__=="__main__":sys.exit(main())
