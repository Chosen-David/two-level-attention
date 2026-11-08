from pathlib import Path
import json,hashlib,datetime
B=Path(__file__).resolve().parent
excluded={'session.log','session_retry.log','CLI_FINAL.md','CLI_FINAL_RETRY.md','EVIDENCE_SHA256.json'}
files={}
for p in sorted(B.rglob('*')):
 if not p.is_file() or (p.parent == B and p.name in excluded):continue
 assert not p.is_symlink(),str(p)
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 files[str(p.relative_to(B))]={'sha256':h.hexdigest(),'bytes':p.stat().st_size}
r=dict(schema='independent-review-evidence/v1',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),project_root='/home/wangyuanshuo02/two-level-attention',review_root=str(B),status='stable review evidence only; scientific status in REPORT_DETAIL.md and exact scoped validation',excluded=sorted(excluded),files=files)
(B/'EVIDENCE_SHA256.json').write_text(json.dumps(r,indent=2)+'\n')
# Independent read-back validates each sealed byte; does not write any more files.
for rel,x in files.items():
 h=hashlib.sha256()
 with (B/rel).open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 assert h.hexdigest()==x['sha256'],rel
print(json.dumps({'sealed_files':len(files),'readback':'all hashes matched','manifest_sha256':hashlib.sha256((B/'EVIDENCE_SHA256.json').read_bytes()).hexdigest()}))
