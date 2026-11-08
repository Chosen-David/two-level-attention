"""Create NEW candidate seal inputs below this review; no experiment acceptance.
Original six criteria are retained verbatim; specialty applicability needs review.
Commands are recorded only, never executed.
"""
import argparse
import datetime
from pathlib import Path
import seal_artifacts as s
ap=argparse.ArgumentParser(__doc__)
ap.add_argument("--scope",choices=s.SCOPES,required=True)
ap.add_argument("--output-artifact",action="append",required=True,type=Path)
ap.add_argument("--review-directory",action="append",default=[],type=Path,help="completed immutable independent snapshot; recursively bind ordinary files")
ap.add_argument("--environment-artifact",action="append",required=True,type=Path)
ap.add_argument("--command-evidence",action="append",required=True,type=Path)
ap.add_argument("--command",action="append",required=True)
ap.add_argument("--environment-description",required=True)
a=ap.parse_args()
parent=s.V1 if a.scope=="v1" else s.SPECIAL
original=(parent/"validation_plan.json").resolve(strict=True)
b=s.Bindings();p=b.read(original,"config")
p["parent_plan"]=b.files[str(original)]
p["parent_scope"]=p["scope"]
p["scope"]=s.SCOPES[a.scope]
p["projection_status"]="candidate scope binding only; trusted final reviewer must review original six criteria applicability; no criterion relaxed or silently marked N/A"
s.check_plan(p,s.SCOPES[a.scope])
folder=s.BASE/("candidate_"+a.scope+"_"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
review_files=[]
for root in a.review_directory:
    root=root.resolve(strict=True)
    s.need(root.is_dir(),"review directory missing")
    for item in sorted(root.rglob("*")):
        s.need(not item.is_symlink(),"review bundle symlink")
        if item.is_file():
            s.need(item.name not in ("session.log","FINAL_RESPONSE.md") and ".git" not in item.parts and "SYNTHETIC_NOT_QUALITY" not in item.parts,"use completed snapshot only, not live/full review directory")
            review_files.append(str(item))
spec=dict(scope=s.SCOPES[a.scope],bootstrap=5000 if a.scope=="v1" else 10000,seed=42,
    validation_plan=str(folder/"validation_plan.json"),
    outputs=[str(x.resolve(strict=True)) for x in a.output_artifact]+review_files,
    review_directories=[str(x.resolve(strict=True)) for x in a.review_directory],
    environment=[str(x.resolve(strict=True)) for x in a.environment_artifact],
    execution=dict(command=a.command,environment_description=a.environment_description,
                   command_evidence=[str(x.resolve(strict=True)) for x in a.command_evidence]))
b.finish();folder.mkdir()
(folder/"validation_plan.json").write_bytes(s.encode(p))
(folder/"spec.json").write_bytes(s.encode(spec))
print(folder/"spec.json")
