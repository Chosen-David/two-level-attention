from pathlib import Path
import os,sys,json,subprocess,datetime
B=Path(__file__).resolve().parent
name=sys.argv[1];cmd=sys.argv[2:]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',TMPDIR=str(B/'tmp'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
(B/'tmp').mkdir(exist_ok=True)
r=dict(command=cmd,cwd=str(B),login=False,started=datetime.datetime.now(datetime.timezone.utc).isoformat(),environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE','TOKENIZERS_PARALLELISM','TMPDIR','HF_HUB_OFFLINE']})
with (B/(name+'.log')).open('w') as f:
 p=subprocess.run(cmd,cwd=B,env=env,stdout=f,stderr=subprocess.STDOUT)
r.update(exit_code=p.returncode,ended=datetime.datetime.now(datetime.timezone.utc).isoformat());(B/(name+'.command.json')).write_text(json.dumps(r,indent=2));print(json.dumps(r))
raise SystemExit(p.returncode)
