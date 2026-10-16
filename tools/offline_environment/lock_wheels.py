#!/usr/bin/env python3
"""Lock an already downloaded wheel set; does not claim target validation."""
from pathlib import Path
import argparse, hashlib, json, zipfile, email

def lock(folder, target, source="origin not inferred from local wheel bytes"):
    rows=[]; req=[]
    for w in sorted((folder/'wheelhouse').glob('*.whl')):
        with zipfile.ZipFile(w) as z:
            m=email.message_from_bytes(z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))))
            license_files=[n for n in z.namelist() if any(s in n.lower() for s in ['license','copying','notice'])]
        h=hashlib.sha256(w.read_bytes()).hexdigest()
        rows.append(dict(filename=w.name,name=str(m['Name']),version=str(m['Version']),sha256=h,bytes=w.stat().st_size,license_expression=str(m.get('License-Expression','see embedded license files')),license_files=license_files))
        req.append(f"{m['Name']}=={m['Version']} --hash=sha256:{h}")
    if not rows: raise ValueError('no wheels')
    (folder/'requirements.lock').write_text('\n'.join(req)+'\n')
    (folder/'manifest.json').write_text(json.dumps(dict(target=target,source=source,files=rows,total_bytes=sum(r['bytes'] for r in rows),requirements_sha256=hashlib.sha256((folder/'requirements.lock').read_bytes()).hexdigest()),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('--target',required=True);p.add_argument('--source',default='origin not inferred from local wheel bytes');a=p.parse_args();lock(a.folder,a.target,a.source)
