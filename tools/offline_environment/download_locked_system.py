#!/usr/bin/env python3
"""Retrieve the exact recorded Debian payload; no install or re-resolution.
Old archive objects can disappear: report failure rather than substitute versions.
"""
import hashlib,json,tempfile,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    manifest=json.loads(Path(__file__).with_name('debian13-packages.json').read_text())
    out=ROOT/'.offline-cache/debian13';out.mkdir(parents=True,exist_ok=True)
    for row in manifest['files']:
        name=row['filename'];rel=row['official_archive_path']
        if Path(name).name!=name or not rel.startswith('pool/') or '..' in Path(rel).parts:
            raise ValueError('unsafe manifest path')
        dst=out/name
        if dst.is_symlink():raise ValueError('refusing symlink destination')
        if dst.exists() and dst.stat().st_size==row['bytes'] and hashlib.sha256(dst.read_bytes()).hexdigest()==row['sha256']:
            print('verified cached',name,flush=True);continue
        errors=[]
        for origin in ['https://deb.debian.org/debian/','https://security.debian.org/debian-security/']:
            tmp=None
            try:
                with urllib.request.urlopen(origin+rel,timeout=60) as src, tempfile.NamedTemporaryFile(dir=out,delete=False) as f:
                    tmp=Path(f.name);digest=hashlib.sha256();size=0
                    while chunk:=src.read(1024*1024):
                        size+=len(chunk)
                        if size>row['bytes']:raise ValueError('oversized payload')
                        f.write(chunk);digest.update(chunk)
                if size!=row['bytes'] or digest.hexdigest()!=row['sha256']:raise ValueError('hash/size mismatch')
                tmp.replace(dst);print('downloaded and verified',name,flush=True);break
            except Exception as exc:
                errors.append(str(exc))
                if tmp is not None:tmp.unlink(missing_ok=True)
        else:raise RuntimeError(f'Exact package unavailable: {name}: {errors}')
    print('Exact Debian13 amd64 payload verified; target installation NOT performed.')
if __name__=='__main__':main()
