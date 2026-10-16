#!/usr/bin/env python3
"""Inspect locked baseline identities, or explicitly fetch missing sources.
No installation, compilation, model loading, vendoring or GPU work.
"""
import argparse
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
REGISTRY=Path(__file__).with_name('baselines.json')

def git(path,*args):
    return subprocess.check_output(['git','-C',str(path),*args],text=True,stderr=subprocess.STDOUT,timeout=360).strip()

def fetch(row,recursive=False):
    path=ROOT/row['local_cache']
    if path.is_symlink() or path.resolve().parent != (ROOT/'third_party/offline-sources').resolve():
        raise ValueError('unsafe source cache path')
    if not row['repository'].startswith('https://github.com/'):
        raise ValueError('only registered official GitHub HTTPS sources supported')
    marker=path/'.git/tla-offline-source.json'
    identity={'repository':row['repository'],'requested_object':row['requested_object'],'commit':row['commit']}
    needs_fetch=False
    if path.exists():
        if not (path/'.git').is_dir():raise ValueError('existing cache is not a standalone checkout')
        if git(path,'status','--porcelain'):raise ValueError('existing source has changes; refusing to overwrite')
        try:
            head=git(path,'rev-parse','--verify','HEAD')
        except subprocess.CalledProcessError:
            # Only resume an unborn checkout that this tool created for this pin.
            if not marker.is_file() or json.loads(marker.read_text())!=identity:
                raise ValueError('unborn cache has no matching acquisition receipt; inspect manually')
            needs_fetch=True
        else:
            if head!=row['commit']:raise ValueError('existing source is at another revision; preserve it and inspect manually')
    else:
        path.mkdir(parents=True)
        git(path,'init')
        marker.write_text(json.dumps(identity,sort_keys=True)+'\n')
        needs_fetch=True
    if needs_fetch:
        git(path,'fetch','--depth=1',row['repository']+'.git',row['requested_object'])
        git(path,'checkout','--detach','FETCH_HEAD')
    if git(path,'rev-parse','HEAD') != row['commit']:raise ValueError('pinned commit mismatch')
    if recursive:git(path,'submodule','update','--init','--recursive','--depth=1')
    modules=git(path,'submodule','status','--recursive').splitlines()
    return {'name':row['name'],'commit':row['commit'],'source_present':True,
            'submodules':modules,'all_gitlinks_materialized':all(l and l[0] not in '-+U' for l in modules),
            'license':row['license'],'environment_installed':False}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fetch',nargs='+');p.add_argument('--recursive',action='store_true');a=p.parse_args()
    rows=json.loads(REGISTRY.read_text())['sources']
    if not a.fetch:print(json.dumps([{'name':r['name'],'commit':r['commit'],'license':r['license'],'interface':r['model_stage_interface']} for r in rows],indent=2));return
    byname={r['name']:r for r in rows}
    if set(a.fetch)-byname.keys():p.error('unknown source name')
    for name in a.fetch:print(json.dumps(fetch(byname[name],a.recursive)))
if __name__=='__main__':main()
