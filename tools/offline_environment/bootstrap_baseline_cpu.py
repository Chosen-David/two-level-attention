#!/usr/bin/env python3
"""Recreate a separate CPU dependency candidate from the transported large bundle.
This does NOT install CUDA extensions or claim an official native baseline env.
"""
import argparse,hashlib,json,os,platform,subprocess,sys,venv
from pathlib import Path
from paper_env import ROOT,verify

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True,choices=['ffd','quest','fasa','sparq','minference','moba'])
    p.add_argument('--bundle',required=True,type=Path)
    a=p.parse_args()
    expected=Path(__file__).with_name('baseline-cpu-manifest.json').read_bytes()
    if (a.bundle/'manifest.json').read_bytes()!=expected:raise SystemExit('Bundle manifest differs from repository lock')
    verify(a.bundle)
    if platform.system()!='Linux' or platform.machine()!='x86_64' or platform.python_implementation()!='CPython' or sys.version_info[:2]!=(3,12):
        raise SystemExit('CPU bundle requires CPython3.12 Linux x86_64')
    libc,version=platform.libc_ver()
    if libc!='glibc' or tuple(map(int,version.split('.')[:2]))<(2,28):raise SystemExit('glibc >=2.28 required')
    dst=ROOT/('.venv-'+a.name+'-cpu')
    if dst.exists() or dst.is_symlink():raise SystemExit('Refusing to overwrite existing environment')
    venv.EnvBuilder(with_pip=True).create(dst)
    env=dict(os.environ,PIP_CONFIG_FILE=os.devnull,PIP_NO_INDEX='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    py=dst/'bin/python'
    subprocess.run([str(py),'-m','pip','--isolated','install','--no-index','--no-cache-dir','--only-binary=:all:','--require-hashes','--find-links',str(a.bundle/'wheelhouse'),'-r',str(a.bundle/'requirements.lock')],check=True,env=env)
    subprocess.run([str(py),'-m','pip','--isolated','check'],check=True,env=env)
    print(json.dumps({'name':a.name,'scope':'CPU dependencies only; native GPU environment incomplete','manifest_sha256':hashlib.sha256(expected).hexdigest(),'model_executed':False}))
if __name__=='__main__':main()
