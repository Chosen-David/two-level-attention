#!/usr/bin/env python3
"""Offline integrity/check/plan only. Never installs, imports GPU code or downloads."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT=Path(__file__).resolve().parents[1]
def check(root=ROOT):
    manifest=json.loads((root/'UPSTREAM.json').read_text())
    for name,digest in manifest['files'].items():
        path=root/name
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('missing/modified upstream file: '+name)
    for base in ('upstream','environment/vllm_metadata'):
        actual={str(p.relative_to(root)) for p in (root/base).rglob('*') if p.is_file()}
        expected={p for p in manifest['files'] if p.startswith(base+'/')}
        if actual!=expected:raise ValueError('unexpected source inventory: '+base)
    return manifest

def plan():
    manifest=check()
    installed={}
    for name in ('torch','transformers','vllm','triton'):
        try:installed[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:installed[name]=None
    return dict(mode='offline check/plan',source_verified=True,jssa_commit=manifest['commit'],vllm_commit=manifest['vllm_commit'],host=dict(python=platform.python_version(),platform=platform.system(),machine=platform.machine(),packages=installed),required_python='3.12',python_matches=sys.version_info[:2]==(3,12),gpu_initialized=False,installation_started=False,execution_ready=False,dependency_lock_complete=False,blockers=['Resolve and hash target CUDA wheel and complete transitive environment lock on authorized host','Verify driver/CUDA variant and Triton compiler compatibility','Require frozen basis provenance with checkpoint/tokenizer/data/producer and identical effective calibration/serving RoPE; upstream basis M alone is insufficient','HF flash-attn default calibration dependency/version/build unresolved; SDPA would be an explicit alternative','No GPU numerical or benchmark acceptance; no project runtime adapter'])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='same safe offline integrity check as default')
    parser.parse_args()
    print(json.dumps(plan(),indent=2))
if __name__=='__main__':main()
