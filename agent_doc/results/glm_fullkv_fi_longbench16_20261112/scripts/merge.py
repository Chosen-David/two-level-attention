#!/usr/bin/env python3
"""合并按主机保存的 GLM LongBench 分片，冲突或重复内容不一致时拒绝写入。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--host', action='append', required=True,
                        help='NAME=已收割到本机的该主机输出目录')
    parser.add_argument('--output', required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f'输出目录非空，拒绝覆盖：{output}')
    output.mkdir(parents=True, exist_ok=True)
    copied = {}
    evidence = []
    for specification in args.host:
        name, separator, raw_path = specification.partition('=')
        if not separator or not name:
            raise ValueError(f'非法 --host：{specification}')
        root = Path(raw_path)
        if not root.is_dir():
            raise FileNotFoundError(root)
        for source in sorted(path for path in root.rglob('*') if path.is_file()):
            relative = source.relative_to(root)
            digest = sha_file(source)
            evidence.append({
                'host': name, 'source': str(source),
                'relative_path': str(relative), 'bytes': source.stat().st_size,
                'sha256': digest,
            })
            destination = output / relative
            key = str(relative)
            if key in copied:
                if copied[key] != digest:
                    raise RuntimeError(f'跨主机同路径内容冲突：{relative}')
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if sha_file(destination) != digest:
                    raise RuntimeError(f'目标已存在且内容不同：{destination}')
            else:
                shutil.copy2(source, destination)
            copied[key] = digest
    manifest = {
        'schema': 'glm-fullkv-fi-lb16-harvest-v1',
        'hosts': args.host, 'merged_output': str(output),
        'merged_file_count': len(copied), 'source_file_count': len(evidence),
        'source_files': evidence,
    }
    path = Path(args.manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({
        'merged_output': str(output), 'merged_file_count': len(copied),
        'source_file_count': len(evidence), 'manifest': str(path),
        'manifest_sha256': sha_file(path),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
