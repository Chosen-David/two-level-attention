"""两臂九rank闭集后，经SSH/tar收割远端产物并生成逐文件SHA256清单。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HEADS = {
    "nfa": "cbfd70da607eb5afed4343d6d055073f450e4162",
    "flashinfer": "7fac274fda9df13e82caadd95f3cca5c8374a1a3",
}
HOSTS = {
    "pc40": ("10.238.139.251", (2, 3)),
    "pc31": ("10.245.58.57", (4,)),
    "pc32": ("10.245.84.251", (5,)),
    "pc02": ("10.238.122.187", (6, 7, 8)),
}
LOCAL_RANKS = (0, 1)
METHODS = ("nfa", "flashinfer")
REMOTE_ROOT_NAME = "qwen3_nfa_fi_longbench_quality_eager_cbfd70da_20261114"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def ssh(host, command, *, stdout=None):
    return subprocess.run([
        "sshpass", "-e", "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=30", "-p", "8416",
        f"wangyuanshuo02@{host}", command,
    ], check=True, stdout=stdout)


def remote_closed(host, ranks):
    rank_text = ",".join(str(rank) for rank in ranks)
    command = (
        "python3 -c 'from pathlib import Path; import sys; "
        f"r=Path.home()/\"ak_verify_results/{REMOTE_ROOT_NAME}\"; "
        f"ranks=[{rank_text}]; "
        "missing=[str(p) for rank in ranks for p in "
        "[r/\"_status\"/f\"rank{rank}.done\"] + "
        "[r/m/f\"rank{rank}\"/f\"quality_manifest_s{rank}of9.json\" "
        "for m in (\"nfa\",\"flashinfer\")] if not p.is_file()]; "
        "missing += [str(r/\"_status\"/f\"rank{rank}_flashinfer.reused.json\") "
        "for rank in ranks if rank >= 6 and not "
        "(r/\"_status\"/f\"rank{rank}_flashinfer.reused.json\").is_file()]; "
        "failed=[str(p) for rank in ranks for suffix in (\"failed\",\"stopped\") "
        "for p in [r/\"_status\"/f\"rank{rank}.{suffix}\"] if p.exists()]; "
        "print({\"missing\":missing,\"failed\":failed}); "
        "sys.exit(1 if missing or failed else 0)'"
    )
    ssh(host, command)


def local_closed():
    for rank in LOCAL_RANKS:
        require((ROOT / "_status" / f"rank{rank}.done").is_file(),
                f"本机rank{rank}整链尚未完成")
        for suffix in ("failed", "stopped"):
            require(not (ROOT / "_status" / f"rank{rank}.{suffix}").exists(),
                    f"本机rank{rank}存在{suffix}状态")
        for method in METHODS:
            manifest = (ROOT / method / f"rank{rank}" /
                        f"quality_manifest_s{rank}of9.json")
            require(manifest.is_file(), f"缺少本机manifest: {manifest}")


def copy_verified(source, destination):
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(sha256(source) == sha256(destination),
                f"收割目标与来源不一致: {destination}")
        return
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copyfile(source, temporary)
    require(sha256(source) == sha256(temporary), f"复制后SHA不一致: {source}")
    os.replace(temporary, destination)


def safe_extract(archive, destination):
    with tarfile.open(fileobj=archive, mode="r:") as handle:
        for member in handle:
            path = Path(member.name)
            require(not path.is_absolute() and ".." not in path.parts,
                    f"tar包含非法路径: {member.name}")
            require(member.isfile() or member.isdir(),
                    f"tar包含非常规文件: {member.name}")
        handle.extractall(destination, filter="data")


def harvest_host(name, host, ranks, staging):
    remote_closed(host, ranks)
    members = [f"{method}/rank{rank}" for method in METHODS for rank in ranks]
    members += [f"_logs/rank{rank}_{method}.log"
                for method in METHODS for rank in ranks]
    members += [f"_status/rank{rank}.done" for rank in ranks]
    members += [f"_status/rank{rank}_flashinfer.reused.json"
                for rank in ranks if rank >= 6]
    remote_root = f"$HOME/ak_verify_results/{REMOTE_ROOT_NAME}"
    command = f"tar -C {remote_root} -cf - " + " ".join(members)
    target = staging / name
    target.mkdir()
    with (staging / f"{name}.tar").open("wb") as archive:
        ssh(host, command, stdout=archive)
    with (staging / f"{name}.tar").open("rb") as archive:
        safe_extract(archive, target)
    for source in sorted(target.rglob("*")):
        if source.is_file():
            copy_verified(source, ROOT / source.relative_to(target))


local_closed()
require("SSHPASS" in os.environ, "收割远端前必须设置SSHPASS环境变量")
with tempfile.TemporaryDirectory(prefix="harvest-", dir=ROOT) as temporary:
    staging = Path(temporary)
    for name, (host, ranks) in HOSTS.items():
        harvest_host(name, host, ranks, staging)

records = []
for method in METHODS:
    for rank in range(9):
        rank_root = ROOT / method / f"rank{rank}"
        require(rank_root.is_dir(), f"缺少rank目录: {rank_root}")
        for path in sorted(rank_root.rglob("*")):
            if path.is_file():
                records.append({
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                })
for rank in range(9):
    for method in METHODS:
        path = ROOT / "_logs" / f"rank{rank}_{method}.log"
        require(path.is_file(), f"缺少日志: {path}")
        records.append({"path": str(path.relative_to(ROOT)),
                        "bytes": path.stat().st_size, "sha256": sha256(path)})
    path = ROOT / "_status" / f"rank{rank}.done"
    require(path.is_file(), f"缺少完成状态: {path}")
    records.append({"path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size, "sha256": sha256(path)})
    if rank >= 6:
        path = ROOT / "_status" / f"rank{rank}_flashinfer.reused.json"
        require(path.is_file(), f"缺少完整复用回执: {path}")
        records.append({"path": str(path.relative_to(ROOT)),
                        "bytes": path.stat().st_size, "sha256": sha256(path)})

summary = {
    "schema": "qwen3-nfa-fi-longbench-quality-harvest-v2",
    "verification": "pending-independent-verification",
    "source_heads": EXPECTED_HEADS,
    "ranks": 9,
    "arms": list(METHODS),
    "files": records,
    "status": "PASS",
}
output = ROOT / "harvest_manifest.json"
output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                  encoding="utf-8")
print(json.dumps({"status": "PASS", "files": len(records),
                  "manifest": str(output), "manifest_sha256": sha256(output)},
                 ensure_ascii=False))
