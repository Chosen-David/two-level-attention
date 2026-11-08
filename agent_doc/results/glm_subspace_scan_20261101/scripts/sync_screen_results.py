"""Sync completed GLM subspace screen raw dirs from remote hosts to local results."""
import argparse
import json
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
STATE = BASE / 'screen_launch_state.json'
LOCAL_RAW = BASE / 'raw'

HOSTS = {
    'h20-122': dict(host='10.238.122.187', port=8416, user='wangyuanshuo02', password='zHWXp=eK5JvX~I|v'),
    'h20-139': dict(host='10.238.139.251', port=8416, user='wangyuanshuo02', password='zHWXp=eK5JvX~I|v'),
    'h20-57': dict(host='10.245.58.57', port=8416, user='wangyuanshuo02', password='zHWXp=eK5JvX~I|v'),
    'gpu-125': dict(host='33.253.189.125', port=22, user='hadoop-aipnlp', password='bK=/g2RN4a[0'),
}
# Special fixed runs: label -> (host info, remote path, required worker_done count)
EXTRA = {
    'full256-full': dict(host='33.253.189.125', port=22, user='hadoop-aipnlp', password='bK=/g2RN4a[0',
                         remote='/tmp/tli-glm-full256-aligned-full-run', workers=7),
    'full256-dual-screen': dict(host='33.253.189.125', port=22, user='hadoop-aipnlp', password='bK=/g2RN4a[0',
                                remote='/tmp/tli-glm-screen-full256-dual', workers=1),
}


def scp_recursive(h, remote, local):
    dest = local
    parent = dest.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_name = f'.sync-{dest.name}-{Path(remote).name}'
    tmp = parent / tmp_name
    cmd = ['sshpass', '-p', h['password'], 'scp', '-r',
           '-P', str(h['port']), '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=10',
           f"{h['user']}@{h['host']}:{remote}", str(tmp) + '/']
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        return res.returncode, res.stdout.strip()
    if not tmp.exists():
        return 1, f'expected downloaded dir {tmp} not found'
    if dest.exists():
        import shutil
        shutil.rmtree(dest)
    tmp.rename(dest)
    return 0, res.stdout.strip()


def done_count(h, remote):
    cmd = ['sshpass', '-p', h['password'], 'ssh',
           '-p', str(h['port']), '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=10',
           f"{h['user']}@{h['host']}",
           f"find {remote} -maxdepth 1 -name 'worker_done_rank*.json' 2>/dev/null | wc -l"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        return 0
    try:
        return int(res.stdout.strip().splitlines()[-1])
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    state = json.loads(STATE.read_text()) if STATE.exists() else {'launched': {}}
    for label, meta in state['launched'].items():
        host = HOSTS[meta['host']]
        remote = meta['out']
        count = done_count(host, remote)
        local = LOCAL_RAW / label
        if not args.force and count < 1:
            print(f'skip {label}: done={count}')
            continue
        ok, msg = scp_recursive(host, remote, local)
        print(f'sync {label}: done={count} ok={ok} {msg[:200]}')
    for label, h in EXTRA.items():
        count = done_count(h, h['remote'])
        if not args.force and count < h['workers']:
            print(f'skip {label}: done={count}/{h["workers"]}')
            continue
        ok, msg = scp_recursive(h, h['remote'], LOCAL_RAW / label)
        print(f'sync {label}: done={count} ok={ok} {msg[:200]}')


if __name__ == '__main__':
    main()
