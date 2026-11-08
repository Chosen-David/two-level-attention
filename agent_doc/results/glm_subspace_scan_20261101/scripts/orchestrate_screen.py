"""Launch queued GLM subspace screen configs onto free H20 GPUs.

State is stored in a JSON file so the tool is idempotent and can be called
repeatedly from a shell/poller. It never touches local GPUs.
"""
import argparse
import json
import os
import re
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
CONFIGS = json.loads((BASE / 'configs.json').read_text())['configs']
SCREEN = json.loads((BASE / 'configs.json').read_text())['screen']
RULER_LIMIT = str(SCREEN['ruler_limit'])
PER_TASK = str(SCREEN['per_task'])
GATE_MODE = 'full'
EPSILON = '0.05'
STATISTIC = 'mean'

HOSTS = {
    'h20-122': dict(host='10.238.122.187', port=8416, user='wangyuanshuo02',
                    password='zHWXp=eK5JvX~I|v', ngpus=4, prefix='/tmp',
                    python='/home/wangyuanshuo02/ffd-venv/bin/python',
                    pythonpath='/home/wangyuanshuo02/.local/pylibs_tf512:/home/wangyuanshuo02/tli-glm-origin-main',
                    code='/home/wangyuanshuo02/tli-glm-origin-main',
                    model='/home/wangyuanshuo02/models/huggingface.co/zai-org/GLM-4.7-Flash',
                    data='/home/wangyuanshuo02/datasets/LongBench/data',
                    ruler='/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER/65536/niah_multikey_2.jsonl',
                    launcher='/tmp/run_glm_subspace_worker_pool.sh'),
    'h20-139': dict(host='10.238.139.251', port=8416, user='wangyuanshuo02',
                    password='zHWXp=eK5JvX~I|v', ngpus=2, prefix='/tmp',
                    python='/home/wangyuanshuo02/ffd-venv/bin/python',
                    pythonpath='/home/wangyuanshuo02/.local/pylibs_tf512:/home/wangyuanshuo02/tli-glm-origin-main',
                    code='/home/wangyuanshuo02/tli-glm-origin-main',
                    model='/home/wangyuanshuo02/models/huggingface.co/zai-org/GLM-4.7-Flash',
                    data='/home/wangyuanshuo02/datasets/LongBench/data',
                    ruler='/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER/65536/niah_multikey_2.jsonl',
                    launcher='/tmp/run_glm_subspace_worker_pool.sh'),
    'gpu-125': dict(host='33.253.189.125', port=22, user='hadoop-aipnlp',
                    password='bK=/g2RN4a[0', ngpus=8, prefix='/tmp',
                    python='/home/hadoop-aipnlp/tli-glm-eval/venv/bin/python',
                    pythonpath='/tmp/tli-glm-origin-main',
                    code='/tmp/tli-glm-origin-main',
                    model='/home/hadoop-aipnlp/tli-glm-eval/models/GLM-4.7-Flash',
                    manifest='/home/hadoop-aipnlp/tli-glm-eval/model_files.json',
                    data='/home/hadoop-aipnlp/tli-glm-eval/data/LongBench',
                    ruler='/home/hadoop-aipnlp/tli-glm-eval/data/RULER/65536/niah_multikey_2.jsonl',
                    launcher='/tmp/run_glm_subspace_worker_pool.sh'),
    'h20-57': dict(host='10.245.58.57', port=8416, user='wangyuanshuo02',
                   password='zHWXp=eK5JvX~I|v', ngpus=2, prefix='/tmp',
                   python='/home/wangyuanshuo02/ffd-venv/bin/python',
                   pythonpath='/home/wangyuanshuo02/.local/pylibs_tf512:/home/wangyuanshuo02/tli-glm-origin-main',
                   code='/home/wangyuanshuo02/tli-glm-origin-main',
                   model='/home/wangyuanshuo02/models/huggingface.co/zai-org/GLM-4.7-Flash',
                   data='/home/wangyuanshuo02/datasets/LongBench/data',
                   ruler='/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER/65536/niah_multikey_2.jsonl',
                   launcher='/tmp/run_glm_subspace_worker_pool.sh'),
}


def ssh_cmd(host, remote_cmd, timeout=25):
    return ['sshpass', '-p', host['password'], 'ssh',
            '-p', str(host['port']), '-o', 'StrictHostKeyChecking=no',
            '-o', 'ConnectTimeout=8', f"{host['user']}@{host['host']}", remote_cmd]


def remote_call(host, remote_cmd, timeout=25):
    return subprocess.run(ssh_cmd(host, remote_cmd), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)


def free_gpus(host):
    cmd = ("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | "
           "grep -E '^[0-9]+,' | awk -F', ' '$2 < 1000 {print $1}'")
    res = remote_call(host, cmd)
    if res.returncode != 0:
        return []
    gpus = []
    for line in res.stdout.splitlines():
        match = re.match(r'^\s*(\d+)\s*$', line)
        if match:
            gpus.append(int(match.group(1)))
    return gpus


def state_path():
    return BASE / 'screen_launch_state.json'


def load_state():
    p = state_path()
    return json.loads(p.read_text()) if p.exists() else {'launched': {}}


def save_state(state):
    state_path().write_text(json.dumps(state, indent=2) + '\n')


def expand_dims(text):
    parts = []
    for tok in text.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if '-' in tok:
            a, b = tok.split('-', 1)
            parts.extend(range(int(a), int(b) + 1))
        else:
            parts.append(int(tok))
    return ','.join(str(x) for x in parts)


def launch(host_name, host, cfg, gpu):
    out = f"{host['prefix']}/tli-glm-screen-{cfg['label']}"
    script = host['launcher']
    exports = [
        f"export PYTHON={host['python']}",
        f"export PYTHONPATH={host['pythonpath']}",
        f"export CODE={host['code']}",
        f"export MODEL={host['model']}",
        f"export DATA={host['data']}",
        f"export RULER={host['ruler']}",
        f"export MANIFEST={host.get('manifest', host['model'] + '/model_files.json')}",
        f"export OUT={out}",
        f"export DIMS={expand_dims(cfg['dims'])}",
        f"export SCHEMA={cfg['schema']}",
        f"export LABEL={cfg['label']}",
        f"export GPUS={gpu}",
        f"export RULER_LIMIT={RULER_LIMIT}",
        f"export PER_TASK={PER_TASK}",
        f"export GATE_MODE={GATE_MODE}",
        f"export EPSILON={EPSILON}",
        f"export STATISTIC={STATISTIC}",
        f"rm -rf {out}",
        f"nohup {script} > {host['prefix']}/tli-glm-screen-{cfg['label']}-launcher.log 2>&1 & echo LAUNCHED:$!",
    ]
    remote = 'set -e; ' + '; '.join(exports)
    res = remote_call(host, remote, timeout=30)
    return res.returncode == 0, res.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    state = load_state()
    queued = [c for c in CONFIGS if c['label'] not in state['launched']]
    print(f"queued={len(queued)} launched={len(state['launched'])}")
    for host_name, host in HOSTS.items():
        free = free_gpus(host)
        print(f"{host_name} free={free}")
        for gpu in free:
            if not queued:
                break
            cfg = queued.pop(0)
            print(f"  launch {cfg['label']} on {host_name}:{gpu} {'DRY' if args.dry_run else ''}")
            if args.dry_run:
                continue
            ok, out = launch(host_name, host, cfg, gpu)
            if ok:
                state['launched'][cfg['label']] = dict(host=host_name, gpu=gpu,
                                                       out=f"{host['prefix']}/tli-glm-screen-{cfg['label']}")
                print(f"    -> {out}")
            else:
                print(f"    FAILED: {out}")
    save_state(state)


if __name__ == '__main__':
    main()
