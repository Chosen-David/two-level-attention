"""Serialized 30-minute Codex supervision; GPU workers remain independent."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--project', required=True)
    ap.add_argument('--interval', type=int, default=1800)
    ap.add_argument('--prompt', required=True)
    args = ap.parse_args()
    if args.interval < 60:
        raise ValueError('interval must be >=60 seconds')
    out = Path(args.out)
    directory = out / 'ai_supervision'
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'supervisor.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    codex = shutil.which('codex')
    if not codex:
        raise RuntimeError('codex executable unavailable')
    while not (directory / 'STOP').exists():
        started = time.time()
        name = time.strftime('%Y%m%dT%H%M%S', time.gmtime(started))
        report = directory / (name + '.md')
        logpath = directory / (name + '.log')
        state = dict(pid=os.getpid(), interval_seconds=args.interval,
                     started_at=started, next_check_at=started + args.interval,
                     status='running_ai_review', report=str(report), log=str(logpath))
        atomic(directory / 'status.json', state)
        try:
            with logpath.open('a') as log:
                proc = subprocess.Popen([
                    codex, 'exec', '--ephemeral', '-s', 'danger-full-access',
                    '-c', 'approval_policy="never"', '-C', args.project,
                    '-o', str(report), '-'], stdin=subprocess.PIPE,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state['review_pid'] = proc.pid
                atomic(directory / 'status.json', state)
                try:
                    proc.communicate(Path(args.prompt).read_bytes(), timeout=args.interval - 30)
                    state.update(returncode=proc.returncode, status='waiting' if proc.returncode == 0 else 'review_failed_retry_next_cycle')
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try: proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                    state.update(status='review_timeout_retry_next_cycle')
        except Exception as exc:
            state.update(status='review_failed_retry_next_cycle', error=repr(exc))
        state['finished_at'] = time.time()
        if (directory / 'COMPLETE.json').exists():
            complete = json.loads((directory / 'COMPLETE.json').read_text())
            evidence = complete.get('independent_verification_path')
            if complete.get('status') == 'verified_complete' and evidence and Path(evidence).is_file():
                state['status'] = 'verified_complete'
        atomic(directory / 'status.json', state)
        with (directory / 'history.jsonl').open('a') as log:
            log.write(json.dumps(state) + '\n')
        if state['status'] == 'verified_complete':
            return
        while time.time() < started + args.interval:
            if (directory / 'STOP').exists(): return
            time.sleep(min(10, max(0.01, started + args.interval - time.time())))


if __name__ == '__main__':
    main()
