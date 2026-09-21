# -*- coding: utf-8 -*-
"""
revision/run_all.py — orchestrates the full revision experiment matrix on the
server with two concurrent workers (GPUs 2 and 3 of node03).

Matrix:
  main runs:        {QFAM, QFAM_C, QFAM_R, FAM, mlp, lstm, transformer}
                    x seeds {42, 0, 1, 2, 3} x pred_len {1, 3}
  rolling origins:  {QFAM, QFAM_C, FAM} x offset_days {30, 60, 90}
                    x pred_len {1, 3} (seed 42)

Skips runs whose DONE marker already exists. One retry on failure.
"""
import os
import subprocess
import sys
import threading
import time
import queue

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(os.path.dirname(HERE), 'results_revision')
# GPU allow-list can be overridden with e.g.  python run_all.py --gpus 0,2,5
# Default: auto-pick the five GPUs with the most free memory (>= MIN_FREE_MB)
# at each job launch, so long-running neighbor jobs cannot starve us.
MIN_FREE_MB = 12000
N_WORKERS = 8
DEFAULT_GPUS = None
JOB_GB = 12  # conservative per-job GPU footprint used by the scheduler

MODELS = ['QFAM', 'QFAM_C', 'QFAM_R', 'FAM', 'mlp', 'lstm', 'transformer']
SEEDS = [42, 0, 1, 2, 3]
PREDS = [1, 3]
ROLL_MODELS = ['QFAM', 'QFAM_C', 'FAM']
ROLL_OFFSETS = [30, 60, 90]


def jobs():
    js = []
    for m in MODELS:
        for s in SEEDS:
            for p in PREDS:
                js.append((m, s, p, 0))
    for m in ROLL_MODELS:
        for o in ROLL_OFFSETS:
            for p in PREDS:
                js.append((m, 42, p, o))
    return js


def tag(m, s, p, o, exp):
    return f'{m}_s{s}_P{p}_o{o}_{exp}'


_gpu_lock = threading.Lock()
_gpu_allow = None
_gpu_busy = {}  # gpu -> number of our running jobs


def query_gpus():
    """Return {gpu_id: free_mb} from nvidia-smi (best effort)."""
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=30)
        out = {}
        for line in r.stdout.strip().splitlines():
            idx, free = line.split(',')
            out[idx.strip()] = int(free.strip())
        return out
    except Exception:
        return {}


def pick_gpu():
    """Pick the GPU with the most *effective* free memory across the whole
    node (nvidia-smi free minus our own running jobs' reservation).
    GPUs whose raw free memory is below MIN_FREE_MB are excluded; if none
    passes the floor, returns None (caller waits — backpressure)."""
    with _gpu_lock:
        free = query_gpus()
        if not free:
            return None
        ok = {g: f - _gpu_busy.get(g, 0) * JOB_GB * 1024
              for g, f in free.items() if f >= MIN_FREE_MB}
        return max(ok, key=ok.get) if ok else None


def claim_gpu(g):
    with _gpu_lock:
        _gpu_busy[g] = _gpu_busy.get(g, 0) + 1


def release_gpu(g):
    with _gpu_lock:
        _gpu_busy[g] = max(0, _gpu_busy.get(g, 0) - 1)


def done_marker(m, s, p, o):
    for d in os.listdir(OUT_ROOT) if os.path.isdir(OUT_ROOT) else []:
        if d.startswith(f'{m}_s{s}_P{p}_o{o}_'):
            if os.path.exists(os.path.join(OUT_ROOT, d, 'DONE')):
                return True
    return False


def running_elsewhere(m, s, p, o):
    """True if a run_revision process with this exact config is alive
    (e.g. orphaned from a previous orchestrator). NOTE: the pattern is
    prefixed with the script name so pgrep does not mistake it for an
    option (patterns starting with '--' silently fail)."""
    pat = f'run_revision.py --model {m} --seed {s} --pred_len {p} --offset_days {o} '
    try:
        r = subprocess.run(['pgrep', '-f', pat], capture_output=True,
                           text=True)
        pids = [ln for ln in r.stdout.split() if ln]
        return len(pids) > 0
    except Exception:
        return False


def worker(wid, q, log_lock):
    idle_spins = 0
    while True:
        try:
            m, s, p, o = q.get_nowait()
        except queue.Empty:
            if idle_spins >= 3:
                return
            time.sleep(60)
            idle_spins += 1
            continue
        idle_spins = 0
        if done_marker(m, s, p, o):
            with log_lock:
                print(f'[w{wid}] SKIP (done) {m} s{s} P{p} o{o}', flush=True)
            q.task_done()
            continue
        if running_elsewhere(m, s, p, o):
            # someone else (orphan) is already computing this job: requeue
            q.put((m, s, p, o))
            q.task_done()
            time.sleep(120)
            continue
        exp = time.strftime('%Y%m%d_%H%M%S')
        cmd = [sys.executable, os.path.join(HERE, 'run_revision.py'),
               '--model', m, '--seed', str(s), '--pred_len', str(p),
               '--offset_days', str(o), '--exp_id', exp]
        attempt = 0
        while attempt < 3:
            gpu = pick_gpu()
            if gpu is None:
                # backpressure: no GPU has enough free memory right now
                with log_lock:
                    print(f'[w{wid}] WAIT (no GPU >= {MIN_FREE_MB} MB free) '
                          f'{m} s{s} P{p} o{o}', flush=True)
                time.sleep(180)
                continue
            attempt += 1
            claim_gpu(gpu)
            env = dict(os.environ)
            env['CUDA_VISIBLE_DEVICES'] = gpu
            env.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
            with log_lock:
                print(f'[w{wid}/gpu{gpu}] START (try {attempt}) {m} s{s} '
                      f'P{p} o{o}', flush=True)
            r = subprocess.run(cmd, env=env, cwd=os.path.dirname(HERE),
                               capture_output=True, text=True)
            release_gpu(gpu)
            if r.returncode == 0:
                with log_lock:
                    print(f'[w{wid}/gpu{gpu}] OK    {m} s{s} P{p} o{o}',
                          flush=True)
                break
            with log_lock:
                print(f'[w{wid}/gpu{gpu}] FAIL  {m} s{s} P{p} o{o}\n'
                      f'{r.stderr[-2000:]}', flush=True)
            time.sleep(120)  # let neighbor jobs settle before re-picking GPU
        q.task_done()


def main():
    global _gpu_allow
    ap_gpus = None
    for i, a in enumerate(sys.argv):
        if a == '--gpus' and i + 1 < len(sys.argv):
            ap_gpus = sys.argv[i + 1].split(',')
    _gpu_allow = [g.strip() for g in ap_gpus] if ap_gpus else None
    os.makedirs(OUT_ROOT, exist_ok=True)
    free = query_gpus()
    print(f'GPU snapshot at launch: {free} (scheduler is dynamic)',
          flush=True)
    q = queue.Queue()
    for j in jobs():
        q.put(j)
    log_lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(i, q, log_lock))
               for i in range(N_WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print('ALL JOBS PROCESSED')


if __name__ == '__main__':
    main()
