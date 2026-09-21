# -*- coding: utf-8 -*-
"""hw_run.py — quantum execution + splicing for the hardware inference
validation of the trained QFAM checkpoint on WK_C180_2 (Wukong).

Phases (resumable, registry file per tag):
  submit  — build VQC circuits from the capture file, submit in chunks
  collect — poll taskDetail for each job, parse counts, save registry
  splice  — ZNE-extrapolate <Z>, patch the model, recompute predictions,
            metrics + accounting -> hw_eval_<tag>.json

Usage examples:
  python hw_run.py submit  --tag P1high --capture hw_capture_P1.npz \
      --mode high --folds 1,2,3 --mitigate --chunk 576
  python hw_run.py collect --tag P1high
  python hw_run.py splice  --tag P1high --capture hw_capture_P1.npz \
      --run_dir results_revision/QFAM_s42_P1_o0_20260915_111231
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

API_KEY = os.environ.get('OQCS_KEY', '')
if not API_KEY:
    raise SystemExit('Set the OQCS_KEY environment variable (Origin Quantum cloud API key) before hardware runs.')
DETAIL_URL = ('http://pyqanda-admin.qpanda.cn/oqcs/task/origin/'
              'taskDetail.json')
JOBS_DIR = os.path.join(REPO, 'hw_jobs')
SHOTS = 1000

from pyqpanda3.qcloud import QCloudService, QCloudOptions
import pyqpanda3.core as core
import urllib.request


# ------------------------------------------------------------------ utils
def reg_path(tag):
    return os.path.join(JOBS_DIR, f'reg_{tag}.json')


def load_reg(tag):
    p = reg_path(tag)
    if os.path.exists(p):
        return json.load(open(p))
    return {'jobs': []}


def save_reg(tag, reg):
    json.dump(reg, open(reg_path(tag), 'w'), indent=1)


def vqc_prog(x6, alpha, beta, fold, nbit=6):
    p = core.QProg()
    for i in range(nbit):
        p << core.RY(i, float(x6[i]))
    for i in range(nbit):
        p << core.RX(i, float(alpha[i]))
        p << core.RY(i, float(beta[i]))
    for i in range(nbit):
        j = (i + 1) % nbit
        for _ in range(fold):
            p << core.CNOT(i, j)
    p << core.measure(list(range(nbit)), list(range(nbit)))
    return p


def task_detail(task_id):
    body = json.dumps({'taskId': task_id}).encode()
    req = urllib.request.Request(DETAIL_URL, data=body, headers={
        'Authorization': f'oqcs_auth={API_KEY}',
        'task_mode': '1',
        'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


# ------------------------------------------------------------------ submit
def phase_submit(a):
    cap = np.load(a.capture)
    sel = json.load(open(os.path.join(REPO, 'results_revision',
                                      'selection.json')))
    xs = cap['xs']                    # [2, O, T, 6]
    angles = cap['angles']            # [2, 1, 6, 2]
    O, T = xs.shape[1], xs.shape[2]
    folds = [int(f) for f in a.folds.split(',')]
    block_ids = [int(b) for b in a.blocks.split(',')]
    origins = list(range(O)) if a.origins is None else \
        [int(o) for o in a.origins.split(',')]

    if a.mode == 'high':
        block = list(sel['selected_ring_order'])
    elif a.mode.startswith('random'):
        block = list(sel['random_controls'][int(a.mode[6:])])
    else:
        raise SystemExit(f'bad mode {a.mode}')

    # build program list: (meta, prog)
    items = []
    for oi in origins:
        for k in block_ids:
            alpha = angles[k][0, :, 0]
            beta = angles[k][0, :, 1]
            for t in range(T):
                for f in folds:
                    items.append(((oi, k, t, f),
                                  vqc_prog(xs[k, oi, t], alpha, beta, f)))
    print(f'total programs: {len(items)}  block={block}')

    svc = QCloudService(api_key=API_KEY)
    bk = svc.backend('WK_C180_2')
    reg = load_reg(a.tag)
    # skip programs already covered by a job that is done, running, or
    # queued (any state except failed '4'); only failed/uncovered
    # programs are resubmitted
    covered = set()
    for j in reg['jobs']:
        if j.get('taskState') in ('3', '2', '1') or \
                (j.get('taskState') is None and 'job_id' in j):
            for p in j['programs']:
                covered.add(tuple(p))
    chunk = a.chunk
    for ci in range(0, len(items), chunk):
        grp = items[ci:ci + chunk]
        metas = [g[0] for g in grp]
        if all(m in covered for m in metas):
            print(f'chunk {ci}: covered by an active/done job, skip')
            continue
        opts = QCloudOptions()
        opts.set_mapping(True)
        opts.set_optimization(False)     # preserve ZNE fold structure
        opts.set_amend(bool(a.mitigate))
        opts.set_specified_block(block)
        t0 = time.time()
        job = bk.run([g[1] for g in grp], SHOTS, opts)
        jid = job.job_id()
        reg['jobs'].append({'job_id': jid, 'programs': [list(m)
                                                        for m in metas],
                            'chunk': ci, 'mode': a.mode,
                            'mitigate': bool(a.mitigate),
                            'block': block, 'shots': SHOTS,
                            'submit_ts': time.time()})
        save_reg(a.tag, reg)
        print(f'chunk {ci}: job {jid} ({len(grp)} progs, '
              f'{time.time() - t0:.1f}s)')
        time.sleep(a.gap)
    print('submit phase done')


# ------------------------------------------------------------------ collect
def phase_collect(a):
    reg = load_reg(a.tag)
    import urllib.error
    for j in reg['jobs']:
        if j.get('taskState') == '3' and 'counts' in j:
            continue
        for attempt in range(30):
            try:
                d = task_detail(j['job_id'])
                break
            except Exception as e:
                print('detail err', e)
                time.sleep(min(60, 5 * (attempt + 1)))
        else:
            print('giving up on', j['job_id'])
            continue
        obj = d.get('obj') or {}
        st = str(obj.get('taskState', ''))
        j['taskState'] = st
        j['totalTime'] = obj.get('totalTime')
        j['qpuRunTime'] = obj.get('qpuRunTime')
        if st == '3':
            try:
                res = [json.loads(s) for s in obj['taskResult']]
                j['counts'] = res
                print(f"job {j['job_id'][:8]} DONE "
                      f"({len(res)} results, qpu {obj.get('qpuRunTime')} ms)")
            except Exception as e:
                print('parse err', e)
        elif st == '4':
            print(f"job {j['job_id'][:8]} FAILED: "
                  f"{obj.get('errInfo', '?')}")
        else:
            print(f"job {j['job_id'][:8]} state={st}")
        save_reg(a.tag, reg)
    pending = [j for j in reg['jobs'] if j.get('taskState') != '3']
    print(f'collect done; pending jobs: {len(pending)}')


# ------------------------------------------------------------------ splice
def phase_splice(a):
    cap = np.load(a.capture)
    reg = load_reg(a.tag)
    xs, zs_sim = cap['xs'], cap['zs_sim']
    angles = cap['angles']
    idxs = cap['idxs']
    y_true = cap['y_true_mw']
    y_sim = cap['y_sim_mw']
    O, T = xs.shape[1], xs.shape[2]
    pred_len = int(cap['pred_len'])

    # gather z per (oi, k, t, fold)
    zmap = {}
    for j in reg['jobs']:
        if j.get('taskState') != '3' or 'counts' not in j:
            continue
        metas = [tuple(p) for p in j['programs']]
        for (oi, k, t, f), cnt in zip(metas, j['counts']):
            # <Z_i> = sum_v p(v) * (1 - 2*bit_i(v))
            z = np.zeros(6)
            tot = 0.0
            for key, val in zip(cnt['key'], cnt['value']):
                v = int(key, 16)
                p = float(val)
                tot += p
                for i in range(6):
                    z[i] += p * (1 - 2 * ((v >> i) & 1))
            if tot > 0:
                z /= tot                     # normalize (amend quasi-probs)
            zmap[(oi, k, t, f)] = z
    print('collected z entries:', len(zmap))

    # ZNE per (oi, k, t): available folds from registry
    folds_avail = sorted({f for (_, _, _, f) in zmap})
    print('folds available:', folds_avail)
    z_hw = np.zeros_like(zs_sim)
    cov = np.zeros_like(zs_sim)
    for oi in range(O):
        for k in range(2):
            for t in range(T):
                if (oi, k, t, 1) not in zmap:
                    continue
                if (oi, k, t, 3) in zmap:
                    # linear 2-point Richardson on folds {1, 3} (same
                    # logical circuit, 1x / 3x CNOT noise):
                    #   z(0) = z1 + (z1 - z3) * 1 / (1 - 3)
                    #        = (3*z1 - z3) / 2
                    # NOTE: fold=2 (CNOT^2 = I) realizes a *different*
                    # logical circuit (ring removed) and is excluded.
                    z = (3 * zmap[(oi, k, t, 1)]
                         - zmap[(oi, k, t, 3)]) / 2.0
                else:
                    z = zmap[(oi, k, t, 1)]
                z = np.clip(z, -1, 1)
                z_hw[k, oi, t] = z
                cov[k, oi, t] = 1
    print('coverage: %.3f' % cov.mean())

    # ---- classical re-run with patched VQC ----
    import torch
    from torch.utils.data import DataLoader, Subset
    from sklearn.preprocessing import StandardScaler
    from src.data_processing import (add_time_features, add_power_features,
                                     encode_wd)
    from src.data_processing_winf import (preprocess_winf,
                                          build_feature_cols_winf)
    from revision.run_revision import build_model, SeqForecastDataset

    cfg = json.load(open(os.path.join(a.run_dir, 'config.json')))
    seq_len = cfg.get('seq_len', 96)
    horizon = cfg.get('horizon', 1)

    df = preprocess_winf(os.path.join(REPO, 'dataset', 'winf_fam1.csv'),
                         freq='1hour', use_obs_lag=True)
    df = add_time_features(df)
    df = encode_wd(df)
    df = add_power_features(df)
    feature_cols = build_feature_cols_winf(df, use_obs_lag=True)
    df = df.dropna(subset=['date', 'power'] + feature_cols) \
        .sort_values('date').reset_index(drop=True)
    X_all = df[feature_cols].to_numpy(dtype=np.float32)
    y_all = df['power'].to_numpy(dtype=np.float32)
    n = len(df)
    n_train0, n_val = int(n * 0.7), int(n * 0.15)
    test_start = n_train0 + n_val
    ctx = seq_len + horizon - 1
    X_test_raw = X_all[max(0, test_start - ctx):n]
    y_test_raw = y_all[max(0, test_start - ctx):n]
    x_scaler = StandardScaler().fit(X_all[:test_start - n_val])
    y_scaler = StandardScaler().fit(
        y_all[:test_start - n_val].reshape(-1, 1))
    X_test = x_scaler.transform(X_test_raw)
    y_test = y_scaler.transform(y_test_raw.reshape(-1, 1)).ravel()
    ds_full = SeqForecastDataset(X_test, y_test, seq_len, horizon, pred_len)
    common = len(X_test_raw) - seq_len - horizon - 12 + 2
    ds = Subset(ds_full, list(range(common)))
    device = torch.device('cpu')
    model = build_model('QFAM', len(feature_cols), pred_len).to(device)
    ck = torch.load(os.path.join(a.run_dir, 'best.pt'),
                    map_location=device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    # lookup tables: bytes(x) -> z_hw  (per block); optionally restricted
    # to a subset of origins (for same-origin comparisons)
    keep = None
    if a.origins:
        keep = {int(o) for o in a.origins.split(',')}
    lut = [dict(), dict()]
    for k in range(2):
        for oi in range(O):
            if keep is not None and oi not in keep:
                continue
            for t in range(T):
                key = xs[k, oi, t].tobytes()
                lut[k][key] = z_hw[k, oi, t].astype(np.float32)

    # origins actually evaluated on hardware (any coverage), further
    # restricted by --origins; ALL reported metrics use exactly these
    cov_oi = cov.any(axis=(0, 2, 3))         # [O] bool
    sel = [oi for oi in range(O)
           if cov_oi[oi] and (keep is None or oi in keep)]
    if not sel:
        raise SystemExit('no covered origins selected')
    print('metric origins (%d):' % len(sel), sel)

    orig_vqc = [model.flow_attn_layers[k].proj.vqc for k in range(2)]
    orig_fwd = [v.forward for v in orig_vqc]
    patched_hits = [0, 0]
    patched_miss = [0, 0]

    def make_patched(k):
        def fwd(x):
            out = []
            arr = x.detach().cpu().numpy().astype(np.float64)
            for row in arr:
                z = lut[k].get(row.tobytes())
                if z is None:
                    patched_miss[k] += 1
                    z = orig_fwd[k](
                        torch.from_numpy(
                            row.astype(np.float32)).unsqueeze(0)
                    ).numpy()[0]
                else:
                    patched_hits[k] += 1
                out.append(torch.from_numpy(np.asarray(z, dtype=np.float32)))
            return torch.stack(out).to(x.device)
        return fwd

    for k in range(2):
        orig_vqc[k].forward = make_patched(k)

    # forward only the selected (hardware-covered) origins
    sel_ds = Subset(ds, [int(idxs[oi]) for oi in sel])
    loader = DataLoader(sel_ds, batch_size=len(sel), shuffle=False)
    xb, yb = next(iter(loader))
    with torch.no_grad():
        out = model(xb).numpy()
    y_hw = out * np.sqrt(y_scaler.var_[0]) + y_scaler.mean_[0]
    y_true_s = y_true[sel]
    y_sim_s = y_sim[sel]
    print('patched hits/miss:', patched_hits, patched_miss)

    # metrics (strictly on the hardware-covered origin subset)
    def metr(yp):
        e = y_true_s - yp
        rmse = float(np.sqrt(np.mean(e ** 2)))
        mae = float(np.mean(np.abs(e)))
        r2 = float(1 - np.sum(e ** 2) /
                   np.sum((y_true_s - y_true_s.mean()) ** 2))
        return {'rmse': rmse, 'mae': mae, 'r2': r2}

    sel_cov = np.zeros_like(cov)
    sel_cov[:, sel, :, :] = cov[:, sel, :, :]
    dz = float(np.mean(np.abs(z_hw - zs_sim)[sel_cov > 0]))
    res = {
        'tag': a.tag, 'mode': reg['jobs'][0]['mode'] if reg['jobs'] else None,
        'mitigate': reg['jobs'][0]['mitigate'] if reg['jobs'] else None,
        'folds': folds_avail, 'n_origins': int(O),
        'n_origin_sel': len(sel),
        'coverage': float(cov.mean()),
        'metrics_hw': metr(y_hw), 'metrics_sim_same_origins': metr(y_sim_s),
        'mean_abs_dz_vs_sim': dz,
        'y_hw_mw': y_hw.tolist(), 'y_sim_mw': y_sim_s.tolist(),
        'y_true_mw': y_true_s.tolist(),
        'accounting': {
            'jobs': len(reg['jobs']),
            'programs': sum(len(j['programs']) for j in reg['jobs']),
            'shots_per_program': SHOTS,
            'total_shots': sum(len(j['programs']) for j in reg['jobs'])
            * SHOTS,
            'total_time_ms': sum(j.get('totalTime') or 0
                                 for j in reg['jobs']),
            'qpu_time_ms': sum(j.get('qpuRunTime') or 0
                               for j in reg['jobs']),
        },
    }
    out_path = os.path.join(REPO, 'results_revision',
                            f'hw_eval_{a.tag}.json')
    json.dump(res, open(out_path, 'w'), indent=1)
    np.savez_compressed(os.path.join(REPO, 'results_revision',
                                     f'hw_z_{a.tag}.npz'),
                        z_hw=z_hw, z_sim=zs_sim, cov=cov)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ('y_hw_mw', 'y_sim_mw', 'y_true_mw')},
                     indent=1))
    print('wrote', out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['submit', 'collect', 'splice'])
    ap.add_argument('--tag', required=True)
    ap.add_argument('--capture')
    ap.add_argument('--run_dir')
    ap.add_argument('--mode', default='high')
    ap.add_argument('--folds', default='1,2,3')
    ap.add_argument('--blocks', default='0,1')
    ap.add_argument('--mitigate', action='store_true')
    ap.add_argument('--chunk', type=int, default=576)
    ap.add_argument('--gap', type=float, default=5.0)
    ap.add_argument('--origins')
    a = ap.parse_args()
    os.makedirs(JOBS_DIR, exist_ok=True)
    if a.phase == 'submit':
        phase_submit(a)
    elif a.phase == 'collect':
        phase_collect(a)
    else:
        phase_splice(a)


if __name__ == '__main__':
    main()
