# -*- coding: utf-8 -*-
"""sim_noisy12.py — simulation-side redo for the hardware comparison
(NO hardware usage):

  (a) noise-aware (density-matrix) inference on the SAME 12 origins as
      the hardware run — the missing honest simulator tier;
  (b) shot-noise bootstrap of the hardware row (multinomial resampling
      of the archived task counts, Richardson + splice per draw) to
      quantify the uncertainty of the hardware metrics.

Usage: python sim_noisy12.py --run_dir <dir> --capture <npz> \
         --reg hw_jobs/reg_P1high.json --out <json> [--draws 100]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import StandardScaler

from src.data_processing import (add_time_features, add_power_features,
                                 encode_wd)
from src.data_processing_winf import preprocess_winf, build_feature_cols_winf
from revision.run_revision import build_model, SeqForecastDataset
from revision.noisy_eval import NoisyTorchVQC


def build_all(run_dir, capture):
    cfg = json.load(open(os.path.join(run_dir, 'config.json')))
    seq_len, horizon = cfg.get('seq_len', 96), cfg.get('horizon', 1)
    pred_len = cfg['pred_len']
    df = preprocess_winf(os.path.join(REPO, 'dataset', 'winf_fam1.csv'),
                         freq='1hour', use_obs_lag=True)
    df = add_time_features(df)
    df = encode_wd(df)
    df = add_power_features(df)
    fcols = build_feature_cols_winf(df, use_obs_lag=True)
    df = df.dropna(subset=['date', 'power'] + fcols) \
        .sort_values('date').reset_index(drop=True)
    X_all = df[fcols].to_numpy(dtype=np.float32)
    y_all = df['power'].to_numpy(dtype=np.float32)
    n = len(df)
    n_train0, n_val = int(n * 0.7), int(n * 0.15)
    test_start = n_train0 + n_val
    ctx = seq_len + horizon - 1
    X_test_raw = X_all[max(0, test_start - ctx):n]
    y_test_raw = y_all[max(0, test_start - ctx):n]
    xsc = StandardScaler().fit(X_all[:test_start - n_val])
    ysc = StandardScaler().fit(y_all[:test_start - n_val].reshape(-1, 1))
    X_test = xsc.transform(X_test_raw)
    y_test = ysc.transform(y_test_raw.reshape(-1, 1)).ravel()
    dsf = SeqForecastDataset(X_test, y_test, seq_len, horizon, pred_len)
    common = len(X_test_raw) - seq_len - horizon - 12 + 2
    ds = Subset(dsf, list(range(common)))
    model = build_model('QFAM', len(fcols), pred_len)
    ck = torch.load(os.path.join(run_dir, 'best.pt'), map_location='cpu')
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    cap = np.load(capture)
    return model, ds, ysc, cap, pred_len


def run_patched(model, ds, idxs, lut, orig_fwd):
    """One forward pass over the selected origins with per-block LUT
    {bytes(x): z} patching; returns de-normalized predictions."""
    hits = [0, 0]

    def mk(k):
        def fwd(x):
            out = []
            for row in x.detach().cpu().numpy().astype(np.float64):
                z = lut[k].get(row.tobytes())
                if z is None:
                    z = orig_fwd[k](
                        torch.from_numpy(
                            row.astype(np.float32)).unsqueeze(0)
                    ).numpy()[0]
                else:
                    hits[k] += 1
                out.append(torch.from_numpy(
                    np.asarray(z, dtype=np.float32)))
            return torch.stack(out)
        return fwd

    vqcs = [model.flow_attn_layers[k].proj.vqc for k in range(2)]
    saved = [v.forward for v in vqcs]
    for k in range(2):
        vqcs[k].forward = mk(k)
    loader = DataLoader(Subset(ds, [int(i) for i in idxs]),
                        batch_size=len(idxs), shuffle=False)
    xb, _ = next(iter(loader))
    with torch.no_grad():
        out = model(xb).numpy()
    for k in range(2):
        vqcs[k].forward = saved[k]
    return out, hits


def metr(y_true, yp):
    e = y_true - yp
    return {'rmse': float(np.sqrt(np.mean(e ** 2))),
            'mae': float(np.mean(np.abs(e))),
            'r2': float(1 - np.sum(e ** 2) /
                        np.sum((y_true - y_true.mean()) ** 2))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--capture', required=True)
    ap.add_argument('--reg', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--draws', type=int, default=100)
    a = ap.parse_args()

    model, ds, ysc, cap, pred_len = build_all(a.run_dir, a.capture)
    idxs = cap['idxs']
    xs, zs_sim = cap['xs'], cap['zs_sim']
    O, T = xs.shape[1], xs.shape[2]
    y_true = cap['y_true_mw'].ravel()
    sig, mu = float(np.sqrt(ysc.var_[0])), float(ysc.mean_[0])
    orig_fwd = [model.flow_attn_layers[k].proj.vqc.forward for k
                in range(2)]

    # ---------- (a) noise-aware sim on the same origins ----------
    nvqc = [NoisyTorchVQC(model.flow_attn_layers[k].proj.vqc)
            for k in range(2)]
    z_noisy = np.zeros_like(zs_sim)
    with torch.no_grad():
        for k in range(2):
            flat = torch.from_numpy(
                xs[k].reshape(-1, 6).astype(np.float32))
            zn = []
            for i in range(0, len(flat), 512):
                zn.append(nvqc[k](flat[i:i + 512]).numpy())
            z_noisy[k] = np.concatenate(zn, 0).reshape(O, T, 6)
    print('mean |z_noisy - z_sim| =',
          float(np.mean(np.abs(z_noisy - zs_sim))))
    lut = [{}, {}]
    for k in range(2):
        for oi in range(O):
            for t in range(T):
                lut[k][xs[k, oi, t].tobytes()] = \
                    z_noisy[k, oi, t].astype(np.float32)
    out, hits = run_patched(model, ds, idxs, lut, orig_fwd)
    y_noisy = out.ravel() * sig + mu
    m_noisy = metr(y_true, y_noisy)
    print('noise-aware sim (12 origins):', m_noisy, 'hits', hits)

    # ---------- (b) shot-noise bootstrap of the hardware row ----------
    reg = json.load(open(a.reg))
    # counts per (oi,k,t,fold)
    cmap = {}
    for j in reg['jobs']:
        if j.get('taskState') != '3' or 'counts' not in j:
            continue
        for meta, cnt in zip([tuple(p) for p in j['programs']],
                             j['counts']):
            keys = [int(k, 16) for k in cnt['key']]
            vals = np.asarray(cnt['value'], dtype=np.float64)
            cmap[meta] = (keys, vals)

    def z_of(meta, keys, vals):
        tot = vals.sum()
        v = np.asarray(keys)
        z = np.zeros(6)
        for i in range(6):
            bit = (v >> i) & 1
            z[i] = np.sum(vals * (1 - 2 * bit)) / max(tot, 1e-12)
        return z

    rng = np.random.default_rng(20260916)
    boot = []
    keys_all = sorted(cmap)
    for d in range(a.draws):
        z_hw = {}
        for oi in range(O):
            for k in range(2):
                for t in range(T):
                    f1 = cmap.get((oi, k, t, 1))
                    if f1 is None:
                        continue
                    if (oi, k, t, 3) in cmap:
                        zs = []
                        for f in (1, 3):
                            keys, vals = cmap[(oi, k, t, f)]
                            if d > 0:
                                p = np.clip(vals, 0, None)
                                s = p.sum()
                                p = p / s if s > 0 else None
                                vals = rng.multinomial(1000, p) \
                                    if p is not None else vals
                            zs.append(z_of((oi, k, t, f), keys, vals))
                        # linear 2-point Richardson on folds {1,3};
                        # fold=2 is a different logical circuit (ring
                        # removed, CNOT^2 = I) and is excluded
                        z = np.clip((3 * zs[0] - zs[1]) / 2.0, -1, 1)
                    else:
                        keys, vals = f1
                        z = z_of((oi, k, t, 1), keys, vals)
                    z_hw[(k, oi, t)] = z.astype(np.float32)
        lutb = [{}, {}]
        for (k, oi, t), z in z_hw.items():
            lutb[k][xs[k, oi, t].tobytes()] = z
        outb, _ = run_patched(model, ds, idxs, lutb, orig_fwd)
        yb = outb.ravel() * sig + mu
        boot.append(metr(y_true, yb))
        if (d + 1) % 20 == 0:
            print(f'draw {d + 1}/{a.draws}: rmse={boot[-1]["rmse"]:.4f}')

    rm = np.array([b['rmse'] for b in boot])
    res = {
        'noise_aware_sim_12origins': m_noisy,
        'sim_noise_free_12origins': metr(y_true, cap['y_sim_mw'].ravel()),
        'hw_bootstrap': {
            'draws': a.draws,
            'rmse_pct': [float(np.percentile(rm, q))
                         for q in (5, 50, 95)],
            'rmse_mean': float(rm.mean()),
            'mae_pct': [float(np.percentile([b['mae'] for b in boot], q))
                        for q in (5, 50, 95)],
            'r2_pct': [float(np.percentile([b['r2'] for b in boot], q))
                       for q in (5, 50, 95)],
        },
    }
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps(res, indent=1))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
