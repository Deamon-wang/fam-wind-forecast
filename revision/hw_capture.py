# -*- coding: utf-8 -*-
"""hw_capture.py — classical-side capture for hardware inference:

Load the trained QFAM checkpoint, run the exact test pipeline, hook both
VQC instances to capture their 6-dim inputs (encoded angles) and exact
simulator outputs. Save per-origin slices for quantum execution +
splicing.

Usage:
  python hw_capture.py --run_dir results_revision/QFAM_s42_P1_o0_... \
      --n_origins 12 --out results_revision/hw_capture_P1.npz
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import StandardScaler

from src.data_processing import (add_time_features, add_power_features,
                                 encode_wd)
from src.data_processing_winf import preprocess_winf, build_feature_cols_winf
from revision.run_revision import build_model, SeqForecastDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--n_origins', type=int, default=12)
    ap.add_argument('--out', required=True)
    ap.add_argument('--csv_path',
                    default=os.path.join(REPO, 'dataset', 'winf_fam1.csv'))
    args = ap.parse_args()

    cfg = __import__('json').load(open(os.path.join(args.run_dir,
                                                    'config.json')))
    pred_len = cfg['pred_len']
    seq_len = cfg.get('seq_len', 96)
    horizon = cfg.get('horizon', 1)
    print('pred_len', pred_len, 'seq_len', seq_len)

    # ---------------- data pipeline (audited copy) ----------------
    df = preprocess_winf(args.csv_path, freq='1hour', use_obs_lag=True)
    df = add_time_features(df)
    df = encode_wd(df)
    df = add_power_features(df)
    feature_cols = build_feature_cols_winf(df, use_obs_lag=True)
    required = ['date', 'power'] + feature_cols
    df = df.dropna(subset=required).sort_values('date').reset_index(drop=True)

    X_all = df[feature_cols].to_numpy(dtype=np.float32)
    y_all = df['power'].to_numpy(dtype=np.float32)
    date_all = df['date'].reset_index(drop=True)
    n = len(df)
    n_train0 = int(n * 0.7)
    n_val = int(n * 0.15)
    test_start = n_train0 + n_val
    test_end = n
    ctx = seq_len + horizon - 1
    X_test_raw = X_all[max(0, test_start - ctx):test_end]
    y_test_raw = y_all[max(0, test_start - ctx):test_end]
    date_test_raw = date_all[max(0, test_start - ctx):test_end] \
        .reset_index(drop=True)

    # scalers must be fit on train only (same as run_revision)
    X_train_raw = X_all[:test_start - n_val]
    y_train_raw = y_all[:test_start - n_val]
    x_scaler = StandardScaler().fit(X_train_raw)
    y_scaler = StandardScaler().fit(y_train_raw.reshape(-1, 1))
    X_test = x_scaler.transform(X_test_raw)
    y_test = y_scaler.transform(y_test_raw.reshape(-1, 1)).ravel()

    test_ds_full = SeqForecastDataset(X_test, y_test, seq_len, horizon,
                                      pred_len)
    max_eval_pred_len = 12
    common_n_test = len(X_test_raw) - seq_len - horizon - max_eval_pred_len + 2
    test_ds = Subset(test_ds_full, list(range(common_n_test)))
    loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                        drop_last=False)
    print('n_test samples:', len(test_ds))

    # ---------------- model ----------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model('QFAM', len(feature_cols), pred_len).to(device)
    ck = torch.load(os.path.join(args.run_dir, 'best.pt'), map_location=device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    vqcs = [model.flow_attn_layers[k].proj.vqc for k in range(2)]

    # capture per-block: list over batches of [B*96, 6]
    cap_x = [[], []]
    cap_z = [[], []]
    hooks = []

    def mk_hook(k):
        def hook(mod, inp, out):
            x = inp[0].detach()
            cap_x[k].append(x.cpu().numpy().astype(np.float64))
            cap_z[k].append(out.detach().cpu().numpy().astype(np.float64))
        return hook

    for k in range(2):
        hooks.append(vqcs[k].register_forward_hook(mk_hook(k)))

    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            preds.append(out)
            trues.append(yb.numpy())
    for h in hooks:
        h.remove()

    y_pred = np.concatenate(preds)          # [N, pred_len] scaled
    y_true = np.concatenate(trues)
    # de-normalize
    y_pred_mw = y_pred * np.sqrt(y_scaler.var_[0]) + y_scaler.mean_[0]
    y_true_mw = y_true * np.sqrt(y_scaler.var_[0]) + y_scaler.mean_[0]

    for k in range(2):
        cap_x[k] = np.concatenate(cap_x[k], 0)   # [N*96, 6]
        cap_z[k] = np.concatenate(cap_z[k], 0)
    n_test = y_pred.shape[0]
    for k in range(2):
        assert cap_x[k].shape[0] == n_test * seq_len, \
            (cap_x[k].shape, n_test * seq_len)

    # ---------------- select origins ----------------
    idxs = np.unique(np.linspace(0, n_test - 1, args.n_origins)
                     .round().astype(int))
    print('origin sample indices:', idxs.tolist())

    xs = np.zeros((2, len(idxs), seq_len, 6))
    zs = np.zeros((2, len(idxs), seq_len, 6))
    for oi, si in enumerate(idxs):
        for k in range(2):
            xs[k, oi] = cap_x[k][si * seq_len:(si + 1) * seq_len]
            zs[k, oi] = cap_z[k][si * seq_len:(si + 1) * seq_len]

    # angle range sanity
    print('x range: [%.3f, %.3f]' % (xs.min(), xs.max()))
    np.savez_compressed(args.out, xs=xs, zs_sim=zs, idxs=idxs,
                        y_true_mw=y_true_mw[idxs], y_sim_mw=y_pred_mw[idxs],
                        pred_len=pred_len, seq_len=seq_len,
                        angles=np.stack([v.paras.detach().cpu().numpy()
                                         for v in vqcs]))
    d = date_test_raw.iloc[idxs]
    print('origin dates:', d.iloc[0], '..', d.iloc[-1])
    rmse = float(np.sqrt(np.mean((y_true_mw[idxs] - y_pred_mw[idxs]) ** 2)))
    print('sim RMSE on origins: %.3f MW' % rmse)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
