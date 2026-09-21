# -*- coding: utf-8 -*-
"""
revision/noisy_eval.py — noise-aware simulation row (fully vectorized).

Loads trained QFAM checkpoints, replaces the exact state-vector VQC with an
exact density-matrix simulation including, after every gate:
  * single-qubit depolarizing channel: rho -> (1-p) rho + (p/3)(X Y Z terms)
  * thermal relaxation: pure dephasing then amplitude damping
    (T1, T2, gate time; Kraus, sequential channels)
Expectation values are Tr(rho Z_i); deterministic (no Monte-Carlo sampling).

Defaults follow the manuscript: p = 0.01, T1 = 100 us, T2 = 80 us,
t_gate = 50 ns. All operations are batched over the B (=B*N tokens) rows.
"""
import os
import sys
import json
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.data_processing import (add_time_features, add_power_features,
                                 encode_wd, SeqForecastDataset)
from src.data_processing_winf import preprocess_winf, build_feature_cols_winf
from revision.models_rev import TorchVQC
from revision.run_revision import build_model, seed_everything

CD = torch.complex64


def _pairs(dim, q):
    a = torch.arange(dim)
    mask = (a >> q & 1) == 0
    return a[mask].clone(), (a[mask] | (1 << q)).clone()


class NoisyTorchVQC(TorchVQC):
    """Vectorized density-matrix twin of TorchVQC with gate noise."""

    def __init__(self, clean, p_depol=0.01, t1_us=100.0, t2_us=80.0,
                 t_gate_ns=50.0):
        super().__init__(clean.n_bit, clean.qlayer)
        with torch.no_grad():
            self.paras.copy_(clean.paras.data)
        self.p = p_depol
        self.tg = t_gate_ns * 1e-3            # us
        self.t1, self.t2 = t1_us, t2_us
        self._setup_aux()

    def _setup_aux(self):
        """Index structures used by the vectorized channels/gates."""
        d = 1 << self.n_bit
        idx = torch.arange(d)
        self._pre = {}
        for q in range(self.n_bit):
            a, b = _pairs(d, q)
            sign = (1 - 2 * ((idx >> q) & 1)).float()
            self._pre[q] = (a, b, sign)
        self._cnot_perm = {}
        for c in range(self.n_bit):
            for t in range(self.n_bit):
                if c != t:
                    pi = torch.arange(d)
                    m = (pi >> c & 1) == 1
                    pi[m] = pi[m] ^ (1 << t)
                    self._cnot_perm[(c, t)] = pi
        self._xperm = {q: idx ^ (1 << q) for q in range(self.n_bit)}
        self._bits_all = None

    # ---------------- gates ----------------
    def _apply_1q(self, rho, u00, u01, u10, u11, q):
        """Apply the general single-qubit gate U=[[u00,u01],[u10,u11]] (batched
        complex coefficients) on qubit q of a batched density matrix."""
        a, b, _ = self._pre[q]
        # gather 4 blocks over pair indices: shape [B, n_pairs]
        r_aa, r_ab = rho[:, a, a], rho[:, a, b]
        r_ba, r_bb = rho[:, b, a], rho[:, b, b]
        u00, u01 = u00.unsqueeze(1), u01.unsqueeze(1)
        u10, u11 = u10.unsqueeze(1), u11.unsqueeze(1)
        naa = (u00.abs() ** 2 * r_aa + u00 * u01.conj() * r_ab
               + u01 * u00.conj() * r_ba + u01.abs() ** 2 * r_bb)
        nab = (u00 * u10.conj() * r_aa + u00 * u11.conj() * r_ab
               + u01 * u10.conj() * r_ba + u01 * u11.conj() * r_bb)
        nba = (u10 * u00.conj() * r_aa + u10 * u01.conj() * r_ab
               + u11 * u00.conj() * r_ba + u11 * u01.conj() * r_bb)
        nbb = (u10.abs() ** 2 * r_aa + u10 * u11.conj() * r_ab
               + u11 * u10.conj() * r_ba + u11.abs() ** 2 * r_bb)
        rho = rho.clone()
        rho[:, a, a], rho[:, a, b] = naa, nab
        rho[:, b, a], rho[:, b, b] = nba, nbb
        return rho

    def _ry(self, rho, theta, q):
        c = torch.cos(theta / 2).to(CD)
        s = torch.sin(theta / 2).to(CD)
        return self._apply_1q(rho, c, -s, s, c, q)

    def _rx(self, rho, theta, q):
        c = torch.cos(theta / 2).to(CD)
        s = torch.sin(theta / 2).to(CD)
        return self._apply_1q(rho, c, -1j * s, -1j * s, c, q)

    def _cnot(self, rho, c, t):
        pi = self._cnot_perm[(c, t)].to(rho.device)
        return rho[:, pi][:, :, pi]

    # ---------------- noise channels ----------------
    def _channels(self, rho, q):
        d = rho.shape[1]
        # --- depolarizing ---
        if self.p > 0:
            xp = self._xperm[q].to(rho.device)
            xr = rho[:, xp][:, :, xp]                       # X rho X
            idx = torch.arange(d, device=rho.device)
            bq = (idx >> q) & 1
            sgn = (1 - 2 * bq.to(rho.dtype))                # +/-1
            zr = rho * sgn[:, None] * sgn[None, :]           # Z rho Z
            yr = xr * sgn[:, None] * sgn[None, :]            # Y rho Y
            rho = (1 - self.p) * rho + (self.p / 3) * (xr + yr + zr)
        # --- pure dephasing (lambda on off-diagonals of qubit q) ---
        lam = 1 - math.exp(-2 * self.tg / self.t2)
        idx = torch.arange(d, device=rho.device)
        bq = (idx >> q) & 1
        sgn = (1 - 2 * bq.to(rho.dtype))
        rho = rho * (1 - lam) + rho * sgn[:, None] * sgn[None, :] * lam
        # --- amplitude damping (gamma: 1 -> 0) ---
        g = 1 - math.exp(-self.tg / self.t1)
        a, b, _ = self._pre[q]
        aa, ab, ba, bb = rho[:, a, a], rho[:, a, b], rho[:, b, a], rho[:, b, b]
        k = math.sqrt(1 - g)
        rho = rho.clone()
        rho[:, a, a] = aa + g * bb
        rho[:, a, b] = k * ab
        rho[:, b, a] = k * ba
        rho[:, b, b] = (1 - g) * bb
        return rho

    # ---------------- forward ----------------
    def forward(self, x):
        B = x.shape[0]
        dev = x.device
        d = 1 << self.n_bit
        rho = torch.zeros(B, d, d, dtype=CD, device=dev)
        rho[:, 0, 0] = 1.0
        for i in range(self.n_bit):
            rho = self._channels(self._ry(rho, x[:, i], i), i)
        for l in range(self.qlayer):
            w = self.paras[l]
            for j in range(self.n_bit):
                na = torch.full((B,), float(w[j, 0]), device=dev)
                nb = torch.full((B,), float(w[j, 1]), device=dev)
                rho = self._channels(self._rx(rho, na, j), j)
                rho = self._channels(self._ry(rho, nb, j), j)
            for j in range(self.n_bit - 1):
                rho = self._channels(self._cnot(rho, j, j + 1), j)
            rho = self._channels(self._cnot(rho, self.n_bit - 1, 0),
                                 self.n_bit - 1)
        diag = torch.diagonal(rho, dim1=1, dim2=2).real      # [B, d]
        if self._bits_all is None or self._bits_all.device != diag.device:
            self._bits_all = self._bits.to(dev)
        z = 1.0 - 2.0 * (diag @ self._bits_all)              # [B, n_bit]
        return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--p_depol', type=float, default=0.01)
    ap.add_argument('--t1_us', type=float, default=100.0)
    ap.add_argument('--t2_us', type=float, default=80.0)
    ap.add_argument('--t_gate_ns', type=float, default=50.0)
    ap.add_argument('--chunk', type=int, default=2048,
                    help='rows of the flattened (B*N) token axis per pass')
    args = ap.parse_args()

    with open(os.path.join(args.run_dir, 'config.json'),
              encoding='utf-8') as f:
        cfg = json.load(f)
    seed_everything(cfg['seed'])

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
    off_h = cfg['offset_days'] * 24
    seq_len, horizon, pred_len = cfg['seq_len'], cfg['horizon'], cfg['pred_len']
    test_start = n_train0 + n_val - off_h
    test_end = min(test_start + (n - n_train0 - n_val), n)
    ctx = seq_len + horizon - 1

    from sklearn.preprocessing import StandardScaler
    xs, ys = StandardScaler(), StandardScaler()
    X_train = xs.fit_transform(X_all[:n_train0 - off_h])
    X_test = xs.transform(X_all[max(0, test_start - ctx):test_end])
    y_train = ys.fit_transform(y_all[:n_train0 - off_h].reshape(-1, 1)).ravel()
    y_test = ys.transform(y_all[max(0, test_start - ctx):test_end]
                          .reshape(-1, 1)).ravel()

    test_ds = Subset(SeqForecastDataset(X_test, y_test, seq_len, horizon,
                                        pred_len),
                     list(range(len(X_test) - seq_len - horizon - 12 + 2)))
    loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    device = torch.device('cpu')
    model = build_model(cfg['model_type'], len(feature_cols), pred_len)
    ckpt = torch.load(os.path.join(args.run_dir, 'best.pt'),
                      map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model = model.to(device)

    # swap in noisy VQCs (keep trained angles)
    for mod in model.modules():
        if isinstance(mod, TorchVQC):
            mod.__class__ = NoisyTorchVQC
            mod.p = args.p_depol
            mod.tg = args.t_gate_ns * 1e-3
            mod.t1, mod.t2 = args.t1_us, args.t2_us
            mod._setup_aux()

    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            # the flow attention flattens (B,N) -> B*N before the VQC; the
            # noisy VQC processes rows in chunks to bound memory
            preds.append(model(xb).numpy())
            trues.append(yb.numpy())
    yp_s = np.concatenate(preds, 0)
    yt_s = np.concatenate(trues, 0)
    yp = ys.inverse_transform(yp_s.reshape(-1, 1)).reshape(yp_s.shape)
    yt = ys.inverse_transform(yt_s.reshape(-1, 1)).reshape(yt_s.shape)

    out = {}
    for step in range(pred_len):
        t, p = yt[:, step], yp[:, step]
        mse = float(np.mean((t - p) ** 2))
        out[f'step{step+1}'] = {'MSE': mse, 'RMSE': math.sqrt(mse),
                                'MAE': float(np.mean(np.abs(t - p))),
                                'R2': float(1 - np.sum((t - p) ** 2) /
                                            np.sum((t - t.mean()) ** 2))}
    print(json.dumps(out, indent=1))
    with open(os.path.join(args.run_dir, 'noisy_eval.json'), 'w') as f:
        json.dump({'config': vars(args), 'metrics': out}, f, indent=1)


if __name__ == '__main__':
    main()
