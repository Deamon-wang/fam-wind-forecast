# -*- coding: utf-8 -*-
"""Random-angle sensitivity: evaluate a trained QFAM checkpoint with the VQC
angles replaced by fresh uniform draws (inference-time ablation; pure
simulation, no training). Quantifies how much the *trained* quantum
parameters contribute relative to random ones."""
import os
import sys
import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.data_processing import (add_time_features, add_power_features,
                                 encode_wd, SeqForecastDataset)
from src.data_processing_winf import preprocess_winf, build_feature_cols_winf
from revision.models_rev import TorchVQC
from revision.run_revision import build_model, seed_everything
from sklearn.preprocessing import StandardScaler


def main():
    run_dir = sys.argv[1]
    n_draws = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    with open(os.path.join(run_dir, 'config.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    seed_everything(cfg['seed'])

    df = preprocess_winf(os.path.join(REPO, 'dataset', 'winf_fam1.csv'),
                         freq='1hour', use_obs_lag=True)
    df = add_time_features(df)
    df = encode_wd(df)
    df = add_power_features(df)
    fcols = build_feature_cols_winf(df, use_obs_lag=True)
    df = df.dropna(subset=['date', 'power'] + fcols) \
        .sort_values('date').reset_index(drop=True)
    X_all = df[fcols].to_numpy(np.float32)
    y_all = df['power'].to_numpy(np.float32)
    n = len(df)
    n_train0, n_val = int(n * 0.7), int(n * 0.15)
    seq, hor, pl = cfg['seq_len'], cfg['horizon'], cfg['pred_len']
    ts = n_train0 + n_val - cfg['offset_days'] * 24
    te = min(ts + (n - n_train0 - n_val), n)
    ctx = seq + hor - 1
    xs = StandardScaler().fit(X_all[:ts])
    X_test = xs.transform(X_all[max(0, ts - ctx):te])
    ys = StandardScaler().fit(y_all[:ts].reshape(-1, 1))
    y_test = ys.transform(y_all[max(0, ts - ctx):te].reshape(-1, 1)).ravel()
    ds = Subset(SeqForecastDataset(X_test, y_test, seq, hor, pl),
                list(range(len(X_test) - seq - hor - 12 + 2)))
    loader = DataLoader(ds, batch_size=128, shuffle=False)

    model = build_model(cfg['model_type'], len(fcols), pl)
    ckpt = torch.load(os.path.join(run_dir, 'best.pt'), map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    def evaluate():
        ps, ts_ = [], []
        with torch.no_grad():
            for xb, yb in loader:
                ps.append(model(xb).numpy())
                ts_.append(yb.numpy())
        yp = np.concatenate(ps, 0)
        yt = np.concatenate(ts_, 0)
        ypi = ys.inverse_transform(yp.reshape(-1, 1)).reshape(yp.shape)
        yti = ys.inverse_transform(yt.reshape(-1, 1)).reshape(yt.shape)
        mse = float(np.mean((yti - ypi) ** 2))
        return {'MSE': mse, 'RMSE': math.sqrt(mse),
                'MAE': float(np.mean(np.abs(yti - ypi))),
                'R2': float(1 - np.sum((yti - ypi) ** 2) /
                            np.sum((yti - yti.mean()) ** 2))}

    vqcs = [m for m in model.modules() if isinstance(m, TorchVQC)]
    trained = [v.paras.data.clone() for v in vqcs]
    base = evaluate()
    rand = []
    for k in range(n_draws):
        for v, tr in zip(vqcs, trained):
            with torch.no_grad():
                v.paras.copy_((torch.rand_like(tr) * 2 - 1) * math.pi)
        rand.append(evaluate())
    for v, tr in zip(vqcs, trained):
        with torch.no_grad():
            v.paras.copy_(tr)

    agg = {key: {'mean': float(np.mean([r[key] for r in rand])),
                 'std': float(np.std([r[key] for r in rand], ddof=1))}
           for key in ('MSE', 'RMSE', 'MAE', 'R2')}
    out = {'trained': base, 'random_draws': n_draws, 'random': agg,
           'per_draw': rand}
    print(json.dumps(out, indent=1))
    with open(os.path.join(run_dir, 'random_angle_eval.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
