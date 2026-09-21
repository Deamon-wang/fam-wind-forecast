# -*- coding: utf-8 -*-
"""
revision/run_revision.py — single-run driver for the EGY-D-26-19497 revision
experiments. Mirrors wind_mamba_private_QFAM_winf.py's data pipeline, split,
training protocol, and output formats exactly (audited configuration: Adam
lr 1e-3 wd 1e-4, ReduceLROnPlateau(0.5, patience 5), max 100 epochs, early stop
10, chronological 70/15/15, batch 128 for the QFAM family / 64 for baselines,
grad clip 1.0, MSE loss), with three additions:

  * models QFAM_C (classical GELU bottleneck) and QFAM_R (frozen random VQC)
  * --offset_days: slide the 1228-hour test window backward for rolling-origin
    evaluation (0 = the archived reference split)
  * parameter-count assertions pinning the audited architecture totals

Usage:
  python3 revision/run_revision.py --model QFAM --seed 42 --pred_len 1 \
      [--offset_days 0] [--epochs 100]
"""
import os
import sys
import json
import math
import random
import argparse
import logging
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data_processing import (add_time_features, add_power_features,
                                 encode_wd, SeqForecastDataset)
from src.data_processing_winf import preprocess_winf, build_feature_cols_winf
from src.baseline_models import (WindPowerMLP, WindPowerLSTM,
                                 WindPowerSimpleTransformer)
from revision.models_rev import QFAMRev, FAMRev, count_parameters


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_mape(y_true, y_pred, eps=1e-3):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_pred - y_true) / denom))


def smape(y_true, y_pred, eps=1e-3):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


# expected total parameter counts (audited; count_parameters counts only
# trainable parameters, so QFAM_R's 2x12 frozen angles are EXCLUDED).
# QFAM: 652,573 (P1). QFAM_C: VQC angles absent in both HQPs (-24).
# QFAM_R: angles present but frozen (-24 from the trainable count).
# pred_len=3 adds 2*(3-1)+128*(3-1)=130 (QFAM family head) / 2 (FAM head).
EXPECT_PARAMS = {
    ('QFAM', 1): 652573, ('QFAM', 3): 652703,
    ('QFAM_C', 1): 652549, ('QFAM_C', 3): 652679,
    ('QFAM_R', 1): 652549, ('QFAM_R', 3): 652679,
    # FAM P3 head: final Linear(128->3) vs (128->1) adds 2*128+2 = 258
    ('FAM', 1): 764549, ('FAM', 3): 764807,
}


def build_model(model_type, input_dim, pred_len, d_model=128, seed_note=None):
    if model_type == 'QFAM':
        return QFAMRev(input_dim, model_dim=d_model, pred_horizon=pred_len,
                       variant='vqc')
    if model_type == 'QFAM_C':
        return QFAMRev(input_dim, model_dim=d_model, pred_horizon=pred_len,
                       variant='gelu')
    if model_type == 'QFAM_R':
        return QFAMRev(input_dim, model_dim=d_model, pred_horizon=pred_len,
                       variant='random')
    if model_type == 'FAM':
        return FAMRev(input_dim, model_dim=d_model, pred_horizon=pred_len)
    if model_type == 'mlp':
        return WindPowerMLP(input_dim=input_dim, d_model=512, pred_len=pred_len,
                            num_layers=4, dropout=0.3)
    if model_type == 'lstm':
        return WindPowerLSTM(input_dim=input_dim, d_model=d_model,
                             pred_len=pred_len, num_layers=1, dropout=0.1)
    if model_type == 'transformer':
        return WindPowerSimpleTransformer(input_dim=input_dim, d_model=d_model,
                                          pred_len=pred_len, num_layers=1,
                                          dropout=0.1, n_heads=4)
    raise ValueError(model_type)


def evaluate_scaled(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            preds.append(out.cpu().numpy())
            trues.append(yb.numpy())
    return (None,
            np.concatenate(preds, 0),
            np.concatenate(trues, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                    choices=['QFAM', 'QFAM_C', 'QFAM_R', 'FAM', 'mlp', 'lstm',
                             'transformer'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--pred_len', type=int, default=1, choices=[1, 3])
    ap.add_argument('--horizon', type=int, default=1)
    ap.add_argument('--seq_len', type=int, default=96)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--offset_days', type=int, default=0,
                    help='slide the test window backward by this many days '
                         '(rolling-origin evaluation; 0 = reference split)')
    ap.add_argument('--csv_path', default=os.path.join(REPO_ROOT, 'dataset',
                                                       'winf_fam1.csv'))
    ap.add_argument('--out_root', default=os.path.join(REPO_ROOT,
                                                       'results_revision'))
    ap.add_argument('--exp_id', default=None)
    args = ap.parse_args()

    seed_everything(args.seed)
    freq = '1hour'
    horizon, seq_len, pred_len = args.horizon, args.seq_len, args.pred_len
    batch_size = 128 if args.model in ('QFAM', 'QFAM_C', 'QFAM_R') else 64
    lr = 1e-3

    exp_id = args.exp_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    model_tag = f'{args.model}_s{args.seed}_P{pred_len}_o{args.offset_days}_{exp_id}'
    out_dir = os.path.join(args.out_root, model_tag)
    os.makedirs(out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(os.path.join(out_dir, 'run.log'),
                                       encoding='utf-8'),
                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(f'REVISION RUN model={args.model} seed={args.seed} '
                f'pred_len={pred_len} offset_days={args.offset_days}')
    logger.info(f'batch_size={batch_size}, lr={lr}, epochs={args.epochs}, '
                f'patience={args.patience}')

    # ---------------- data pipeline (audited copy) ----------------
    df = preprocess_winf(args.csv_path, freq=freq, use_obs_lag=True)
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
    off_h = int(round(args.offset_days * 24))
    test_start = n_train0 + n_val - off_h          # sliding origin
    test_end = min(test_start + (n - n_train0 - n_val), n)
    val_start = max(0, test_start - n_val)
    train_end = val_start
    logger.info(f'n={n}, train=[0,{train_end}), val=[{val_start},{test_start}), '
                f'test=[{test_start},{test_end}) '
                f'({date_all.iloc[test_start]} .. {date_all.iloc[test_end-1]})')

    ctx = seq_len + horizon - 1
    X_train_raw, y_train_raw = X_all[:train_end], y_all[:train_end]
    X_val_raw = X_all[max(0, val_start - ctx):test_start]
    y_val_raw = y_all[max(0, val_start - ctx):test_start]
    date_val_raw = date_all[max(0, val_start - ctx):test_start].reset_index(drop=True)
    X_test_raw = X_all[max(0, test_start - ctx):test_end]
    y_test_raw = y_all[max(0, test_start - ctx):test_end]
    date_test_raw = date_all[max(0, test_start - ctx):test_end].reset_index(drop=True)

    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train_raw)
    X_val = x_scaler.transform(X_val_raw)
    X_test = x_scaler.transform(X_test_raw)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
    y_val = y_scaler.transform(y_val_raw.reshape(-1, 1)).ravel()
    y_test = y_scaler.transform(y_test_raw.reshape(-1, 1)).ravel()

    train_ds = SeqForecastDataset(X_train, y_train, seq_len, horizon, pred_len)
    val_ds = SeqForecastDataset(X_val, y_val, seq_len, horizon, pred_len)
    test_ds_full = SeqForecastDataset(X_test, y_test, seq_len, horizon, pred_len)
    max_eval_pred_len = 12
    common_n_test = len(X_test_raw) - seq_len - horizon - max_eval_pred_len + 2
    test_ds = Subset(test_ds_full, list(range(common_n_test)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             drop_last=False)
    logger.info(f'samples: train={len(train_ds)} val={len(val_ds)} '
                f'test={len(test_ds)}')

    # ---------------- model ----------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.model, len(feature_cols), pred_len).to(device)
    total = count_parameters(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_all = sum(p.numel() for p in model.parameters())
    logger.info(f'params: trainable={trainable:,} all={total_all:,}')
    if (args.model, pred_len) in EXPECT_PARAMS:
        exp = EXPECT_PARAMS[(args.model, pred_len)]
        assert total == exp, f'{args.model} P{pred_len}: expected {exp}, got {total}'

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best_val, best_state, best_epoch, bad = float('inf'), None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tot += loss.item() * xb.size(0)
            nb += xb.size(0)
        train_loss = tot / max(nb, 1)

        model.eval()
        pv, pt_ = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                pv.append(model(xb.to(device)).cpu().numpy())
                pt_.append(yb.numpy())
        vp = np.concatenate(pv, 0)
        vt = np.concatenate(pt_, 0)
        val_mse = float(np.mean((vp - vt) ** 2))
        scheduler.step(val_mse)

        if val_mse < best_val - 1e-6:
            best_val, best_epoch, bad = val_mse, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
        logger.info(f'[Epoch {epoch:03d}] train_loss={train_loss:.6f} '
                    f'val_MSE={val_mse:.6f} bad={bad}/{args.patience}')
        if bad >= args.patience:
            logger.info(f'early stop at epoch {epoch}')
            break

    model.load_state_dict(best_state)
    ckpt_path = os.path.join(out_dir, 'best.pt')
    torch.save({'model_state_dict': best_state, 'args': vars(args)}, ckpt_path)

    # ---------------- test evaluation ----------------
    _, y_pred_scaled, y_true_scaled = evaluate_scaled(model, test_loader, device)
    y_pred = y_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)
    y_true = y_scaler.inverse_transform(
        y_true_scaled.reshape(-1, 1)).reshape(y_true_scaled.shape)

    metrics = {}
    for step in range(pred_len):
        t, p = y_true[:, step], y_pred[:, step]
        metrics[f'step{step+1}'] = {
            'MSE': float(mean_squared_error(t, p)),
            'RMSE': float(math.sqrt(mean_squared_error(t, p))),
            'MAE': float(mean_absolute_error(t, p)),
            'R2': float(r2_score(t, p)),
        }
    metrics['overall'] = {
        'MSE': float(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1))),
        'RMSE': float(math.sqrt(mean_squared_error(y_true.reshape(-1),
                                                   y_pred.reshape(-1)))),
        'MAE': float(mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))),
        'R2': float(r2_score(y_true.reshape(-1), y_pred.reshape(-1))),
        'MAPE': safe_mape(y_true.reshape(-1), y_pred.reshape(-1)),
        'SMAPE': smape(y_true.reshape(-1), y_pred.reshape(-1)),
    }
    logger.info('TEST METRICS (original scale): ' + json.dumps(metrics, indent=1))

    # test_pred CSV (same long format as archived runs)
    rows = []
    n_test_samples = y_true.shape[0]
    for i in range(n_test_samples):
        origin_idx = i + seq_len - 1
        for s in range(pred_len):
            t_idx = i + seq_len + horizon - 1 + s
            rows.append({
                'forecast_origin_id': i,
                'forecast_origin_date': date_test_raw.iloc[origin_idx],
                'target_date': date_test_raw.iloc[t_idx],
                'pred_step': s + 1,
                'horizon_step': horizon + s,
                'power_true': float(y_true[i, s]),
                'power_pred': float(y_pred[i, s]),
            })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'test_pred.csv'),
                              index=False)

    config = {
        'model_type': args.model, 'seed': args.seed, 'pred_len': pred_len,
        'horizon': horizon, 'seq_len': seq_len, 'offset_days': args.offset_days,
        'batch_size': batch_size, 'lr': lr, 'epochs': args.epochs,
        'patience': args.patience, 'optimizer': 'adam', 'wd': 1e-4,
        'scheduler': 'plateau(0.5,5)', 'loss': 'MSE', 'clip': 1.0,
        'feature_cols': feature_cols,
        'test_window': [str(date_all.iloc[test_start]),
                        str(date_all.iloc[test_end - 1])],
        'n_test_samples': int(n_test_samples),
        'params_total': int(total_all), 'params_trainable': int(trainable),
        'best_epoch': best_epoch, 'best_val_mse': best_val,
        'metrics': metrics,
    }
    with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=1, ensure_ascii=False)
    with open(os.path.join(out_dir, 'DONE'), 'w') as f:
        f.write('ok\n')
    logger.info('RUN COMPLETE')


if __name__ == '__main__':
    main()
