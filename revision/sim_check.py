# -*- coding: utf-8 -*-
"""sim_check.py — integrity check: compare the 12-origin sim reference
(from hw_capture) against the training run's saved test predictions
(test_pred.csv), and print per-origin values.

Usage: python sim_check.py <run_dir> <capture.npz>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

RD, CAP = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(RD, 'config.json')))
pred_len = cfg['pred_len']

df = pd.read_csv(os.path.join(RD, 'test_pred.csv'))
print('csv columns:', list(df.columns)[:8], 'rows:', len(df))
cap = np.load(CAP)
idxs = cap['idxs']
y_sim_cap = cap['y_sim_mw']
y_true_cap = cap['y_true_mw']

# locate prediction and truth columns
pc = [c for c in df.columns if 'pred' in c.lower() or 'yhat' in c.lower()]
tc = [c for c in df.columns if 'true' in c.lower() or 'y_' == c[:2]]
print('pred cols:', pc, 'true cols:', tc)

yhat = df['power_pred'].to_numpy()
ytru = df['power_true'].to_numpy()

if len(yhat) == 1206:
    y_sim_csv = yhat[idxs]
    y_true_csv = ytru[idxs]
    capsim = y_sim_cap.ravel()
    captrue = y_true_cap.ravel()
    print('\norigin | capture_sim |   csv_sim   | capture_true |  csv_true')
    for i, oi in enumerate(idxs):
        print(f'{oi:6d} | {capsim[i]:10.4f} | {y_sim_csv[i]:10.4f} | '
              f'{captrue[i]:12.4f} | {y_true_csv[i]:9.4f}')
    d = capsim - y_sim_csv
    print('\nmax |diff| sim capture vs csv:', np.abs(d).max())
    rmse_cap = np.sqrt(np.mean((captrue - capsim) ** 2))
    rmse_csv = np.sqrt(np.mean((captrue - y_sim_csv) ** 2))
    print('12-origin sim RMSE: capture=%.4f csv=%.4f' % (rmse_cap,
                                                         rmse_csv))
else:
    print('row count mismatch:', len(yhat))
