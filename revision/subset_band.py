# -*- coding: utf-8 -*-
"""subset_band.py — quantify 12-origin subset-selection variability of
the simulator RMSE: bootstrap random 12-origin subsets from the 1206
test predictions of the training run.

Usage: python subset_band.py <run_dir> [--n 2000]
"""
import argparse
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('run_dir')
ap.add_argument('--n', type=int, default=2000)
ap.add_argument('--step', type=int, default=0,
                help='pred_step filter (0 = keep all rows)')
a = ap.parse_args()

df = pd.read_csv(os.path.join(a.run_dir, 'test_pred.csv'))
if a.step:
    df = df[df['pred_step'] == a.step].reset_index(drop=True)
    print(f'filtered to pred_step={a.step}: {len(df)} rows')
y_true = df['power_true'].to_numpy()
y_pred = df['power_pred'].to_numpy()
e2 = (y_true - y_pred) ** 2
rng = np.random.default_rng(42)
N = len(e2)
rmses = []
for _ in range(a.n):
    idx = rng.choice(N, size=12, replace=False)
    rmses.append(np.sqrt(e2[idx].mean()))
rmses = np.array(rmses)
print(f'sim full-test RMSE: {np.sqrt(e2.mean()):.4f}')
print(f'12-origin subsets (n={a.n}): mean={rmses.mean():.4f} '
      f'std={rmses.std():.4f}')
print('percentiles 5/50/95:',
      np.round(np.percentile(rmses, [5, 50, 95]), 4))
