# -*- coding: utf-8 -*-
"""plot_figs.py — regenerate Figs. 5 and 6 of the revised manuscript
from the regenerated revision runs (seed 42), replacing the archived
figures so that figures and tables come from the same runs.

Fig. 5 (a) 1 h: two 96-h segments selected by a stated criterion:
    the 96-h window with (i) the highest mean output (high-output
    regime) and (ii) the steepest sustained ramp-down
    (max 24-h decrease within the window).
Fig. 5 (b) 3 h: the 96-h window with the largest cumulative
    ramp-lull variation (std of 24-h changes).
Fig. 6: (a,b) per-step error distributions QFAM vs FAM at both
    horizons; (c,d) |error| box statistics for all learned models.

Usage: python plot_figs.py <results_root> <outdir>
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_ROOT, FIGDIR = sys.argv[1], sys.argv[2]
MODELS = ['mlp', 'lstm', 'transformer', 'FAM', 'QFAM']
LABEL = {'QFAM': 'QFAM', 'FAM': 'FAM', 'mlp': 'MLP', 'lstm': 'LSTM',
         'transformer': 'Transformer'}
COLOR = {'QFAM': '#d62728', 'FAM': '#1f77b4',
         'transformer': '#2ca02c', 'mlp': '#9467bd', 'lstm': '#8c564b'}


def latest_done_dir(model, seed, pl):
    cands = []
    for d in glob.glob(os.path.join(
            OUT_ROOT, f'{model}_s{seed}_P{pl}_o0_*')):
        if os.path.exists(os.path.join(d, 'DONE')):
            cfg = json.load(open(os.path.join(d, 'config.json')))
            if (cfg['model_type'] == model and cfg['seed'] == seed
                    and cfg['pred_len'] == pl and cfg['offset_days'] == 0):
                cands.append((os.path.getmtime(os.path.join(d, 'DONE')), d))
    return max(cands)[1] if cands else None


def load(model, pl, hs):
    d = latest_done_dir(model, 42, pl)
    df = pd.read_csv(os.path.join(d, 'test_pred.csv'),
                     parse_dates=['target_date'])
    return df[df['horizon_step'] == hs].sort_values('target_date')


def pick_window(y, hours=96, mode='high'):
    """Stated criterion (R2-9): (a) 'high' -> window with the highest
    mean true power; 'ramp' -> window containing the steepest 24-h
    decrease; 'var' -> window with the largest std of 24-h changes."""
    best, best_v = 0, -1e18
    for i in range(0, len(y) - hours):
        w = y[i:i + hours]
        if mode == 'high':
            v = w.mean()
        elif mode == 'ramp':
            d24 = w[24:] - w[:-24]
            v = -d24.min()          # steepest sustained decrease
        else:
            d24 = w[24:] - w[:-24]
            v = d24.std()           # ramp/lull variation
        if v > best_v:
            best_v, best = v, i
    return best


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    # ---------------- Fig. 5 ----------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for row, (pl, mode) in enumerate(((1, 'ramp'), (3, 'var'))):
        q = load('QFAM', pl, pl)
        y = q['power_true'].values
        i0 = pick_window(y, 96,
                         'high' if pl == 1 else 'var')
        # 1 h panel: high-output window that also contains the steep
        # ramp-down: take high-output window then extend to cover the
        # following drop if adjacent
        if pl == 1:
            # window with highest mean, then the window with steepest
            # 24-h decrease; use both if disjoint -> pick ramp window
            i_r = pick_window(y, 96, 'ramp')
            i0 = i_r if abs(i_r - i0) > 96 else i0
        sl = slice(i0, i0 + 96)
        ax = axes[row]
        t = np.arange(96)
        ax.plot(t, y[sl], 'k-', lw=1.6, label='Actual')
        for m in ('FAM', 'transformer', 'QFAM'):
            sub = load(m, pl, pl)
            yy = sub['power_pred'].values[sl]
            ax.plot(t, yy, color=COLOR[m], lw=1.1, alpha=0.9,
                    label=LABEL[m])
        ax.set_title(f'({chr(97 + row)}) {pl} h horizon '
                     f'({q["target_date"].iloc[i0]:%d %b}–'
                     f'{q["target_date"].iloc[min(i0 + 95, len(q)-1)]:%d %b})')
        ax.set_ylabel('Power (MW)')
        ax.set_xlim(0, 95)
        if row == 1:
            ax.set_xlabel('hours')
        ax.legend(ncol=4, fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'f05_new.png'), dpi=300)
    plt.close(fig)

    # ---------------- Fig. 6 ----------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    for col, pl in enumerate((1, 3)):
        q = load('QFAM', pl, pl)
        f = load('FAM', pl, pl)
        eq = (q['power_pred'] - q['power_true']).values
        ef = (f['power_pred'] - f['power_true']).values
        ax = axes[0, col]
        bins = np.linspace(-25, 25, 61)
        ax.hist(eq, bins=bins, alpha=0.6, density=True,
                color=COLOR['QFAM'], label='QFAM')
        ax.hist(ef, bins=bins, alpha=0.6, density=True,
                color=COLOR['FAM'], label='FAM')
        ax.set_title(f'({"ab"[col]}) error distribution, {pl} h')
        ax.set_xlabel('forecast − actual (MW)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax = axes[1, col]
        data, labs = [], []
        for m in MODELS:
            sub = load(m, pl, pl)
            data.append(np.abs(sub['power_pred']
                               - sub['power_true']).values)
            labs.append(LABEL[m])
        bp = ax.boxplot(data, tick_labels=labs, showfliers=False,
                        patch_artist=True)
        for patch, m in zip(bp['boxes'], MODELS):
            patch.set_facecolor(COLOR[m])
            patch.set_alpha(0.6)
        ax.set_title(f'({"cd"[col]}) |error| by model, {pl} h')
        ax.set_ylabel('|error| (MW)')
        ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'f06_new.png'), dpi=300)
    plt.close(fig)
    print('wrote f05_new.png f06_new.png')


if __name__ == '__main__':
    main()
