# -*- coding: utf-8 -*-
"""residual_analysis.py — R2-(8): residual analysis across error
quantiles, generation regimes, and ramp intensity, from the regenerated
revision runs' test_pred.csv files (no new training).

Outputs residual_analysis.json + residual_analysis.md in OUT_ROOT.

Usage: python residual_analysis.py <results_root>
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

OUT_ROOT = None
MODELS = ['QFAM', 'FAM', 'QFAM_C', 'QFAM_R', 'transformer', 'mlp',
          'lstm']
LABEL = {'QFAM': 'QFAM', 'QFAM_C': 'QFAM-C', 'QFAM_R': 'QFAM-R',
         'FAM': 'FAM', 'mlp': 'MLP', 'lstm': 'LSTM',
         'transformer': 'Transformer'}


def latest_done_dir(model, seed, pl, off=0):
    cands = []
    for d in glob.glob(os.path.join(
            OUT_ROOT, f'{model}_s{seed}_P{pl}_o{off}_*')):
        if os.path.exists(os.path.join(d, 'DONE')):
            cfg = json.load(open(os.path.join(d, 'config.json')))
            if (cfg['model_type'] == model and cfg['seed'] == seed
                    and cfg['pred_len'] == pl
                    and cfg['offset_days'] == off):
                cands.append((os.path.getmtime(
                    os.path.join(d, 'DONE')), d))
    return max(cands)[1] if cands else None


def met(err):
    ae = np.abs(err)
    return {'RMSE': float(np.sqrt(np.mean(err ** 2))),
            'MAE': float(np.mean(ae)),
            'bias': float(np.mean(err)),
            'P50': float(np.percentile(ae, 50)),
            'P90': float(np.percentile(ae, 90)),
            'P95': float(np.percentile(ae, 95)),
            'P99': float(np.percentile(ae, 99))}


def main():
    global OUT_ROOT
    OUT_ROOT = sys.argv[1]
    out = {}
    lines = ['# Residual analysis (R2-8): regimes / ramps / quantiles\n']

    for pl, hs in ((1, 1), (3, 3)):
        lines.append(f'\n## Horizon {pl} h\n')
        # collect per-model series at this horizon (seed 42 reference)
        series = {}
        for m in MODELS:
            d = latest_done_dir(m, 42, pl)
            if not d:
                continue
            df = pd.read_csv(os.path.join(d, 'test_pred.csv'),
                             parse_dates=['target_date'])
            sub = df[df['horizon_step'] == hs].sort_values('target_date')
            series[m] = sub
        if 'QFAM' not in series or 'FAM' not in series:
            continue
        q = series['QFAM']
        y = q['power_true'].values
        # ---------- generation-regime bins from ground truth ----------
        qedges = np.quantile(y, [0, .2, .4, .6, .8, 1.0])
        qedges[0] -= 1e-9
        qedges[-1] += 1e-9
        regime_idx = np.digitize(y, qedges) - 1
        # ---------- ramp intensity: |y_t - y_{t-h}| from ground truth ----------
        dates = pd.Series(y, index=q['target_date'])
        ylag = dates.shift(hs).values
        ramp = np.abs(y - ylag)
        ramp[hs:] = ramp[hs:]  # first hs entries are NaN -> exclude
        valid = ~np.isnan(ramp)
        redges = np.quantile(ramp[valid], [0, .5, .8, 1.0])
        redges[0] -= 1e-9
        redges[-1] += 1e-9
        ramp_idx = np.digitize(np.where(valid, ramp, -1), redges) - 1

        out[f'P{pl}'] = {'regimes': {}, 'ramps': {}, 'quantiles': {}}
        lines.append('| model | regime RMSE (Q1..Q5 of true power) |'
                     ' ramp RMSE (low / high) | overall |\n|---|---|---|---|')
        for m, sub in series.items():
            e = (sub['power_pred'] - sub['power_true']).values
            rr = [float(np.sqrt(np.mean(e[regime_idx == k] ** 2)))
                  for k in range(5)]
            lo = e[valid & (ramp_idx <= 0)]
            hi = e[valid & (ramp_idx >= 1)]
            rm = [float(np.sqrt(np.mean(lo ** 2))),
                  float(np.sqrt(np.mean(hi ** 2)))]
            ov = met(e)
            out[f'P{pl}']['regimes'][m] = rr
            out[f'P{pl}']['ramps'][m] = rm
            out[f'P{pl}']['quantiles'][m] = ov
            lines.append(f"| {LABEL[m]} | " +
                         ' / '.join(f'{v:.2f}' for v in rr) +
                         f" | {rm[0]:.2f} / {rm[1]:.2f} "
                         f"| RMSE {ov['RMSE']:.2f}, bias {ov['bias']:+.2f}, "
                         f"P90 {ov['P90']:.2f} |")
        # QFAM-vs-FAM per-regime ratio
        if 'FAM' in series and 'QFAM' in series:
            eq = (series['QFAM']['power_pred']
                  - series['QFAM']['power_true']).values
            ef = (series['FAM']['power_pred']
                  - series['FAM']['power_true']).values
            ratio = [float(np.sqrt(np.mean(eq[regime_idx == k] ** 2))
                           / np.sqrt(np.mean(ef[regime_idx == k] ** 2)))
                     for k in range(5)]
            out[f'P{pl}']['QFAM_over_FAM_regime_ratio'] = ratio
            lines.append(f"QFAM/FAM per-regime RMSE ratio: "
                         + ' / '.join(f'{r:.3f}' for r in ratio))

    with open(os.path.join(OUT_ROOT, 'residual_analysis.json'), 'w',
              encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(OUT_ROOT, 'residual_analysis.md'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
