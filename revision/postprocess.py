# -*- coding: utf-8 -*-
"""
revision/postprocess.py — aggregates results_revision runs into the numbers the
revision needs:

  * Table: per model, 1 h (P1) and 3 h (P3, horizon_step==3) metrics,
    mean +- std over seeds
  * persistence + climatology reference rows (per test window)
  * skill vs persistence, nRMSE (capacity configurable, default 50 MW)
  * Diebold-Mariano tests with HAC variance (lag 24), QFAM vs each control,
    per seed
  * rolling-origin table

Outputs: results_revision/summary.json and summary.md
"""
import os
import sys
import json
import math
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
OUT_ROOT = os.path.join(REPO, 'results_revision')
CAPACITY_MW = float(os.environ.get('QFAM_CAPACITY_MW', 50.0))
DM_LAG = 24

MODEL_LABEL = {'QFAM': 'QFAM', 'QFAM_C': 'QFAM-C', 'QFAM_R': 'QFAM-R',
               'FAM': 'FAM', 'mlp': 'MLP', 'lstm': 'LSTM',
               'transformer': 'Transformer'}


def load_runs():
    runs = []
    if not os.path.isdir(OUT_ROOT):
        raise SystemExit('no results_revision directory')
    # dedup: keep only the LATEST DONE run per (model, seed, pred_len, offset)
    best = {}
    for d in sorted(os.listdir(OUT_ROOT)):
        cfg_p = os.path.join(OUT_ROOT, d, 'config.json')
        pred_p = os.path.join(OUT_ROOT, d, 'test_pred.csv')
        done_p = os.path.join(OUT_ROOT, d, 'DONE')
        if not (os.path.exists(cfg_p) and os.path.exists(pred_p)
                and os.path.exists(done_p)):
            continue
        with open(cfg_p, encoding='utf-8') as f:
            cfg = json.load(f)
        key = (cfg['model_type'], cfg['seed'], cfg['pred_len'],
               cfg['offset_days'])
        best[key] = {'dir': d, 'cfg': cfg, 'pred_path': pred_p}
    return list(best.values())


def met(tp, fp):
    tp, fp = np.asarray(tp, float), np.asarray(fp, float)
    mse = float(np.mean((tp - fp) ** 2))
    return {'MSE': mse, 'RMSE': math.sqrt(mse),
            'MAE': float(np.mean(np.abs(tp - fp))),
            'R2': float(1 - np.sum((tp - fp) ** 2) /
                        np.sum((tp - tp.mean()) ** 2))}


def dm_test(e1, e2, lag=DM_LAG):
    """Diebold-Mariano with HAC (Newey-West) variance, two-sided."""
    d = np.asarray(e1, float) - np.asarray(e2, float)
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    gamma0 = np.mean(dc ** 2)
    var = gamma0
    for l in range(1, min(lag, n - 1) + 1):
        w = 1.0 - l / (lag + 1.0)
        cov = np.mean(dc[l:] * dc[:-l])
        var += 2.0 * w * cov
    var = max(var, 1e-18) / n
    stat = dbar / math.sqrt(var)
    from scipy import stats
    p = 2 * (1 - stats.norm.cdf(abs(stat)))
    return float(stat), float(p)


def climatology_forecast(dataset_csv, train_end_idx, target_dates):
    """Hour-of-day mean power over the training period -> forecast."""
    from src.data_processing_winf import preprocess_winf
    df = preprocess_winf(dataset_csv, freq='1hour', use_obs_lag=True)
    df = df.sort_values('date').reset_index(drop=True)
    tr = df.iloc[:train_end_idx]
    hod = tr.groupby(tr['date'].dt.hour)['power'].mean()
    gm = tr['power'].mean()
    f = []
    for td in pd.to_datetime(target_dates):
        f.append(float(hod.get(td.hour, gm)))
    return np.array(f)


def main():
    runs = load_runs()
    print(f'found {len(runs)} runs')

    # ---------- per-run 1h/3h metrics from test_pred ----------
    # Each run is evaluated ONLY at its own target horizon: pred_len=1
    # runs at horizon_step 1, pred_len=3 runs at horizon_step 3 (the
    # 3-h-ahead step). Step-1 rows of pred_len-3 models are deliberately
    # NOT mixed into the 1-h table.
    per_run = []
    for r in runs:
        cfg = r['cfg']
        pl = cfg['pred_len']
        hs = pl
        df = pd.read_csv(r['pred_path'], parse_dates=['target_date',
                                                      'forecast_origin_date'])
        sub = df[df['horizon_step'] == hs]
        if len(sub) == 0:
            continue
        m = met(sub['power_true'], sub['power_pred'])
        per_run.append({
            'model': cfg['model_type'], 'seed': cfg['seed'],
            'pred_len': pl, 'offset': cfg['offset_days'],
            'window': cfg['test_window'], 'metrics': m,
            'sub': sub, 'dir': r['dir'],
        })
    # persistence/climatology computed per (offset, pred_len) once
    refs = {}
    for pr in per_run:
        key = (pr['offset'], pr['pred_len'])
        if key in refs:
            continue
        sub = pr['sub']
        # ground-truth series indexed by target_date
        g = sub.groupby('target_date')['power_true'].mean().sort_index()
        dates = g.index
        y = g.values
        hs = pr['pred_len']
        # persistence: use the last observed value at issue time
        # (target_date - hs hours)
        lag_dates = dates - pd.Timedelta(hours=hs)
        series = pd.Series(y, index=dates)
        aligned = series.reindex(lag_dates)
        pers_pred = aligned.values  # y at issue time == persistence forecast
        mask = ~np.isnan(pers_pred)
        pers_m = met(y[mask], pers_pred[mask])
        # climatology: train-period hourly means
        # train_end for offset 0 = int(n*0.7); recompute generally:
        from src.data_processing_winf import preprocess_winf
        dfall = preprocess_winf(os.path.join(REPO, 'dataset', 'winf_fam1.csv'),
                                freq='1hour', use_obs_lag=True)
        dfall = dfall.sort_values('date').reset_index(drop=True)
        n = len(dfall)
        n_train0 = int(n * 0.7)
        # training set = everything before (test_start - val); test_start is
        # slid backward by offset days, so train shrinks accordingly
        train_end = max(0, n_train0 - pr['offset'] * 24)
        clim = climatology_forecast(os.path.join(REPO, 'dataset',
                                                 'winf_fam1.csv'),
                                    train_end, dates[mask])
        clim_m = met(y[mask], clim)
        refs[key] = {'persistence': pers_m, 'climatology': clim_m,
                     'n': int(mask.sum())}

    # ---------- aggregate mean +- std ----------
    summary = {'capacity_MW': CAPACITY_MW, 'dm_lag': DM_LAG,
               'tables': {}, 'dm': {}, 'refs': {}}
    lines = ['# Revision experiment summary\n']
    for pl in (1, 3):
        lines.append(f'\n## Lead {pl} h  (P{pl}, horizon_step={pl})\n')
        lines.append('| model | MSE | RMSE | MAE | R2 | seeds |')
        lines.append('|---|---|---|---|---|---|')
        tab = {}
        for m in ['QFAM', 'QFAM_C', 'QFAM_R', 'FAM', 'mlp', 'lstm',
                  'transformer']:
            ms = [pr['metrics'] for pr in per_run
                  if pr['model'] == m and pr['pred_len'] == pl
                  and pr['offset'] == 0]
            if not ms:
                continue
            agg = {k: (float(np.mean([x[k] for x in ms])),
                       float(np.std([x[k] for x in ms], ddof=1))
                       if len(ms) > 1 else 0.0)
                   for k in ('MSE', 'RMSE', 'MAE', 'R2')}
            tab[m] = {'mean': {k: v[0] for k, v in agg.items()},
                      'std': {k: v[1] for k, v in agg.items()},
                      'n_seeds': len(ms),
                      'per_seed': ms}
            lines.append(
                f"| {MODEL_LABEL.get(m, m)} | "
                f"{agg['MSE'][0]:.2f}±{agg['MSE'][1]:.2f} | "
                f"{agg['RMSE'][0]:.3f}±{agg['RMSE'][1]:.3f} | "
                f"{agg['MAE'][0]:.3f}±{agg['MAE'][1]:.3f} | "
                f"{agg['R2'][0]:.4f}±{agg['R2'][1]:.4f} | {len(ms)} |")
        ref = refs.get((0, pl))
        if ref:
            for name in ('persistence', 'climatology'):
                r = ref[name]
                lines.append(f"| {name} | {r['MSE']:.2f} | {r['RMSE']:.3f} | "
                             f"{r['MAE']:.3f} | {r['R2']:.4f} | - |")
            tab['persistence'] = ref['persistence']
            tab['climatology'] = ref['climatology']
            tab['n_test'] = ref['n']
        # skill vs persistence + nRMSE
        if ref and 'QFAM' in tab:
            q = tab['QFAM']['mean']
            p = ref['persistence']
            tab['skill'] = {k: 1 - q[k] / p[k] for k in ('RMSE', 'MAE')}
            tab['nRMSE_pct'] = 100 * q['RMSE'] / CAPACITY_MW
            lines.append(f"\nskill(QFAM vs persistence): "
                         f"RMSE {tab['skill']['RMSE']*100:.1f}%, "
                         f"MAE {tab['skill']['MAE']*100:.1f}%")
            lines.append(f"nRMSE(QFAM, capacity {CAPACITY_MW} MW): "
                         f"{tab['nRMSE_pct']:.2f}%")
        summary['tables'][f'P{pl}'] = tab
        summary['refs'][f'P{pl}'] = refs.get((0, pl))

    # ---------- DM tests ----------
    lines.append('\n## Diebold-Mariano (HAC lag 24), squared errors, '
                 'QFAM vs control\n')
    lines.append('| lead | control | seed | DM stat | p |\n|---|---|---|---|---|')
    for pl in (1, 3):
        qruns = {pr['seed']: pr for pr in per_run
                 if pr['model'] == 'QFAM' and pr['pred_len'] == pl
                 and pr['offset'] == 0}
        for ctrl in ('FAM', 'QFAM_C', 'QFAM_R', 'mlp', 'lstm', 'transformer'):
            cruns = {pr['seed']: pr for pr in per_run
                     if pr['model'] == ctrl and pr['pred_len'] == pl
                     and pr['offset'] == 0}
            stats_list = []
            for s in sorted(set(qruns) & set(cruns)):
                a = qruns[s]['sub'].sort_values('target_date')
                b = cruns[s]['sub'].sort_values('target_date')
                assert (a['target_date'].values == b['target_date'].values).all()
                e1 = (a['power_true'] - a['power_pred']).values ** 2
                e2 = (b['power_true'] - b['power_pred']).values ** 2
                st, pv = dm_test(e1, e2)
                stats_list.append((s, st, pv))
                lines.append(f'| {pl}h | {MODEL_LABEL.get(ctrl, ctrl)} | {s} | '
                             f'{st:+.3f} | {pv:.4f} |')
            if stats_list:
                summary['dm'][f'P{pl}_{ctrl}'] = [
                    {'seed': s, 'stat': st, 'p': pv} for s, st, pv in stats_list]

    # ---------- rolling origins ----------
    lines.append('\n## Rolling origins (test window slid backward)\n')
    lines.append('| model | lead | offset d | RMSE | test window |\n'
                 '|---|---|---|---|---|')
    roll = {}
    for pr in per_run:
        if pr['offset'] == 0 or pr['model'] not in ('QFAM', 'QFAM_C', 'FAM'):
            continue
        key = (pr['model'], pr['pred_len'], pr['offset'])
        roll[key] = {'RMSE': pr['metrics']['RMSE'],
                     'MSE': pr['metrics']['MSE'],
                     'window': pr['window']}
        lines.append(f"| {MODEL_LABEL[pr['model']]} | {pr['pred_len']}h | "
                     f"-{pr['offset']} | {pr['metrics']['RMSE']:.3f} | "
                     f"{pr['window'][0][:10]}..{pr['window'][1][:10]} |")
    summary['rolling'] = {f'{k[0]}_P{k[1]}_o{k[2]}': v for k, v in roll.items()}

    with open(os.path.join(OUT_ROOT, 'summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump(summary, f, indent=1, ensure_ascii=False, default=str)
    with open(os.path.join(OUT_ROOT, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('wrote summary.json / summary.md')


if __name__ == '__main__':
    main()
