# -*- coding: utf-8 -*-
"""make_tex_rows.py — turn results_revision artifacts into ready-to-paste
LaTeX rows for Energy_QFAM_Overleaf.tex.

Inputs (on the server, produced by postprocess.py / noisy_eval.py /
rand_angle_eval.py):
  results_revision/summary.json
  results_revision/QFAM_s42_P1_o0_*/noisy_eval.json      (1h noise row)
  results_revision/QFAM_s42_P3_o0_*/noisy_eval.json      (3h noise row)
  results_revision/QFAM_s42_P1_o0_*/random_angle_eval.json
  results_revision/QFAM_s42_P3_o0_*/random_angle_eval.json

Usage:  python make_tex_rows.py <results_root> [--capacity 50]
"""
import glob
import json
import os
import sys

import numpy as np

CAP = 50.0
LABEL = {'QFAM': 'QFAM', 'QFAM_C': 'QFAM-C', 'QFAM_R': 'QFAM-R',
         'FAM': 'FAM', 'mlp': 'MLP', 'lstm': 'LSTM',
         'transformer': 'Transformer'}


def fmt_ms(v, s):
    return f"{v:.3f}$\\pm${s:.3f}"


def fmt_r2(v, s):
    return f"{v:.4f}$\\pm${s:.4f}"


def main():
    global CAP
    root = sys.argv[1]
    if '--capacity' in sys.argv:
        CAP = float(sys.argv[sys.argv.index('--capacity') + 1])
    with open(os.path.join(root, 'summary.json'), encoding='utf-8') as f:
        S = json.load(f)

    print('=' * 72)
    print('TABLE tab:model_comparison rows  (mean±std over seeds; offset 0)')
    print('=' * 72)
    for pl in ('P1', 'P3'):
        print(f'--- {pl} ---')
        tab = S['tables'][pl]
        for m in ('mlp', 'lstm', 'transformer', 'FAM', 'QFAM', 'QFAM_C',
                  'QFAM_R'):
            if m not in tab:
                continue
            mu, sd = tab[m]['mean'], tab[m]['std']
            n = tab[m]['n_seeds']
            rmse = fmt_ms(mu['RMSE'], sd['RMSE'])
            mae = fmt_ms(mu['MAE'], sd['MAE'])
            r2 = fmt_r2(mu['R2'], sd['R2'])
            nrmse = 100 * mu['RMSE'] / CAP
            print(f"    {LABEL[m]} & {rmse} & {nrmse:.1f} & {mae} & "
                  f"{r2} &  ({n} seeds)")
        for ref in ('persistence', 'climatology'):
            if ref in tab:
                r = tab[ref]
                print(f"    {ref} & {r['RMSE']:.3f} & "
                      f"{100*r['RMSE']/CAP:.1f} & {r['MAE']:.3f} & "
                      f"{r['R2']:.4f}")

    print()
    print('=' * 72)
    print('TABLE tab:controls rows (RMSE mean±std, DM vs QFAM)')
    print('=' * 72)
    for pl in ('P1', 'P3'):
        tab = S['tables'][pl]
        print(f'--- {pl} ---')
        for m in ('QFAM', 'QFAM_C', 'QFAM_R', 'FAM'):
            if m not in tab:
                continue
            mu, sd = tab[m]['mean'], tab[m]['std']
            cell = f"{fmt_ms(mu['RMSE'], sd['RMSE'])}"
            dm = S['dm'].get(f'{pl}_{m}', [])
            if m == 'QFAM':
                print(f"    QFAM & {cell} & --- ")
            elif dm:
                stats = [d['stat'] for d in dm]
                ps = [d['p'] for d in dm]
                print(f"    {LABEL[m]} & {cell} & "
                      f"{np.mean(stats):+.2f} (p "
                      f"{min(ps):.3f}-{max(ps):.3f}, n={len(dm)})")
        # rolling-origin
        roll_by_model = {}
        for key, v in S['rolling'].items():
            # key format: MODEL_P{pl}_o{offset}; MODEL may contain '_'
            # (e.g. QFAM_C), so split from the right
            mm, pp, oo = key.rsplit('_', 2)
            roll_by_model.setdefault((pp, mm), []).append(v['RMSE'])
        for m in ('QFAM', 'QFAM_C', 'FAM'):
            vals = roll_by_model.get((pl, m))
            if vals:
                ref = tab.get('QFAM', {}).get('mean', {}).get('RMSE')
                allv = list(vals)
                if ref:
                    allv.append(ref)  # include offset-0 reference
                print(f"    rolling {LABEL[m]}: offsets RMSE "
                      f"{['%.3f' % v for v in vals]} + ref "
                      f"{ref if ref else '?'} -> mean "
                      f"{np.mean(allv):.3f} over {len(allv)} origins")

    print()
    print('=' * 72)
    print('HARDWARE TABLE lower block (regenerated seed-42 checkpoint)')
    print('=' * 72)
    for pl, tag in (('P1', '1 h'), ('P3', '3 h')):
        dirs = sorted(glob.glob(os.path.join(
            root, f'QFAM_s42_{pl}_o0_*')))
        d = dirs[-1] if dirs else None
        if not d:
            print(f'--- {tag}: NO RUN DIR')
            continue
        with open(os.path.join(d, 'config.json'), encoding='utf-8') as f:
            cfg = json.load(f)
        # config metrics: 'step1'..'stepN' + 'overall' (mean over steps).
        # The reported horizon is step1 for P1 and step3 for P3.
        clean = cfg['metrics'][f'step{1 if pl == "P1" else 3}']
        print(f"--- {tag} (dir {os.path.basename(d)}) ---")
        print(f"    Sim. (noise-free)  & {clean['MSE']:.2f} & "
              f"{clean['RMSE']:.3f} & {clean['MAE']:.3f} & "
              f"{clean['R2']:.4f}")
        npath = os.path.join(d, 'noisy_eval.json')
        if os.path.exists(npath):
            with open(npath, encoding='utf-8') as f:
                nz = json.load(f)['metrics']
            # noisy_eval stores per-step metrics; the reported horizon is
            # step 1 for P1 runs and step 3 for P3 runs
            nz = nz.get(f'step{1 if pl == "P1" else 3}')
            print(f"    Sim. (noise-aware)  & {nz['MSE']:.2f} & "
                  f"{nz['RMSE']:.3f} & {nz['MAE']:.3f} & "
                  f"{nz['R2']:.4f}")
        else:
            print('    Sim. (noise-aware)  & (noisy_eval.json missing)')
        rpath = os.path.join(d, 'random_angle_eval.json')
        if os.path.exists(rpath):
            with open(rpath, encoding='utf-8') as f:
                ra = json.load(f)
            t = ra['trained']['RMSE']
            rmu, rsd = ra['random']['RMSE']['mean'], ra['random']['RMSE']['std']
            print(f"    random-angle: trained RMSE {t:.3f}; 10 random "
                  f"draws {rmu:.3f}±{rsd:.3f}")

    print()
    print('=' * 72)
    print('SKILL vs persistence (QFAM mean)')
    print('=' * 72)
    for pl in ('P1', 'P3'):
        tab = S['tables'][pl]
        if 'QFAM' in tab and 'persistence' in tab:
            q = tab['QFAM']['mean']['RMSE']
            p = tab['persistence']['RMSE']
            print(f"    {pl}: 1 - {q:.3f}/{p:.3f} = {100*(1-q/p):.1f}% "
                  f"(RMSE skill); nRMSE {100*q/CAP:.2f}%")


if __name__ == '__main__':
    main()
