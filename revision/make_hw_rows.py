# -*- coding: utf-8 -*-
"""make_hw_rows.py — format LaTeX rows for Table 7 (hardware) and
Table 8 (ablation) from the hw_eval_*.json files, plus accounting
numbers for Sect. 3.3.

Usage: python make_hw_rows.py
"""
import glob
import json
import os

import numpy as np

R = '/data1/home/wangwei/Huaneng_0915/QFAM_code/results_revision'


def load(tag):
    p = os.path.join(R, f'hw_eval_{tag}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(m, keys=('mse', 'rmse', 'mae', 'r2')):
    mse = m['rmse'] ** 2
    vals = [mse] + [m[k] for k in ('rmse', 'mae', 'r2')]
    return ' & '.join(f'{v:.6g}' if i == 3 else f'{v:.5g}'
                      for i, v in enumerate(vals))


def gaps(mhw, msim):
    out = []
    for k in ('rmse', 'mae'):
        out.append(f'{(mhw[k] / msim[k] - 1) * 100:+.1f}\\%')
    out.append(f'{mhw["r2"] - msim["r2"]:+.4f}')
    return ' & -- & ' + ' & '.join(out)


print('==== Table 7 lower block (regenerated protocol) ====')
for tag, name in (('P1high', '1~h'), ('P3high', '3~h')):
    e = load(tag)
    if e is None:
        print(f'[{name}] MISSING')
        continue
    print(f'[{name}]')
    print('  Wukong (real, M3+ZNE) & %s \\\\' % fmt(e['metrics_hw']))
    print('  gap vs sim (same origins) %s \\\\' %
          gaps(e['metrics_hw'], e['metrics_sim_same_origins']))
    print('  sim-same-origins: %s' % fmt(e['metrics_sim_same_origins']))
    print('  dz=%.4f coverage=%.2f' % (e['mean_abs_dz_vs_sim'],
                                       e['coverage']))

print('\n==== Table 8 (ablation, 6-origin subset) ====')
for pref, name in (('P1', '1~h'), ('P3', '3~h')):
    rows = {}
    e = load(f'{pref}highNM')
    if e:
        rows['High-fidelity, no mitigation'] = e['metrics_hw']
    for grp in ('NM', 'MIT'):
        es = [load(f'{pref}rand{i}{grp}') for i in range(3)]
        es = [x for x in es if x]
        if es:
            label = (f'Random+M3/ZNE' if grp == 'MIT'
                     else 'Random, no mitigation')
            rows[label] = {k: float(np.mean([e_['metrics_hw'][k]
                                             for e_ in es]))
                           for k in ('rmse', 'mae', 'r2')}
    print(f'--- {name} ---')
    for k, v in rows.items():
        print(f'{k:35s} & {fmt(v)} \\\\')

print('\n==== Accounting ====')
tot_prog = tot_shots = tot_wall = tot_qpu = 0
for p in glob.glob(os.path.join(R, 'hw_eval_*.json')):
    e = json.load(open(p))
    a = e['accounting']
    tag = os.path.basename(p)[8:-5]
    print(f"{tag:12s} progs={a['programs']:6d} shots={a['total_shots']:9d} "
          f"wall={a['total_time_ms'] / 1000:7.0f}s "
          f"qpu={a['qpu_time_ms'] / 1000:7.0f}s")
    if tag != 'test0':
        tot_prog += a['programs']
        tot_shots += a['total_shots']
        tot_wall += a['total_time_ms']
        tot_qpu += a['qpu_time_ms']
print(f'TOTAL (excl. test0): programs={tot_prog}, shots={tot_shots}, '
      f'wall={tot_wall / 3600:.2f} h, qpu={tot_qpu / 3600:.2f} h')
