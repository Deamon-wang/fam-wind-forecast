# -*- coding: utf-8 -*-
"""z_level_stats.py — compare <Z> fidelity across ablation configs:
mean |z_config - z_sim| per tag, and per-config metric deltas.

Usage: python z_level_stats.py <capture.npz> <tag1> [<tag2> ...]
"""
import sys

import numpy as np

CAP = sys.argv[1]
TAGS = sys.argv[2:]
R = '/data1/home/wangwei/Huaneng_0915/QFAM_code/results_revision'

cap = np.load(CAP)
zs_sim = cap['zs_sim']
print('tag              mean|dz|  med|dz|  corr(z,sim)')
for tag in TAGS:
    try:
        d = np.load(f'{R}/hw_z_{tag}.npz')
    except FileNotFoundError:
        print(f'{tag:16s} MISSING')
        continue
    z, cov = d['z_hw'], d['cov']
    m = cov > 0
    dz = np.abs(z[m] - zs_sim[m])
    a, b = z[m].ravel(), zs_sim[m].ravel()
    corr = float(np.corrcoef(a, b)[0, 1])
    print(f'{tag:16s} {dz.mean():8.4f}  {np.median(dz):7.4f}  {corr:9.4f}'
          f'   (n={int(m.sum())})')
