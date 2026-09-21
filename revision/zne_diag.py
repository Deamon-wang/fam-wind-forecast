# -*- coding: utf-8 -*-
"""zne_diag.py — fold-wise <Z> diagnostics for a few circuits: how does
z(lambda) behave across lambda=1,2,3, and what does Richardson give?

Usage: python zne_diag.py <reg.json> <capture.npz> [n_show]
"""
import json
import sys

import numpy as np

REG, CAP = sys.argv[1], sys.argv[2]
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 8

cap = np.load(CAP)
zs_sim = cap['zs_sim']
xs = cap['xs']
O = xs.shape[1]

reg = json.load(open(REG))
cmap = {}
for j in reg['jobs']:
    if j.get('taskState') != '3' or 'counts' not in j:
        continue
    for meta, cnt in zip([tuple(p) for p in j['programs']], j['counts']):
        cmap[meta] = cnt


def z_of(cnt):
    keys = [int(k, 16) for k in cnt['key']]
    vals = np.asarray(cnt['value'], dtype=np.float64)
    tot = vals.sum()
    z = np.zeros(6)
    for i in range(6):
        bit = (np.asarray(keys) >> i) & 1
        z[i] = np.sum(vals * (1 - 2 * bit)) / max(tot, 1e-12)
    return z, tot


shown = 0
print('meta (oi,k,t)      |  z_sim[0:3]       |  z(l=1)[0:3]      |  '
      'z(l=2)            |  z(l=3)           | rich[0:3]')
for oi in range(O):
    for k in range(2):
        for t in range(0, 96, 12):
            if (oi, k, t, 1) not in cmap or shown >= NS:
                continue
            zs = {}
            ok = True
            for f in (1, 2, 3):
                if (oi, k, t, f) in cmap:
                    zs[f], _ = z_of(cmap[(oi, k, t, f)])
                else:
                    ok = False
            if not ok:
                continue
            rich = 3 * zs[1] - 3 * zs[2] + zs[3]
            s = zs_sim[k, oi, t]
            print(f'({oi:2d},{k},{t:2d})  | '
                  f'{np.round(s[:3], 3)} | {np.round(zs[1][:3], 3)} | '
                  f'{np.round(zs[2][:3], 3)} | {np.round(zs[3][:3], 3)} |'
                  f' {np.round(rich[:3], 3)}')
            shown += 1
    if shown >= NS:
        break

# aggregate: mean |z| and mean z per fold across all circuits
agg = {1: [], 2: [], 3: []}
for (oi, k, t, f), cnt in cmap.items():
    if f in agg:
        z, _ = z_of(cnt)
        agg[f].append(z)
for f in (1, 2, 3):
    a = np.asarray(agg[f])
    print(f'fold {f}: mean|z|={np.abs(a).mean():.4f} '
          f'mean z={a.mean(0).round(3)} n={len(a)}')
