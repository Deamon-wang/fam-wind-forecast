# -*- coding: utf-8 -*-
"""hw_cost.py — TRUE cloud consumption: unique taskIds across
all registries (aliases excluded), with pending projection."""
import glob
import json

seen = {}
for p in glob.glob('/data1/home/wangwei/Huaneng_0915/'
                   'QFAM_code/hw_jobs/reg_*.json'):
    tag = p.split('reg_')[1][:-5]
    if tag == 'test0':
        continue
    r = json.load(open(p))
    for j in r['jobs']:
        tid = j.get('job_id')
        if not tid:
            continue
        if tid not in seen or (j.get('taskState') == '3'
                               and seen[tid][1] != '3'):
            seen[tid] = (tag, j.get('taskState', 'new'),
                         j.get('qpuRunTime') or 0,
                         j.get('totalTime') or 0,
                         len(j['programs']))

qpu = sum(v[2] for v in seen.values() if v[1] == '3')
wall = sum(v[3] for v in seen.values() if v[1] == '3')
ndone = sum(1 for v in seen.values() if v[1] == '3')
pend = [v for v in seen.values() if v[1] != '3']
pend_progs = sum(v[4] for v in pend)
print(f'unique tasks: {len(seen)} (done {ndone}, pending {len(pend)})')
print(f'true QPU consumed: {qpu/1000:.0f} s')
print(f'sum task wall: {wall/1000:.0f} s, mean {wall/1000/max(ndone,1):.0f} s')
print(f'pending circuits: {pend_progs} -> projected QPU '
      f'{pend_progs*0.337:.0f} s')
print(f'total executed circuits (done tasks): '
      f'{sum(v[4] for v in seen.values() if v[1] == "3")}')
from collections import Counter
tags_p = Counter(v[0] for v in pend)
print('pending by tag:', dict(tags_p))
