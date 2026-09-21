# -*- coding: utf-8 -*-
"""hw_dedupe.py — collapse duplicate jobs in registries: for each
program-set keep exactly one job, preferring state '3' (done) > '2'
(running) > '1' (queued) > None (new/unpolled) > '4' (failed)."""
import glob
import json

PRIO = {'3': 0, '2': 1, '1': 2, None: 3, '4': 4}


def key_of(job):
    progs = tuple(tuple(p) for p in job['programs'])
    st = job.get('taskState')
    return (progs, PRIO.get(st, 5))


for p in sorted(glob.glob('/data1/home/wangwei/Huaneng_0915/QFAM_code/'
                          'hw_jobs/reg_*.json')):
    reg = json.load(open(p))
    best = {}
    for j in reg['jobs']:
        k = tuple(tuple(x) for x in j['programs'])
        rank = PRIO.get(j.get('taskState'), 5)
        if k not in best or rank < best[k][0]:
            best[k] = (rank, j)
    new_jobs = [v[1] for v in best.values()]
    # preserve original submission order
    order = {id(j): i for i, j in enumerate(reg['jobs'])}
    new_jobs.sort(key=lambda j: order.get(id(j), 1 << 30))
    before, after = len(reg['jobs']), len(new_jobs)
    if after != before:
        reg['jobs'] = new_jobs
        json.dump(reg, open(p, 'w'), indent=1)
    print(f'{p.split("reg_")[1][:-5]:12s} {before:4d} -> {after:4d}')
