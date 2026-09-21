# -*- coding: utf-8 -*-
"""hw_select.py — high-fidelity 6-qubit ring selection on the CURRENT
WK_C180_2 calibration snapshot, per main-text Eq. (4):

    QF_i = (T1_i/T1max)^w1 (T2_i/T2max)^w2 (F1q_i)^w3 (F2q_avg_i)^w4

Weights used for this revision's regenerated allocation (documented in
the paper; the archived deployment's exact weights are not preserved in
the code archive): w = (0.1, 0.1, 0.3, 0.5) -- gate fidelities weighted
over coherence for a <2 us shallow circuit.

Search: enumerate all 6-cycles in the available coupling graph, score by
geometric-mean QF x mean ring CZ fidelity; also draw k random connected
6-subgraphs (Table 8 baselines) with a fixed RNG seed.

Usage: python hw_select.py <chip_calibration.json> <selection.json> [k]
"""
import json
import random
import sys
from itertools import combinations

CAL, OUT = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 5
W = (0.1, 0.1, 0.3, 0.5)

snap = json.load(open(CAL))
avail = snap['available']
sq = snap['single']
edges = {tuple(sorted(d['qubits'])): d['cz_fidelity']
         for d in snap['double']}
topo = snap['topology']

t1m = max(sq[str(q)]['t1'] for q in avail)
t2m = max(sq[str(q)]['t2'] for q in avail)


def f2q_avg(q):
    vals = [f for (a, b), f in edges.items() if q in (a, b)]
    return sum(vals) / len(vals) if vals else 0.0


def qf(q):
    r = sq[str(q)]
    return ((r['t1'] / t1m) ** W[0] * (r['t2'] / t2m) ** W[1]
            * r['single_gate_fidelity'] ** W[2] * f2q_avg(q) ** W[3])


adj = {q: set() for q in avail}
for a, b in edges:
    if a in adj and b in adj:
        adj[a].add(b)
        adj[b].add(a)

# ---- enumerate 6-cycles (canonical, undirected) ----
cycles = set()


def dfs(path):
    if len(path) == 6:
        if path[0] in adj[path[-1]]:
            c = tuple(sorted(path))
            cycles.add((c, tuple(path)))
        return
    for nxt in adj[path[-1]]:
        if nxt in path:
            continue
        # canonical pruning: smallest vertex must be path[0] to cut
        # rotations/reflections; allow closure only at the end
        if nxt < path[0]:
            continue
        dfs(path + [nxt])


for v in avail:
    dfs([v])
uniq = {c: ring for c, ring in cycles}
print(f'found {len(uniq)} distinct 6-cycles')


def score(ring):
    ring_edges = [(ring[i], ring[(i + 1) % 6]) for i in range(6)]
    fids = [edges.get(tuple(sorted(e)), 0.0) for e in ring_edges]
    mean_cz = sum(fids) / 6
    gqf = 1.0
    for q in ring:
        gqf *= qf(q)
    gqf **= (1 / 6)
    return gqf * mean_cz, mean_cz, gqf


ranked = sorted(((score(r), r) for r in uniq.values()),
                key=lambda t: -t[0][0])
best_score, best_ring = ranked[0]
print('best ring (in ring order):', best_ring)
print('  joint=%.5f mean_CZ=%.5f geoQF=%.5f' % best_score)
print('  per-qubit QF:', {q: round(qf(q), 5) for q in best_ring})
print('  ring CZ:', {e: edges.get(e) for e in
                     [tuple(sorted((best_ring[i],
                                    best_ring[(i + 1) % 6])))
                      for i in range(6)]})

# runner-up for reference
for sc, ring in ranked[1:3]:
    print('alt:', ring, 'joint=%.5f mean_CZ=%.5f' % (sc[0], sc[1]))

# ---- random connected 6-subgraphs (Table 8 baselines) ----
rng = random.Random(20260916)
random_sets = []
tries = 0
while len(random_sets) < K and tries < 10000:
    tries += 1
    seed_q = rng.choice(avail)
    sub = {seed_q}
    frontier = [seed_q]
    while len(sub) < 6 and frontier:
        v = rng.choice(frontier)
        cand = [u for u in adj[v] if u not in sub]
        if not cand:
            frontier.remove(v)
            continue
        u = rng.choice(cand)
        sub.add(u)
        frontier.append(u)
    if len(sub) == 6:
        key = tuple(sorted(sub))
        if key not in random_sets and key != tuple(sorted(best_ring)):
            random_sets.append(key)
print('random control subgraphs:', random_sets)

sel = {'weights_w1_w4': list(W), 't1_max': t1m, 't2_max': t2m,
       'selected_ring_order': list(best_ring),
       'selected_sorted': sorted(best_ring),
       'joint_score': best_score[0], 'mean_ring_cz': best_score[1],
       'geo_qf': best_score[2],
       'per_qubit_qf': {str(q): qf(q) for q in best_ring},
       'ring_edges_cz': {str(e): edges.get(e) for e in
                         [tuple(sorted((best_ring[i],
                                        best_ring[(i + 1) % 6])))
                          for i in range(6)]},
       'random_controls': [list(s) for s in random_sets],
       'n_cycles_found': len(uniq),
       'available': avail}
json.dump(sel, open(OUT, 'w'), indent=1)
print('wrote', OUT)
