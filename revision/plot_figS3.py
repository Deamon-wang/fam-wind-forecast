# -*- coding: utf-8 -*-
"""plot_figS3.py — Fig S3: current calibrated working partition of
WK_C180_2 (17 qubits, hexagonal lattice) with the selected 6-qubit ring
highlighted, plus the three random control subgraphs.

Usage: python plot_figS3.py <chip_calibration.json> <selection.json> <out.pdf>
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

CAL, SEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]

snap = json.load(open(CAL))
sel = json.load(open(SEL))

pos = {39: (0.0, 5.0), 40: (1.5, 5.0),
       48: (-0.75, 4.0), 49: (0.75, 4.0), 50: (2.25, 4.0),
       57: (-0.75, 3.0), 58: (0.75, 3.0), 59: (2.25, 3.0),
       66: (-0.75, 2.0), 67: (0.75, 2.0), 68: (2.25, 2.0),
       75: (-0.75, 1.0), 76: (0.75, 1.0), 77: (2.25, 1.0),
       84: (-0.75, 0.0), 85: (0.75, 0.0), 86: (2.25, 0.0)}

edges = {tuple(sorted(d['qubits'])) for d in snap['double']}
ring = sel['selected_ring_order']
ring_set = set(ring)
ring_edges = [tuple(sorted((ring[i], ring[(i + 1) % 6])))
              for i in range(6)]

fig, ax = plt.subplots(figsize=(6.4, 7.4))
for a, b in edges:
    is_ring = tuple(sorted((a, b))) in ring_edges
    ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
            color='#E08214' if is_ring else '#BBBBBB',
            lw=3.2 if is_ring else 1.1, zorder=1,
            solid_capstyle='round')
for q, (x, y) in pos.items():
    in_ring = q in ring_set
    ax.add_patch(Circle((x, y), 0.24,
                        fc='#FDB863' if in_ring else '#F5F5F5',
                        ec='#E08214' if in_ring else '#999999',
                        lw=2.0 if in_ring else 1.0, zorder=3))
    r = snap['single'][str(q)]
    ax.text(x, y + 0.035, str(q), ha='center', va='center',
            fontsize=9, zorder=4,
            fontweight='bold' if in_ring else 'normal')
    ax.text(x, y - 0.42,
            f"T1 {r['t1']:.0f} T2 {r['t2']:.1f}",
            ha='center', va='top', fontsize=5.6, color='#555555',
            zorder=4)
    ax.text(x, y - 0.86,
            f"RO {r['readout_fidelity']:.3f}",
            ha='center', va='top', fontsize=5.6, color='#555555',
            zorder=4)

# ring order arrows
for i in range(6):
    a, b = ring[i], ring[(i + 1) % 6]
    xa, ya = pos[a]
    xb, yb = pos[b]
    ax.annotate('', xy=(xb, yb), xytext=(xa, ya),
                arrowprops=dict(arrowstyle='-|>', color='#7D4A0E',
                                lw=1.4, shrinkA=26, shrinkB=26),
                zorder=5)

ax.set_title('WK_C180_2 working partition (calibrated 2026-09-16):\n'
             'selected 6-qubit ring '
             r'$\{40, 49, 58, 68, 59, 50\}$' +
             ' (CNOT order annotated)', fontsize=10)
ax.set_xlim(-1.6, 3.1)
ax.set_ylim(-1.35, 5.6)
ax.set_aspect('equal')
ax.axis('off')
fig.tight_layout()
fig.savefig(OUT)
plt.close(fig)
print('wrote', OUT)
