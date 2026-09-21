# Regenerate the two-panel Wukong coupling-lattice figure (PDF + SVG),
# print-optimized font sizes (figure included at full \linewidth ~7.1in,
# 0.57x reduction -> in-figure fonts sized so printed sizes stay >=5pt).
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch
plt.rcParams['pdf.fonttype'] = 42   # TrueType: avoids Type-3 ghosting in some viewers
plt.rcParams['svg.fonttype'] = 'none'  # real text in SVG (editable, crisp)
plt.rcParams['font.family'] = 'DejaVu Sans'

G = json.load(open('lattice_graph.json'))

CAL = {  # qubit: (T1 us, T2echo us, FRO)  -- T2 = T2,Echo per export
 39: (25.28, 11.26, 0.9802), 40: (47.39, 14.67, 0.9359),
 48: (34.17, 18.58, 0.9622), 49: (38.47, 14.79, 0.9472),
 50: (40.44, 20.55, 0.9602), 57: (51.34, 14.72, 0.9287),
 58: (33.03, 44.12, 0.9453), 59: (37.54, 15.88, 0.9464),
 66: (20.89, 15.16, 0.9038), 67: (41.25, 15.58, 0.9505),
 68: (54.78, 46.52, 0.9419), 75: (19.08, 27.15, 0.9436),
 76: (32.36, 34.01, 0.9334), 77: (48.44, 39.41, 0.9713),
 84: (22.19, 11.94, 0.9632), 85: (43.04, 14.33, 0.9185),
 86: (4.36, 16.78, 0.9418),
}
PART = set(CAL)
RING = [40, 49, 58, 68, 59, 50]
C_GRAY, C_TEAL, C_RED, C_DARK, C_NODE = '#c9cfd5', '#207a78', '#c13227', '#263238', '#8a949c'

apos = {lab: (x, y) for x, y, lab in G['a_nodes']}
seen = {}
for u, v, s in G['a_edges']:
    key = tuple(sorted((u, v)))
    pri = {'#c94b3c': 2, '#207a78': 1}.get(s, 0)
    if key not in seen or pri > seen[key][0]:
        seen[key] = (pri, s)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 6.6),
                               gridspec_kw={'width_ratios': [1, 2.9]})

# ---- panel (a): full 180-qubit lattice ----
for (u, v), (pri, s) in sorted(seen.items(), key=lambda kv: -kv[1][0]):
    x1, y1 = apos[u]; x2, y2 = apos[v]
    col = {2: C_RED, 1: C_TEAL, 0: C_GRAY}[pri]
    axA.plot([x1, x2], [-y1, -y2], color=col,
             lw=2.6 if pri else 1.4, zorder=1 + pri, solid_capstyle='round')
for q, (x, y) in apos.items():
    if q in PART:
        axA.plot(x, -y, 'o', ms=8, color=C_TEAL, zorder=4)
    else:
        axA.plot(x, -y, 'o', ms=4.4, color=C_NODE, zorder=2)
    axA.annotate(str(q), (x, -y), ha='center', va='center',
                 fontsize=6.2, color=C_DARK, zorder=5)
axA.set_title('(a) Full 180-qubit Wukong coupling topology',
              fontsize=15, color=C_DARK, pad=10)
axA.set_aspect('equal'); axA.axis('off')
axA.legend(handles=[
    Patch(color=C_TEAL, label='17-qubit calibrated\nworking partition'),
    Patch(color=C_NODE, label='Other processor qubits'),
    Patch(color=C_RED, label='Selected six-qubit VQC ring')],
    loc='upper right', fontsize=10, framealpha=0.95)

# ---- panel (b): calibrated partition, enlarged ----
bpos = {int(lab[1:]): (x, y) for x, y, lab in G['b_nodes']}
for u, v, s in G['b_edges']:
    u, v = int(u[1:]), int(v[1:])
    x1, y1 = bpos[u]; x2, y2 = bpos[v]
    axB.plot([x1, x2], [-y1, -y2], color='#929aa2', lw=3.2, zorder=1)
for i in range(6):
    u, v = RING[i], RING[(i + 1) % 6]
    x1, y1 = bpos[u]; x2, y2 = bpos[v]
    axB.add_patch(FancyArrowPatch((x1, -y1), (x2, -y2), arrowstyle='-|>',
                  mutation_scale=26, color=C_RED, lw=3.6, zorder=3,
                  shrinkA=16, shrinkB=16))
    mx, my = (x1 + x2) / 2, -(y1 + y2) / 2
    axB.annotate(str(i + 1), (mx, my), xytext=(0, 10),
                 textcoords='offset points', ha='center', fontsize=13,
                 color=C_RED, fontweight='bold', zorder=4)
cx = sum(x for x, _ in bpos.values()) / len(bpos)
cy = sum(-y for _, y in bpos.values()) / len(bpos)
for q, (x, y) in bpos.items():
    col = C_RED if q in RING else C_TEAL
    axB.plot(x, -y, 'o', ms=20, color=col, mec='white', mew=1.6, zorder=5)
    axB.annotate('q%d' % q, (x, -y), ha='center', va='center',
                 fontsize=10.5, color='white', fontweight='bold', zorder=6)
    t1, t2e, fro = CAL[q]
    dx, dy = x - cx, -y - cy
    n = (dx * dx + dy * dy) ** .5 or 1
    tx, ty = x + dx / n * 62, -y + dy / n * 54
    axB.annotate('T1/T2 = %.1f/%.1f $\mu$s\nF$_{RO}$ = %.2f%%'
                 % (t1, t2e, fro * 100),
                 (x, -y), xytext=(tx, ty), textcoords='data',
                 ha='center', va='center', fontsize=8.6, color=C_DARK,
                 zorder=6,
                 arrowprops=dict(arrowstyle='-', color='#b7bdc3', lw=1.0,
                                 shrinkA=11, shrinkB=3))
axB.set_title('(b) Calibrated working partition — 2026-09-16 snapshot\n'
              '(T2 denotes T2,Echo from the supplied calibration export)',
              fontsize=15, color=C_DARK, pad=10)
axB.set_aspect('equal'); axB.axis('off')

fig.tight_layout(rect=[0, 0, 1, 1.0])
fig.savefig('figS3_wukong_topology.pdf', bbox_inches='tight')
fig.savefig('figS3_wukong_topology.svg', bbox_inches='tight')
fig.savefig('figS3_wukong_topology.png', bbox_inches='tight', dpi=600)
print('written figS3_wukong_topology.{pdf,svg,png} (fonttype 42, png 600dpi)')
