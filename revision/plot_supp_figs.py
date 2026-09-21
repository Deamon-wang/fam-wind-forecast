# -*- coding: utf-8 -*-
"""plot_supp_figs.py — generate the two supplementary figures:
  figS1_hqp_position.pdf : FlowQAttention internal structure with the
                           HQP applied once per block (after head
                           recombination); N=2 blocks => 2 HQPs total.
  figS2_hqp_internals.pdf: HQP internals: bias-free 128->6 compression,
                           plain RY(x) encoding, one RX(a)RY(b) layer +
                           CNOT ring (L_q=1), <Z_i> readout, biased
                           6->128 re-expansion.

Usage: python plot_supp_figs.py <outdir>
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

BLUE, ORANGE, GREEN, GRAY = ('#4C72B0', '#DD8452', '#55A868', '#8C8C8C')


def box(ax, x, y, w, h, text, fc='#FFFFFF', ec='#333333', fs=9,
        lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0.02,rounding_size=0.02',
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, zorder=4)


def arrow(ax, x1, y1, x2, y2, color='#333333'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle='-|>', mutation_scale=12,
                                 color=color, lw=1.2, zorder=2))


def fig_s1(path):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis('off')
    box(ax, 0.15, 1.9, 1.05, 0.9, 'input\n$Z$ [B,T,128]', fs=8.5)
    box(ax, 1.55, 1.9, 1.25, 0.9, 'QKV projection\n+ head split\n'
        '(H=8, D=16)', fs=8)
    box(ax, 3.15, 1.9, 1.15, 0.9, 'sigmoid kernel\n'
        r'$\tilde Q=\sigma(Q)$, $\tilde K=\sigma(K)$', fs=7.5)
    box(ax, 4.65, 0.95, 2.05, 2.8,
        'flow-conserving\nlinear attention\n\n'
        r'KV agg. $\tilde K^{\top}(V\odot r)$' + '\n'
        r'sink gate $\sigma(\cdot)$, source softmax' + '\n'
        r'$O(TD^2)$', fs=8, fc='#F2F6FB')
    box(ax, 7.05, 1.9, 1.1, 0.9, 'head\nrecombination\n[.,.,.]\u2192128', fs=8)
    box(ax, 8.45, 1.9, 1.35, 0.9, 'HQP\n(once per block)\n128\u21926\u2192128',
        fs=8.5, fc='#FDF3EA', ec=ORANGE, lw=1.8)
    for x1, x2 in ((1.20, 1.55), (2.80, 3.15), (4.30, 4.65),
                   (6.70, 7.05), (8.15, 8.45)):
        arrow(ax, x1, 2.35, x2, 2.35)
    arrow(ax, 9.80, 2.35, 9.93, 2.35)
    ax.text(5.0, 4.25,
            'FlowQAttention branch of one dual-branch block '
            '(N = 2 blocks \u21d2 2 HQP instances in the full model)',
            ha='center', fontsize=10)
    ax.text(9.12, 1.55, 'residual + LayerNorm', ha='center',
            fontsize=7.5, color=GRAY)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_s2(path):
    fig = plt.figure(figsize=(10.5, 5.2))
    axb = fig.add_axes([0.02, 0.02, 0.44, 0.96])
    axb.set_xlim(0, 4.4)
    axb.set_ylim(0, 10.4)
    axb.axis('off')
    box(axb, 0.15, 8.6, 1.7, 1.2, 'attention output\n[B,T,128]\nflatten \u2192 [M,128]',
         fs=8)
    box(axb, 1.0, 6.9, 1.9, 1.1, 'linear pre-proj\n128\u21926, bias-free\n'
        r'$W_{\rm pre}\in\mathbb{R}^{128\times 6}$', fs=8)
    box(axb, 1.0, 5.2, 1.9, 1.1, 'VQC on 6 qubits\n(see right panel)',
        fs=8.5, fc='#FDF3EA', ec=ORANGE, lw=1.8)
    box(axb, 1.0, 3.5, 1.9, 1.1,
        r'$z_i=\langle Z_i\rangle\in[-1,1]$' + '\n6 expectation values',
        fs=8)
    box(axb, 1.0, 1.8, 1.9, 1.1, 'linear post-proj\n6\u2192128, with bias',
        fs=8)
    box(axb, 0.15, 0.2, 1.7, 1.0, 'reshape [B,T,128]', fs=8)
    for y1, y2 in ((8.6, 8.0), (8.0, 6.9 + 1.1), (6.9, 6.3), (6.3, 5.2 + 1.1),
                   (5.2, 4.6), (4.6, 3.5 + 1.1), (3.5, 2.9), (2.9, 1.8 + 1.1),
                   (1.8, 1.2)):
        arrow(axb, 1.95, y1, 1.95, y2)
    axb.text(2.2, 10.15, '(a) classical\u2013quantum interface',
             fontsize=10)

    # ---- quantum circuit panel ----
    axc = fig.add_axes([0.50, 0.02, 0.48, 0.96])
    axc.set_xlim(-0.6, 7.2)
    axc.set_ylim(-1.4, 6.6)
    axc.axis('off')
    n = 6
    for q in range(n):
        y = n - 1 - q
        axc.plot([0, 6.4], [y, y], color='k', lw=1.0, zorder=1)
        axc.text(-0.45, y, f'$q_{q}$  $|0\\rangle$', ha='left',
                 va='center', fontsize=9)
    # encoding column
    for q in range(n):
        y = n - 1 - q
        box(axc, 0.55, y - 0.28, 0.85, 0.56, '', fc='#E8F0FA', ec=BLUE,
            fs=8)
        axc.text(0.975, y, f'$RY(x_{q})$', ha='center', va='center',
                 fontsize=7.5, color=BLUE)
    # variational column
    for q in range(n):
        y = n - 1 - q
        box(axc, 2.0, y - 0.28, 1.35, 0.56, '', fc='#FDF3EA', ec=ORANGE,
            fs=8)
        axc.text(2.675, y, f'$RX(\\alpha_{q})RY(\\beta_{q})$',
                 ha='center', va='center', fontsize=6.8, color='#B05A1E')
    # CNOT ring
    cx = 4.15
    for j in range(n - 1):
        y1, y2 = n - 1 - j, n - 2 - j
        axc.add_patch(Circle((cx, y1), 0.07, fc='k', zorder=5))
        axc.add_patch(Circle((cx, y2), 0.13, fc='w', ec='k', lw=1.2,
                             zorder=5))
        axc.plot([cx, cx], [y1, y2], 'k-', lw=1.2, zorder=4)
    # wrap-around CNOT q5 -> q0 (drawn on the right)
    y5, y0 = 0, n - 1
    axc.add_patch(Circle((cx, y5), 0.07, fc='k', zorder=5))
    axc.add_patch(Circle((cx + 0.75, y0), 0.13, fc='w', ec='k', lw=1.2,
                         zorder=5))
    axc.plot([cx, cx + 0.75], [y5, y5], 'k-', lw=1.2, zorder=4)
    axc.plot([cx + 0.75, cx + 0.75], [y5, y0], 'k-', lw=1.2, zorder=4)
    # measurement
    for q in range(n):
        y = n - 1 - q
        axc.add_patch(Circle((5.6, y), 0.16, fc='w', ec='k', lw=1.2,
                             zorder=5))
        axc.add_patch(Circle((5.66, y + 0.05), 0.09, fc='k', zorder=6))
        axc.plot([5.48, 5.72], [y - 0.12, y + 0.12], 'k-', lw=1.2,
                 zorder=6)
        axc.text(6.05, y, f'$z_{q}=\\langle Z_{q}\\rangle$', fontsize=8,
                 va='center')
    axc.text(3.2, 6.35, '(b) VQC: plain $RY$ encoding, one '
             r'$RX(\alpha)RY(\beta)$ layer + CNOT ring ($L_q{=}1$)',
             fontsize=10, ha='center')
    axc.text(0.98, -0.85, 'encoding (input)', fontsize=8, ha='center',
             color=BLUE)
    axc.text(2.68, -0.85, 'trained angles', fontsize=8, ha='center',
             color='#B05A1E')
    axc.text(4.15, -0.85, 'entangling ring', fontsize=8, ha='center',
             color='k')
    axc.text(5.75, -0.85, 'readout', fontsize=8, ha='center', color='k')
    fig.savefig(path)
    plt.close(fig)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    fig_s1(os.path.join(out, 'figS1_hqp_position.pdf'))
    fig_s2(os.path.join(out, 'figS2_hqp_internals.pdf'))
    print('wrote figS1_hqp_position.pdf figS2_hqp_internals.pdf')


if __name__ == '__main__':
    main()
