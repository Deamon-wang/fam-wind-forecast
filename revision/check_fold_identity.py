# -*- coding: utf-8 -*-
"""check_fold_identity.py — dense-matrix fold identity check (clean).

Builds the exact 64x64 unitary of the hw_run.vqc_prog circuit via
Kronecker products and a CNOT permutation matrix, for folds 1/2/3,
applies it to |0...0>, computes <Z_q> with the LSB hardware
convention (z_q = 1-2*bit_q of state index), and compares.
"""
import numpy as np

rng = np.random.default_rng(7)
N = 6
D = 1 << N


def kron1(U, q):
    """Full 64x64 op applying 2x2 U on qubit q (q=0 -> LSB)."""
    ops = [np.eye(2)] * N
    ops[q] = U
    M = np.array([[1.0]], dtype=complex)
    # kron order: qubit N-1 (MSB) first ... qubit 0 (LSB) last
    for q_ in range(N - 1, -1, -1):
        M = np.kron(M, ops[q_])
    return M


def cnot(c, t):
    P = np.zeros((D, D))
    for col in range(D):
        if (col >> c) & 1:                 # control bit set
            row = col ^ (1 << t)
        else:
            row = col
        P[row, col] = 1.0
    return P


def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rx(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def z_of_circuit(x6, ang, fold):
    psi = np.zeros(D, dtype=complex)
    psi[0] = 1.0
    for i in range(N):
        psi = kron1(ry(x6[i]), i) @ psi
    for l in range(2):
        for j in range(N):
            psi = kron1(rx(ang[l, j, 0]), j) @ psi
            psi = kron1(ry(ang[l, j, 1]), j) @ psi
        for i in range(N):                  # per-edge adjacent folding
            j = (i + 1) % N
            for _ in range(fold):
                psi = cnot(i, j) @ psi
    probs = np.abs(psi) ** 2
    assert abs(probs.sum() - 1) < 1e-12, probs.sum()
    return np.array([1 - 2 * ((np.arange(D) >> q) & 1) @ probs
                     for q in range(N)])


x6 = rng.uniform(-0.04, 0.04, 6)
ang = rng.uniform(-np.pi, np.pi, (2, 6, 2))

# --- minimal sanity: is cnot() an involution? ---
for (c, t) in [(0, 1), (5, 0), (3, 4)]:
    P = cnot(c, t)
    print(f'cnot({c},{t}) @ itself == I ?',
          float(np.abs(P @ P - np.eye(D)).max()))

zs = {}
for fold in (1, 2, 3):
    zs[fold] = z_of_circuit(x6, ang, fold)
    print(f'fold {fold}: z =', np.round(zs[fold], 8))
print('fold2 == fold1 :', np.abs(zs[2] - zs[1]).max())
print('fold3 == fold1 :', np.abs(zs[3] - zs[1]).max())
