"""pyqpanda3 export of the exact Q-FAM circuit.

The training backend is the batched torch state vector (models.TorchVQCv2);
this module builds the same unitary with pyqpanda3 and returns Z/X expectation
values, so the trained angles can be submitted to a qpanda3 backend without
re-encoding the circuit.
"""
from __future__ import annotations

import numpy as np


def build_program(angles, params, n_qubits=6, layers=2):
    """angles: [n] encoding angles; params: [layers,n,3] (RX,RY,RZ)."""
    import pyqpanda3.core as pq

    cir = pq.QCircuit(n_qubits)
    for i in range(n_qubits):
        cir << pq.RY(i, float(angles[i]))
    for l in range(layers):
        for i in range(n_qubits):
            cir << pq.RX(i, float(params[l, i, 0]))
            cir << pq.RY(i, float(params[l, i, 1]))
            cir << pq.RZ(i, float(params[l, i, 2]))
        for i in range(n_qubits):
            cir << pq.CNOT(i, (i + 1) % n_qubits)
    prog = pq.QProg()
    prog << cir
    return prog


def expvals(angles, params, n_qubits=6, layers=2, shots=0):
    """Return [Z0..Z(n-1), X0..X(n-1)] via pyqpanda3."""
    import pyqpanda3.core as pq
    from pyvqnet.qnn.pq3.measure import expval

    prog = build_program(angles, params, n_qubits, layers)
    machine = pq.CPUQVM()
    shots = int(shots) if shots else 1000
    machine.run(prog, shots=shots)
    z = [float(expval(machine, prog, {f"Z{i}": 1.0}, shots=shots)) for i in range(n_qubits)]
    x = [float(expval(machine, prog, {f"X{i}": 1.0}, shots=shots)) for i in range(n_qubits)]
    return np.asarray(z + x, dtype=np.float32)
