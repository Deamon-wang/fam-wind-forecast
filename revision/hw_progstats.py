# -*- coding: utf-8 -*-
"""hw_progstats.py — exact logical-gate statistics of one deployed VQC
program (depth, op counts) for the Table 2 footnote."""
import json
import numpy as np

from pyqpanda3.core import QProg, RY, RX, CNOT, measure

cap = np.load('/data1/home/wangwei/Huaneng_0915/QFAM_code/'
              'results_revision/hw_capture_P1.npz')
angles = cap['angles']
alpha = angles[0][0, :, 0]
beta = angles[0][0, :, 1]
x = cap['xs'][0, 0, 0]

p = QProg()
for i in range(6):
    p << RY(i, float(x[i]))
for i in range(6):
    p << RX(i, float(alpha[i]))
    p << RY(i, float(beta[i]))
for i in range(6):
    p << CNOT(i, (i + 1) % 6)
p << measure(list(range(6)), list(range(6)))

print('depth:', p.depth())
print('count_ops:', p.count_ops())
print('qubits_num:', p.qubits_num())
