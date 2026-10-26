# Hardware notes

The six-qubit, two-layer circuit is exported by `qfam_v2/qpanda3_circuit.py`.
It uses 36 variational angles, a CNOT ring and Z/X readout.

- Backend used in the manuscript: Wukong WK_C180_2.
- Shots: 1,000 per program.
- The repository contains **no API key**. Set `OQCS_KEY` in your local shell if
  you have access to the Origin Quantum cloud.
- The QFAM model executes one circuit per forecast, not one per time step.

Measured circuit-level timing (12 programs, 1,000 shots): 4,254 ms QPU, i.e.
0.3545 s per program. This is the basis of the QPU estimate in the manuscript:
0.57 h for three QFAM seeds, 2.04 h for the full comparison at 1,000 shots,
8.15 h at 4,000 shots. Queueing and calibration are excluded.

Do not commit credentials, QPU task JSON, or the proprietary `winf_fam1.csv`.
