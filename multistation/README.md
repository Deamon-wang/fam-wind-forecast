# Multi-station FAM / QFAM models

Model code for the multi-station experiments of the revised manuscript:
the FAM/QFAM/QFAM_C family and the four classical baselines, plus the quantum
circuits used for simulation and on-chip inference.

This directory contains **model definitions only** — no training/evaluation
infrastructure. The three FAM variants share one backbone
(`qfam/model_qmambaformer.py`) and differ **only** in the Flow-Attention
output projection.

## Models

| model | file | projection head | params |
|---|---|---|---:|
| FAM | `qfam/model_qmambaformer.py` | linear 128→128 (33,024) | 681,477 |
| QFAM | `qfam/model_qmambaformer.py` | 6-qubit VQC bottleneck 128→6→VQC→6→128 (24 angles) | 651,805 |
| QFAM_C | `qfam/model_qmambaformer.py` | GELU bottleneck (parameter-matched control) | 651,781 |
| LSTM | `qfam/baselines.py` | 2-layer unidirectional, d=128 | 226,561 |
| Transformer | `qfam/baselines.py` | 1-layer, d=8, 2 heads | 1,097 |
| Informer | `qfam/baselines.py` | 1-layer, d=204, label_len 24 | 1,307,233 |
| TimesNet | `qfam/baselines.py` | 1-layer, d=128, 6 kernels | 1,036,770 |

Backbone (identical across FAM/QFAM/QFAM_C): input projection → positional
encoding → 2 × (Flow-Attention branch + Mamba branch) with gated fusion →
multi-scale temporal features → prediction head. Configuration:
`d=128, heads=8, layers=2, T=96, K=22, pred_horizon=1, head_hidden=(64,)`.

### Projection variants (`proj_mode`)

- `linear` — FAM: classical linear output projection, the reference head.
- `quantum` — QFAM: `HybridQuantumProjector` compresses each head's output
  128→6, encodes the 6 features as rotation angles on 6 qubits, applies the
  trainable VQC, and maps the 6 measured expectations back to 128. Only the
  24 rotation angles are trainable — 3 orders of magnitude fewer than the
  33,024-weight linear projection it replaces (total −4.4%).
- `gelu` — QFAM_C: same bottleneck shape with a classical GELU in place of
  the VQC; isolates the quantum circuit from the dimensionality reduction.

## Quantum circuits

- `qfam/vqc_circuit.py` — pyvqnet VQC (`Angle_VQC_cir`) used in QFAM training
  and simulator inference.
- `qfam/vqc_circuit_chip.py` — pyqpanda3 circuits (`Angle_VQC_chip`,
  `Angle_VQC_chip_multi`) for on-chip (QPU) inference; loaded by
  `HybridQuantumProjector` when available.
- `qfam/mamba.py` — Mamba branch (self-contained torch/pyvqnet port).

## Usage

```python
import torch
from qfam import WindQMambaformer

cfg = dict(input_dim=22, model_dim=128, mamba_d_state=64, mamba_d_conv=4,
           mamba_expand=3, flow_attn_heads=8, pred_horizon=1, dropout=0.1,
           num_layers=2, qbits=6, qlayer=1, head_hidden=(64,))

fam  = WindQMambaformer(proj_mode="linear",  **cfg)   # FAM
qfam = WindQMambaformer(proj_mode="quantum", **cfg)   # QFAM

y = qfam(torch.randn(16, 96, 22))   # -> [16, 1]
```

Baselines: `from qfam import WindPowerLSTM, WindPowerSimpleTransformer,
WindPowerInformer, WindPowerTimesNet`.

Training protocol (manuscript): train on Location1 2017–2021 (70/15/15
chronological split), zero-shot evaluation on 6 stations, per-unit [0,1]
power, Adam lr=1e-3, batch 128, seed 42.

## Results (six-station mean per-unit RMSE, zero-shot)

| model | 1 h | 3 h |
|---|---:|---:|
| **QFAM** | **0.0467** | **0.0929** |
| QFAM_C | 0.0489 | 0.0943 |
| FAM | 0.0479 | 0.0953 |
| Informer | 0.0483 | 0.1043 |
| LSTM | 0.0485 | 0.1052 |
| Transformer | 0.0531 | 0.1056 |
| TimesNet | 0.0517 | 0.1142 |
| persistence | 0.0514 | 0.1172 |

QFAM is the only model ranked in the top four in all four
station-group × lead-time cells, and the quantum head cuts transfer
degradation on the two cross-region Huaneng farms relative to every
classical baseline. See the manuscript for per-station tables and the
on-chip inference study.

## Requirements

- PyTorch (tested 2.10/CUDA 12.6), `pyvqnet` (required by the Mamba/VQC
  modules), `numpy`.
- `mamba_ssm` is optional — the code falls back to `SimplifiedMamba`.
- `pyqpanda3` is optional and only needed for on-chip inference.

## License

Code: MIT, as per the repository root.
