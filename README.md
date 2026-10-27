# QFAM public repository

Code and the **public Location1 dataset** for the revised QFAM experiments.

## Data availability

- `dataset/Location1.csv` is the public Location1 benchmark used for training.
- `winf_fam1.csv` is a **proprietary wind farm provided by China Huaneng Group**
  and is **not** distributed in this repository. To reproduce the external
  results, obtain the file from the data provider and place it at
  `dataset/winf_fam1.csv` (it is git-ignored). The code will not download it.
- No API key, QPU credential, or private data is included.

## Model

- Classical FAM: 69,893 parameters.
- QFAM: FAM head reduced from 64 to 60 units and one six-qubit correction per
  forecast; 69,435 parameters.
- The correction uses six pooled power/wind features, a two-layer
  `RX-RY-RZ` circuit with a CNOT ring, and Z/X readout.
- The circuit is replayed by the torch state vector, PyVQNet and pyqpanda3.

Reported on the private external farm (three seeds):

| model | params | RMSE | ramp | peak |
|---|---:|---:|---:|---:|
| FAM | 69,893 | 6.181 | 11.460 | 6.998 |
| QFAM | 69,435 | 5.938 | 11.160 | 6.423 |

The zero-readout control is 5.915 MW, slightly better than the trained 5.938 MW:
the forecasting gain is a classical architecture effect, not a quantum-specific
one. See the manuscript for the full attribution and limitations.

## Install

Python 3.10 is recommended. Install a PyTorch build compatible with your CUDA
driver, then:

```bash
pip install -r requirements.txt
```

`pyvqnet` and `pyqpanda3` are optional for training but required for the
simulator-parity and hardware scripts.

## Quick start (public data only)

Within-site training/evaluation on Location1:

```bash
python -m qfam_v2.train \
  --csv_path dataset/Location1.csv \
  --model fam \
  --seed 42 --horizon 1 --epochs 25 --patience 7 --stride 4 \
  --out_dir results/fam_s42
```

QFAM:

```bash
python -m qfam_v2.train \
  --csv_path dataset/Location1.csv \
  --model fam_qh60lat \
  --seed 42 --horizon 1 --epochs 25 --patience 7 --stride 4 \
  --out_dir results/qfam_s42
```

The public run uses a chronological 70/15/15 split. The manuscript's external
number requires the private file:

```bash
python -m qfam_v2.train \
  --csv_path dataset/Location1.csv \
  --external_path dataset/winf_fam1.csv \
  --model fam_qh60lat \
  --seed 42 --horizon 1 --epochs 25 --patience 7 --stride 4 \
  --out_dir results/external/qfam_s42
```

## Simulator inference with PyVQNet and pyqpanda3

After training an external checkpoint:

```bash
python -m qfam_v2.sim_backend_eval \
  --checkpoint results/external/qfam_s42/best_weights.pt \
  --external_csv dataset/winf_fam1.csv \
  --out results/sim_backends.json
```

The script writes the torch, PyVQNet and pyqpanda3 metrics and the maximum
readout difference. On the development CPU, 1,921 windows take about 0.7 s
with PyVQNet and 5 s with pyqpanda3.

## Readout controls

The paper's central control replaces the 12 quantum expectations by zero, their
test-set mean or a shuffled vector, without retraining:

```bash
python -m qfam_v2.readout_controls \
  --checkpoint results/external/qfam_s42/best_weights.pt \
  --external_csv dataset/winf_fam1.csv \
  --out results/readout_controls.json
```

`results/readout_controls.json` reports RMSE for the trained, zero, mean and
shuffled readouts. The zero-readout control is the one that tests whether the
quantum features are necessary.

## Hardware

The six-qubit circuit is exported with `qfam_v2/qpanda3_circuit.py`. The
repository does not contain a QPU credential. Set `OQCS_KEY` locally before any
hardware submission; never commit it. The circuit-level QPU estimate in the
manuscript is 0.3545 s per 1,000-shot program, about 0.57 h per three-seed
QFAM run and about 2.04 h for the full comparison at 1,000 shots. Queueing and
calibration are not included.

## Result tables

Per-seed comparison, ablation, readout-control, paired-statistics and backend
parity tables are in `results_csv/`. They are aggregate metrics only; the raw
`winf_fam1` observations are not included.

## Repository layout

```
qfam_v2/          training, data, models, circuit export, backend replay
revision/         minimal Mamba/positional-encoding dependency
dataset/          Location1.csv (public); winf_fam1.csv is not included
results_csv/      per-seed result and ablation tables
scripts/          runnable examples
hardware/         hardware notes (no credentials)
```

## License

Code: MIT (`LICENSE`). Location1 data: see `DATA_LICENSE.md`. The proprietary
Huaneng file is not covered by this repository and must not be redistributed.

## Citation

If you use this code, cite the QFAM manuscript and the Location1 benchmark.
