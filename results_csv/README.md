# QFAM result tables

All values are external-farm (`winf_fam1`) results on 1,921 forecast
origins, three seeds (42, 123, 2024), 96-h input and 1-h horizon. The raw
Huaneng observations are proprietary and are not included.

## Files

| file | content |
|---|---|
| `main_comparison.csv` | per-seed FAM, QFAM, LSTM, CNN, Transformer, DLinear |
| `summary_main.csv` | mean/std across seeds for the main comparison |
| `ablation.csv` | per-seed FAM h64, FAM h60, QFAM, random-angle, Z-only |
| `summary_ablation.csv` | mean/std for the ablation |
| `readout_controls.csv` | per-seed trained/zero/mean/shuffled/random readout |
| `readout_controls_summary.csv` | mean/std for the readout controls |
| `all_results_long.csv` | all per-seed model/variant rows in one table |
| `paired_stats.csv` | origin-level paired QFAM-FAM DM/bootstrap analysis |
| `qgain_screening.csv` | per-seed QFAM-Q and QFAM-C screening |
| `qgain_summary.csv` | QFAM-Q/QFAM-C means and standard deviations |
| `qgain_paired_stats.csv` | paired tests for QFAM-Q and QFAM-C |
| `backend_parity.csv` | torch/PyVQNet/pyqpanda3 parity and metrics |
| `hardware_planned.csv` | **planned** hardware arms; no hardware results yet |
| `hardware_cost_estimate.csv` | QPU-time estimate at 1,000/4,000 shots |

## Column notes

- `params`: total trainable parameters.
- `RMSE_MW`, `MAE_MW`, `ramp_RMSE_MW`, `peak_RMSE_MW`: MW scale; ramp/peak use
  the top 20% of absolute change/observed power.
- `R2`: coefficient of determination on the normalized target.
- `n`: number of test origins.
- `delta_RMSE = RMSE_QFAM - RMSE_FAM`; negative favours QFAM.
- `ci95_low/high`: 10,000-resample bootstrap over origins.
- `DM_stat`, `p_value`: Diebold--Mariano with HAC lag 1, per seed.

## Important interpretation

The trained QFAM is better than FAM, but the zero/mean/shuffled readout controls
are indistinguishable from the trained readout. The improvement is therefore an
architecture effect, not a quantum-specific effect. See the manuscript.

## Regeneration

The per-seed rows come from `result.json`; paired statistics come from
`predictions.csv`; backend parity comes from `results/sim_backends/seed*.json`.
The public code is at https://github.com/Deamon-wang/fam-wind-forecast.
