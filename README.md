# QFAM Revision Package (EGY-D-26-19497)

Code and hardware-session evidence for the revised manuscript
**"QFAM: A hybrid quantum–classical flow-attention Mamba framework for
short-term wind power forecasting on NISQ hardware"** (*Energy*,
submission EGY-D-26-19497).

## Directory layout

```
QFAM_code/
├── README.md               this file
├── requirements.txt        python dependencies
├── dataset/                preprocessed farm data (CSV + scalers)
├── src/                    archived original training/inference modules
│   └── (model_qmambaformer.py is the archived VQC inference reference)
├── revision/               canonical revision package (see below)
├── scripts/                top-level campaign drivers (bash)
├── hw_jobs/                hardware run registries (per-tag JSON, taskIds
│                           + measured counts) — evidence for Sect. 3.3
├── results_revision/       all regenerated outputs (checkpoints, eval
│                           JSONs, hardware captures, table row dumps)
├── figures_new/            regenerated manuscript figures
├── logs/                   driver/cloud-session logs (wall-clock evidence)
└── archive/                superseded originals and early-exploration
    outputs (retained for provenance only; not used by the revision)
```

## revision/ modules

**Core training & evaluation (classical + simulated VQC)**
| file | role |
|---|---|
| `models_rev.py` | QFAM / FAM / QFAM-C / QFAM-R / baselines (MLP, LSTM, Transformer) |
| `run_revision.py` | one training/eval run (seed, horizon, model, rolling origin) |
| `run_all.py` | full 88-run experiment matrix (7 models × 5 seeds × 2 horizons + rolling) |
| `postprocess.py` | 5-seed aggregates, Diebold–Mariano tests, Table 5/6 rows |
| `noisy_eval.py` | density-matrix noise-aware evaluation (p, T1, T2 sweeps) |
| `rand_angle_eval.py` | frozen-random-angle control evaluation |
| `residual_analysis.py` | power-level / ramp-regime residual breakdown (Sect. 5.3) |
| `make_tex_rows.py` | LaTeX table rows for the classical-result tables |

**Hardware pipeline (pyqpanda3 / Wukong WK_C180_2)**
| file | role |
|---|---|
| `hw_chipinfo.py` | fetch/refresh calibration snapshot (Table 2) |
| `hw_select.py` | connectivity + fidelity scoring, ring & k=3 random subgraphs |
| `hw_capture.py` | same-origin reference capture (x, z_sim, y_sim per origin) |
| `hw_run.py` | submit / poll / collect / splice; VQC circuit builder incl. CNOT-replica folding; M3 readout amendment; two-level ZNE (3f₁−f₃)/2 on odd folds {1,3} |
| `hw_dedupe.py`, `hw_progstats.py` | registry utilities |
| `hw_cost.py` | taskId-deduplicated true QPU consumption (Sect. 3.3, S10) |
| `make_hw_rows.py` | Table 7 / Table 8 rows from hw_eval JSONs |

**Verification & analysis**
| file | role |
|---|---|
| `sim_check.py` | capture-vs-training-CSV identity check (max |Δz| ~5e-6) |
| `check_fold_identity.py` | dense-matrix proof that even CNOT replicas change the logical circuit (why folds {1,3} only) |
| `zne_diag.py` | per-fold ⟨Z⟩ diagnostics that exposed the λ=2 bias |
| `sim_noisy12.py` | 12-origin noise-aware sim + multinomial shot-noise bootstrap |
| `subset_band.py` | subset-selection variability σ (1.38 / 2.31 MW) |
| `z_level_stats.py` | mean |Δz| and correlation per configuration |

**Figures**: `plot_figs.py`, `plot_figS3.py`, `plot_supp_figs.py`.

## scripts/ drivers

```bash
bash scripts/run_hw_campaign.sh     # P1/P3 main validation + P1 ablation tags
bash scripts/run_hw_p3abl.sh        # P3 2×2 ablation tags (k=3 random subgraphs)
bash scripts/run_hw_finalize.sh     # poll→collect→splice→cost→table-rows relay
bash scripts/run_sim_noisy12.sh     # noise-aware + bootstrap analysis
```

## Reproduction order

1. `python revision/run_all.py` → classical matrix (Tables 5–6)
2. `python revision/noisy_eval.py`, `rand_angle_eval.py`, `residual_analysis.py`
3. hardware: `scripts/run_hw_campaign.sh` then `scripts/run_hw_finalize.sh`
   (needs `QPU_API_KEY` environment variable; inference only — no training
   on QPU; ≈0.33 s/circuit at 1,000 shots)
4. `python revision/make_hw_rows.py` → Table 7 / 8 rows
5. campaign totals: `python revision/hw_cost.py`

## Naming conventions

- `hw_*` — hardware-session modules; `run_*` — executable drivers;
  `sim_*` — simulation-side analyses; `check_*` — verification utilities;
  `plot_*`/`fig*` — figure generation; `make_*` — table/row builders.
- Hardware registry tags: `P{1,3}{high,rand[0-2]}{NM,MIT}[6]`
  (NM = no mitigation, MIT = M3+ZNE, `6` = 6-origin ablation subset).

## Hardware session summary (2026-09-16, chip WK_C180_2)

228 cloud tasks, 43,776 circuits, 43.8 M shots, 14.5 ks (4.0 h) QPU;
ring {40, 49, 58, 68, 59, 50}; random subgraphs (seed 20260916):
{68,75,76,77,85,86}, {58,67,75,76,85,86}, {67,68,76,77,85,86}.
Regenerated results reach parity with same-origin simulation at both
horizons (−0.2 % / +0.5 % RMSE) — see Table 7 of the manuscript.

---

## Public-release notes

- **Data**: the proprietary operational wind-farm datasets
  (`dataset/winf_fam1.csv`, `winf_fam2.csv`) are **not included**; they are
  available from the corresponding author upon reasonable request, subject
  to a data-sharing agreement. The expected schema is documented in
  `src/data_processing_winf.py` and Table 1 of the paper.
- **Hardware access**: `revision/hw_run.py` reads the Origin Quantum cloud
  API key from the `OQCS_KEY` environment variable; no key is stored in
  this repository.
- **Evidence**: `hw_jobs/`, `logs/`, and `results_revision/` contain the
  complete hardware-session evidence (task registries with taskIds,
  wall-clock logs, per-tag evaluation JSONs) and all result tables'
  producing outputs.
- **Provenance**: `archive/` retains superseded early configurations
  (parameter configs kept; csv prediction outputs removed).
- **Checkpoints**: trained model weights (`best.pt`, 98 runs) are excluded
  to keep the repository light; every reported metric is reproducible from
  the included configs + `run_all.py` (deterministic seeds), and the
  per-run `config_*.json` / `metrics` outputs are retained under
  `results_revision/`.
