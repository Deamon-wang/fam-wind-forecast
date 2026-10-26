#!/usr/bin/env bash
set -euo pipefail
# Within-site Location1 example. The external Huaneng file is not included.
python -m qfam_v2.train \
  --csv_path dataset/Location1.csv \
  --model fam_qh60lat \
  --seed 42 --horizon 1 --epochs 25 --patience 7 --stride 4 \
  --out_dir results/qfam_location1_s42
