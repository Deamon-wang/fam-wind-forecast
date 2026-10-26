#!/usr/bin/env bash
set -euo pipefail
# Requires dataset/winf_fam1.csv, which is proprietary and not distributed.
if [ ! -f dataset/winf_fam1.csv ]; then
  echo "Missing dataset/winf_fam1.csv (proprietary; not included)." >&2
  exit 1
fi
python -m qfam_v2.train \
  --csv_path dataset/Location1.csv \
  --external_path dataset/winf_fam1.csv \
  --model fam_qh60lat \
  --seed 42 --horizon 1 --epochs 25 --patience 7 --stride 4 \
  --out_dir results/external/qfam_s42
