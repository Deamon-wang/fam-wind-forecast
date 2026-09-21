#!/bin/bash
# run_sim_noisy12.sh — noise-aware 12-origin sim + HW bootstrap (CPU)
cd /data1/home/wangwei/Huaneng_0915/QFAM_code
PY=/data1/home/wangwei/miniforge3/envs/wang_py310/bin/python
RD=$1
CAP=$2
REG=$3
OUT=$4
setsid nohup env CUDA_VISIBLE_DEVICES="" $PY -u revision/sim_noisy12.py \
    --run_dir "$RD" --capture "$CAP" --reg "$REG" --out "$OUT" \
    > logs/sim_noisy12.log 2>&1 < /dev/null &
echo "launched $!"
