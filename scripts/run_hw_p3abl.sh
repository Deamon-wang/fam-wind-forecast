#!/bin/bash
# run_hw_p3abl.sh — P3-horizon allocation/mitigation ablation (Table 8,
# 3h column): same structure as v2 driver, P3 captures/checkpoint.
cd /data1/home/wangwei/Huaneng_0915/QFAM_code
PY=/data1/home/wangwei/miniforge3/envs/wang_py310/bin/python
CAP3=results_revision/hw_capture_P3.npz
RD3=results_revision/QFAM_s42_P3_o0_20260915_111231
OR6=0,2,4,6,8,10
CH=192

subs () {
  $PY -u revision/hw_run.py submit --tag P3highNM --capture $CAP3 \
      --mode high --folds 1 --blocks 0,1 --origins $OR6 --chunk $CH
  for i in 0 1 2; do
    $PY -u revision/hw_run.py submit --tag P3rand${i}NM --capture $CAP3 \
        --mode random$i --folds 1 --blocks 0,1 --origins $OR6 --chunk $CH
  done
  for i in 0 1 2; do
    $PY -u revision/hw_run.py submit --tag P3rand${i}MIT --capture $CAP3 \
        --mode random$i --folds 1,2,3 --blocks 0,1 --origins $OR6 \
        --mitigate --chunk $CH
  done
}

TAGS="P3highNM P3rand0NM P3rand1NM P3rand2NM P3rand0MIT P3rand1MIT P3rand2MIT"

for round in $(seq 1 100); do
  subs 2>&1 | grep -E "chunk" | tail -20
  pend_total=0
  for tag in $TAGS; do
    $PY -u revision/hw_run.py collect --tag $tag 2>&1 | tail -3
    pend=$($PY -c "
import json, os
p = 'hw_jobs/reg_%s.json' % '$tag'
if os.path.exists(p):
    r = json.load(open(p))
    print(sum(1 for j in r['jobs'] if j.get('taskState') != '3'))
else:
    print(0)
")
    pend_total=$((pend_total + pend))
  done
  echo "=== round $round pending total: $pend_total ==="
  [ "$pend_total" -eq 0 ] && break
  sleep 180
done

$PY -u revision/hw_run.py splice --tag P3highNM --capture $CAP3 --run_dir $RD3
for i in 0 1 2; do
  $PY -u revision/hw_run.py splice --tag P3rand${i}NM --capture $CAP3 \
      --run_dir $RD3
  $PY -u revision/hw_run.py splice --tag P3rand${i}MIT --capture $CAP3 \
      --run_dir $RD3
done
echo P3_ABLATION_DONE
