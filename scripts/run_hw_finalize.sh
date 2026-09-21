#!/bin/bash
# run_hw_finalize.sh — relay collector: keep collecting until all tags
# are state 3, then splice everything and emit the cost/rows summary.
cd /data1/home/wangwei/Huaneng_0915/QFAM_code
PY=/data1/home/wangwei/miniforge3/envs/wang_py310/bin/python
CAP1=results_revision/hw_capture_P1.npz
CAP3=results_revision/hw_capture_P3.npz
RD1=results_revision/QFAM_s42_P1_o0_20260915_111231
RD3=results_revision/QFAM_s42_P3_o0_20260915_111231
OR6=0,2,4,6,8,10

TAGS="P1high P3high P1highNM P1rand0NM P1rand1NM P1rand2NM P1rand0MIT P1rand1MIT P1rand2MIT P3highNM P3rand0NM P3rand1NM P3rand2NM P3rand0MIT P3rand1MIT P3rand2MIT"

for round in $(seq 1 400); do
  pend_total=0
  for tag in $TAGS; do
    $PY -u revision/hw_run.py collect --tag $tag 2>&1 | tail -2
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
  echo "=== finalize round $round pending: $pend_total ==="
  [ "$pend_total" -eq 0 ] && break
  sleep 240
done

$PY -u revision/hw_run.py splice --tag P1high --capture $CAP1 --run_dir $RD1
$PY -u revision/hw_run.py splice --tag P3high --capture $CAP3 --run_dir $RD3
$PY -u revision/hw_run.py splice --tag P1highNM --capture $CAP1 --run_dir $RD1
$PY -u revision/hw_run.py splice --tag P3highNM --capture $CAP3 --run_dir $RD3
for i in 0 1 2; do
  $PY -u revision/hw_run.py splice --tag P1rand${i}NM --capture $CAP1 --run_dir $RD1
  $PY -u revision/hw_run.py splice --tag P1rand${i}MIT --capture $CAP1 --run_dir $RD1
  $PY -u revision/hw_run.py splice --tag P3rand${i}NM --capture $CAP3 --run_dir $RD3
  $PY -u revision/hw_run.py splice --tag P3rand${i}MIT --capture $CAP3 --run_dir $RD3
done
echo "=== cost accounting ==="
$PY -u revision/hw_cost.py
echo "=== latex rows ==="
$PY -u revision/make_hw_rows.py
echo FINALIZE_DONE
