#!/bin/bash
# run_hw_campaign.sh — full hardware campaign v2: 200-prog job cap (chunk 192),
# submit-resubmit + collect loop each round until everything is state 3.
cd /data1/home/wangwei/Huaneng_0915/QFAM_code
PY=/data1/home/wangwei/miniforge3/envs/wang_py310/bin/python
CAP1=results_revision/hw_capture_P1.npz
CAP3=results_revision/hw_capture_P3.npz
RD1=results_revision/QFAM_s42_P1_o0_20260915_111231
RD3=results_revision/QFAM_s42_P3_o0_20260915_111231
OR6=0,2,4,6,8,10
CH=192

subs () {
  $PY -u revision/hw_run.py submit --tag P1high --capture $CAP1 \
      --mode high --folds 1,2,3 --blocks 0,1 --mitigate --chunk $CH
  $PY -u revision/hw_run.py submit --tag P3high --capture $CAP3 \
      --mode high --folds 1,2,3 --blocks 0,1 --mitigate --chunk $CH
  $PY -u revision/hw_run.py submit --tag P1highNM --capture $CAP1 \
      --mode high --folds 1 --blocks 0,1 --origins $OR6 --chunk $CH
  for i in 0 1 2; do
    $PY -u revision/hw_run.py submit --tag P1rand${i}NM --capture $CAP1 \
        --mode random$i --folds 1 --blocks 0,1 --origins $OR6 --chunk $CH
  done
  for i in 0 1 2; do
    $PY -u revision/hw_run.py submit --tag P1rand${i}MIT --capture $CAP1 \
        --mode random$i --folds 1,2,3 --blocks 0,1 --origins $OR6 \
        --mitigate --chunk $CH
  done
}

TAGS="P1high P3high P1highNM P1rand0NM P1rand1NM P1rand2NM P1rand0MIT P1rand1MIT P1rand2MIT"

for round in $(seq 1 80); do
  subs 2>&1 | grep -E "chunk|total" | tail -40
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

$PY -u revision/hw_run.py splice --tag P1high --capture $CAP1 --run_dir $RD1
$PY -u revision/hw_run.py splice --tag P3high --capture $CAP3 --run_dir $RD3
$PY -u revision/hw_run.py splice --tag P1highNM --capture $CAP1 --run_dir $RD1
for i in 0 1 2; do
  $PY -u revision/hw_run.py splice --tag P1rand${i}NM --capture $CAP1 \
      --run_dir $RD1
  $PY -u revision/hw_run.py splice --tag P1rand${i}MIT --capture $CAP1 \
      --run_dir $RD1
done
echo ALL_DONE
