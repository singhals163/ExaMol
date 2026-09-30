#!/bin/bash
# Extra seeds for the two cheapest replay settings, to measure their training cost more precisely.
# Every completed run is reported; nothing is selected on the outcome.
set -u
cd "$(dirname "$0")"
export PATH=$HOME/anaconda3/envs/examol/bin:$PATH
mkdir -p logs
export POOL=dev-snapshot NUM_TO_RUN=500 NUM_INITIAL=32 RETRAIN_EVERY=32 INFER_EVERY=8 \
       WORKERS=8 ENSEMBLE=4 EPOCHS=32 TIME_SCALE=0.05
for seed in 10 11 12 13 14; do
    for size in 0 16; do
        dir="runs/dev-snapshot_incremental_n500_k32_w8_ts0.05"
        [ "$size" -gt 0 ] && dir="${dir}_r${size}"
        dir="${dir}_s${seed}"
        done_count=$(grep -c '"status": "finished"' "$dir/simulation-results.json" 2>/dev/null || echo 0)
        if [ "$done_count" -ge 495 ]; then echo "$(date '+%F %T') skip size=$size seed=$seed (already complete)"; continue; fi
        echo "$(date '+%F %T') running replay=$size seed=$seed"
        if [ "$size" -gt 0 ]; then extra="REPLAY=$size"; else extra="REPLAY=0"; fi
        timeout 2700 env MODE=incremental SEED=$seed $extra taskset -c 5-9,15-19 examol run spec.py:spec \
            > "logs/$(date +%s)_extra_r${size}_s${seed}.log" 2>&1
        echo "$(date '+%F %T') finished replay=$size seed=$seed exit=$?"
    done
done
echo "$(date '+%F %T') EXTRA SEEDS COMPLETE"
