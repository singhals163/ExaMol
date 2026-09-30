#!/bin/bash
# Run every (replay size, seed) combination until each has a complete campaign.
# Re-checks after every pass, so a crashed or hung run (killed by RUN_TIMEOUT) is simply retried.
#   Usage: SIZES="16 32 64 128 256" SEEDS="5 6 7 8 9" ./run_until_complete.sh
set -u
cd "$(dirname "$0")"
export PATH=$HOME/anaconda3/envs/examol/bin:$PATH
mkdir -p logs
SIZES="${SIZES:-16 32 64 128 256}"
SEEDS="${SEEDS:-5 6 7 8 9}"
MAX_PASSES="${MAX_PASSES:-3}"
export POOL=dev-snapshot NUM_TO_RUN=500 NUM_INITIAL=32 RETRAIN_EVERY=32 INFER_EVERY=8 \
       WORKERS=8 ENSEMBLE=4 EPOCHS=32 TIME_SCALE=0.05

complete() {  # $1 = run directory; complete when ~all molecules finished
    python - "$1" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'simulation-results.json'
n = sum(1 for line in p.open() if '"status": "finished"' in line) if p.exists() else 0
sys.exit(0 if n >= 495 else 1)
PY
}

for pass_no in $(seq 1 "$MAX_PASSES"); do
    remaining=0
    for size in $SIZES; do
        for seed in $SEEDS; do
            dir="runs/dev-snapshot_incremental_n500_k32_w8_ts0.05_r${size}_s${seed}"
            if complete "$dir"; then continue; fi
            remaining=$((remaining + 1))
            echo "$(date '+%F %T') pass $pass_no: running replay=$size seed=$seed"
            timeout "${RUN_TIMEOUT:-2700}" env MODE=incremental SEED=$seed REPLAY=$size \
                taskset -c "${CPUS:-5-9,15-19}" examol run spec.py:spec \
                > "logs/$(date +%s)_r${size}_s${seed}.log" 2>&1
            status=$?
            if complete "$dir"; then
                echo "$(date '+%F %T') OK replay=$size seed=$seed"
            else
                echo "$(date '+%F %T') FAILED replay=$size seed=$seed (exit $status) - will retry next pass"
            fi
        done
    done
    echo "$(date '+%F %T') pass $pass_no complete; $remaining run(s) attempted"
    incomplete=0
    for size in $SIZES; do for seed in $SEEDS; do
        complete "runs/dev-snapshot_incremental_n500_k32_w8_ts0.05_r${size}_s${seed}" || incomplete=$((incomplete + 1))
    done; done
    if [ "$incomplete" -eq 0 ]; then echo "$(date '+%F %T') ALL COMPLETE"; exit 0; fi
    echo "$(date '+%F %T') $incomplete still incomplete; starting another pass"
done
echo "$(date '+%F %T') giving up after $MAX_PASSES passes"
