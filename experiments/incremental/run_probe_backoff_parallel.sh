#!/bin/bash
# Run the probe-backoff policy for 10 seeds, at WORKERS=8, on this machine. Two campaigns run at a
# time, each pinned via taskset to a disjoint half of this job's 16 allocated CPUs.
#   Usage: COMBO=full-fresh ./run_probe_backoff_parallel.sh
#          COMBO=checkpoint-recent ./run_probe_backoff_parallel.sh
set -u
cd "$(dirname "$0")"
export PATH=/tmp/examol-env/bin:$PATH
export PYTHONNOUSERSITE=1  # This machine's conda python puts ~/.local site-packages ahead of the env's own
export POOL=dev-snapshot NUM_TO_RUN=500 NUM_INITIAL=32 RETRAIN_EVERY=32 INFER_EVERY=8 \
       WORKERS=8 ENSEMBLE=4 EPOCHS=32 "TIME_SCALE=${TIME_SCALE:-0.05}" POLICY=probe_backoff \
       PROBE_INITIAL=32 PROBE_BACKOFF=1.5 PROBE_MAX=128

COMBO="${COMBO:?Set COMBO=full-fresh or COMBO=checkpoint-recent}"
case "$COMBO" in
    full-fresh)
        export MODE=full LOSS_THRESHOLD=0.18 CONSECUTIVE_TO_FREEZE=2
        ;;
    checkpoint-recent)
        export MODE=incremental REPLAY=0 LOSS_THRESHOLD=0.15 CONSECUTIVE_TO_FREEZE=2
        ;;
    *)
        echo "Unknown COMBO=$COMBO" >&2; exit 1
        ;;
esac

mkdir -p logs
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
CPUS_A="${CPUS_A:-16-23}"
CPUS_B="${CPUS_B:-32-39}"
RUN_TIMEOUT="${RUN_TIMEOUT:-2700}"

seeds=($SEEDS)
for ((i=0; i<${#seeds[@]}; i+=2)); do
    s1=${seeds[i]}
    s2=${seeds[i+1]:-}
    echo "$(date) [$COMBO] starting seed=$s1 on cpus $CPUS_A$( [ -n "$s2" ] && echo " + seed=$s2 on cpus $CPUS_B")"

    timeout "$RUN_TIMEOUT" env SEED="$s1" taskset -c "$CPUS_A" examol run spec.py:spec \
        > "logs/$(date +%s)_pb-${COMBO}_s${s1}.log" 2>&1 &
    pid1=$!

    pid2=""
    if [ -n "$s2" ]; then
        timeout "$RUN_TIMEOUT" env SEED="$s2" taskset -c "$CPUS_B" examol run spec.py:spec \
            > "logs/$(date +%s)_pb-${COMBO}_s${s2}.log" 2>&1 &
        pid2=$!
    fi

    wait "$pid1"; echo "$(date) [$COMBO] finished seed=$s1 exit=$?"
    if [ -n "$pid2" ]; then
        wait "$pid2"; echo "$(date) [$COMBO] finished seed=$s2 exit=$?"
    fi
done
echo "$(date) [$COMBO] ALL DONE"
