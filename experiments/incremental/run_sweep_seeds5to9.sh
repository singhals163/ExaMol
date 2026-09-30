#!/bin/bash
# Extend the replay sweep to seeds 5-9, so every replay size has the same 10 seeds as the baselines
set -u
cd "$(dirname "$0")"
while pgrep -f 'examol run spec.py' > /dev/null; do sleep 30; done
export POOL=dev-snapshot NUM_TO_RUN=500 NUM_INITIAL=32 RETRAIN_EVERY=32 INFER_EVERY=8 \
       WORKERS=8 ENSEMBLE=4 EPOCHS=32 TIME_SCALE=0.05
ARGS=()
for size in 16 32 64 128 256; do
    for seed in 5 6 7 8 9; do
        ARGS+=("MODE=incremental SEED=$seed REPLAY=$size")
    done
done
exec ./run_campaigns.sh "${ARGS[@]}"
