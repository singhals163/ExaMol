#!/bin/bash
# Sweep the replay-buffer size, from incremental (replay=0) to warm-started full retraining (replay=all).
# Waits for any campaign already running so the two batches do not share CPU.
set -u
cd "$(dirname "$0")"
while pgrep -f 'examol run spec.py' > /dev/null; do sleep 60; done
export POOL=dev-snapshot NUM_TO_RUN=500 NUM_INITIAL=32 RETRAIN_EVERY=32 INFER_EVERY=8 \
       WORKERS=8 ENSEMBLE=4 EPOCHS=32 TIME_SCALE=0.05
ARGS=()
for size in 16 64 128 256; do
    for seed in 0 1 2 3 4; do
        ARGS+=("MODE=incremental SEED=$seed REPLAY=$size")
    done
done
exec ./run_campaigns.sh "${ARGS[@]}"
