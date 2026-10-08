#!/bin/bash
# Incremental retraining (replay=0), warm-started from a checkpoint vs. always reinitialized from scratch.
# Same pool, recipe, schedule, and seeds as analysis/baseline-10seeds; only the worker count (16, to use
# this machine's full allocation) and the scorer's weight-init policy differ.
#   Usage: SEEDS="0 1 2 3 4 5 6 7 8 9" ./run_scratch_vs_checkpoint.sh
set -u
cd "$(dirname "$0")"
export PATH=/tmp/examol-env/bin:$PATH
export PYTHONNOUSERSITE=1  # This machine's conda python puts ~/.local site-packages ahead of the env's own
export POOL=dev-snapshot NUM_TO_RUN=500 NUM_INITIAL=32 RETRAIN_EVERY=32 INFER_EVERY=8 \
       WORKERS=16 ENSEMBLE=4 EPOCHS=32 TIME_SCALE=0.05
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
CPUS="${CPUS:-16-23,32-39}"

ARGS=()
for seed in $SEEDS; do
    ARGS+=("MODE=incremental SEED=$seed REPLAY=0")
    ARGS+=("MODE=incremental SEED=$seed REPLAY=0 RETRAIN_FROM_SCRATCH=true")
done
CPUS="$CPUS" ./run_campaigns.sh "${ARGS[@]}"
