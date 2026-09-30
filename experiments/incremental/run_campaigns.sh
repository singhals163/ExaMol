#!/bin/bash
# Run campaigns one after another (so training in one does not slow another) on a fixed set of CPUs
#   Usage: CPUS=5-9,15-19 ./run_campaigns.sh "MODE=full SEED=0" "MODE=incremental SEED=0" ...
# Other settings (POOL, NUM_TO_RUN, TIME_SCALE, ...) are read from the environment; see spec.py
set -u
cd "$(dirname "$0")"
export PATH=$HOME/anaconda3/envs/examol/bin:$PATH
mkdir -p logs
for settings in "$@"; do
    name=$(echo "$settings" | tr ' =' '_-')
    echo "$(date) starting $settings"
    # ExaMol can hang if a learning task fails, so cap each campaign (default 45 min; ~13 min is typical)
    timeout "${RUN_TIMEOUT:-2700}" env $settings taskset -c "${CPUS:-5-9,15-19}" examol run spec.py:spec > "logs/$(date +%s)_${name}.log" 2>&1
    echo "$(date) finished $settings exit=$?"
done
