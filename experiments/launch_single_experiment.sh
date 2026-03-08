#!/bin/bash

# Ensure a target directory is provided
if [ -z "$1" ]; then
    echo "Usage: ./launch_single_experiment.sh <path_to_specific_experiment_directory>"
    echo "Example: ./launch_single_experiment.sh batch_experiments/exp_01_loops_0"
    exit 1
fi

CONFIG_DIR=$(realpath "$1")
CONFIG_FILE="$CONFIG_DIR/run_config.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: run_config.json not found in $CONFIG_DIR"
    exit 1
fi

# Create a sanitized session name based on the folder name
SESSION_NAME=$(basename "$CONFIG_DIR" | tr -cd 'A-Za-z0-9_-')

# Automatically kill the old stalled session if it exists
echo "Checking for existing tmux session '$SESSION_NAME'..."
tmux kill-session -t "$SESSION_NAME" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "Killed old stalled session: $SESSION_NAME"
fi

echo "Launching 5 sequential runs for experiment in tmux session: $SESSION_NAME"

# Start a detached tmux session
tmux new-session -d -s "$SESSION_NAME"

# Activate the conda environment first
tmux send-keys -t "$SESSION_NAME" "conda activate examol" C-m

# Send a bash loop to the tmux session to execute 5 runs sequentially
tmux send-keys -t "$SESSION_NAME" "for i in {1..5}; do" C-m
tmux send-keys -t "$SESSION_NAME" "  echo \"========================================\"" C-m
tmux send-keys -t "$SESSION_NAME" "  echo \"Starting run \$i of 5 for $SESSION_NAME\"" C-m
tmux send-keys -t "$SESSION_NAME" "  echo \"========================================\"" C-m

# Create the run-specific subdirectory
tmux send-keys -t "$SESSION_NAME" "  RUN_DIR=\"$CONFIG_DIR/run_\$i\"" C-m
tmux send-keys -t "$SESSION_NAME" "  mkdir -p \"\$RUN_DIR\"" C-m

# Copy the config file into the new run directory
tmux send-keys -t "$SESSION_NAME" "  cp \"$CONFIG_FILE\" \"\$RUN_DIR/run_config.json\"" C-m

# Navigate to the run directory
tmux send-keys -t "$SESSION_NAME" "  cd \"\$RUN_DIR\"" C-m

# Set the environment variable pointing to the COPIED config file path
tmux send-keys -t "$SESSION_NAME" "  export EXAMOL_CONFIG_PATH=\"\$RUN_DIR/run_config.json\"" C-m

# Execute ExaMol
tmux send-keys -t "$SESSION_NAME" "  examol run ~/ExaMol/examples/redoxmers/spec.py:spec" C-m
tmux send-keys -t "$SESSION_NAME" "done" C-m

# Print a completion message inside the tmux session once the loop finishes
tmux send-keys -t "$SESSION_NAME" "echo \"All 5 runs completed for $SESSION_NAME.\"" C-m

echo ""
echo "Experiment $SESSION_NAME launched successfully!"
echo "Use 'tmux attach -t $SESSION_NAME' to view logs."