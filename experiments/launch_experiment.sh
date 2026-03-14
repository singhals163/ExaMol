#!/bin/bash

# Ensure a target directory is provided
if [ -z "$1" ]; then
    echo "Usage: ./launch_experiments.sh <path_to_experiment_subdirectories>"
    exit 1
fi

BASE_DIR=$(realpath "$1")

# Find all run_config.json files and loop through them
find "$BASE_DIR" -type f -name "run_config.json" | while read -r CONFIG_FILE; do
    
    # Get the parent directory of the config file
    CONFIG_DIR=$(dirname "$CONFIG_FILE")
    
    # Create a sanitized session name based on the folder name
    SESSION_NAME=$(basename "$CONFIG_DIR" | tr -cd 'A-Za-z0-9_-')
    
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
    
    # Symlink the search space so spec.py can find it locally without duplicating data
    # tmux send-keys -t "$SESSION_NAME" "  ln -sf \"$CONFIG_DIR/search_space.smi\" \"\$RUN_DIR/search_space.smi\"" C-m
    
    # Navigate to the run directory
    tmux send-keys -t "$SESSION_NAME" "  cd \"\$RUN_DIR\"" C-m
    
    # Set the environment variable pointing to the COPIED config file path
    tmux send-keys -t "$SESSION_NAME" "  export EXAMOL_CONFIG_PATH=\"\$RUN_DIR/run_config.json\"" C-m
    
    # Execute ExaMol
    tmux send-keys -t "$SESSION_NAME" "  examol run ~/ExaMol/examples/redoxmers/spec.py:spec" C-m
    tmux send-keys -t "$SESSION_NAME" "done" C-m
    
    # Print a completion message inside the tmux session once the loop finishes
    tmux send-keys -t "$SESSION_NAME" "echo \"All 5 runs completed for $SESSION_NAME.\"" C-m

done

echo ""
echo "All experiments launched successfully!"
echo "Use 'tmux ls' to view active sessions, and 'tmux attach -t <session_name>' to view logs."