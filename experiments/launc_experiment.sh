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
    
    # Create a sanitized session name based on the folder name (tmux rejects periods and spaces)
    SESSION_NAME=$(basename "$CONFIG_DIR" | tr -cd 'A-Za-z0-9_-')
    
    echo "Launching experiment in tmux session: $SESSION_NAME"
    
    # Start a detached tmux session
    tmux new-session -d -s "$SESSION_NAME"
    
    # Send keys to set the environment variable, navigate to the config dir, activate conda, and run
    tmux send-keys -t "$SESSION_NAME" "export EXAMOL_CONFIG_PATH='$CONFIG_FILE'" C-m
    tmux send-keys -t "$SESSION_NAME" "cd '$CONFIG_DIR'" C-m
    tmux send-keys -t "$SESSION_NAME" "conda activate examol" C-m
    tmux send-keys -t "$SESSION_NAME" "examol run ~/ExaMol/examples/redoxmers/spec.py:spec" C-m

done

echo ""
echo "All experiments launched successfully!"
echo "Use 'tmux ls' to view active sessions, and 'tmux attach -t <session_name>' to view logs."