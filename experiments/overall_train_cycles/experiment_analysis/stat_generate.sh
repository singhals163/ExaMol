#!/bin/bash

# --- Configuration ---
BASE_RESULTS_DIR="../experiment_results"
ANALYSIS_SCRIPT="collect_stats.py"

# --- Script Logic ---
echo "Starting parallel analysis of all ExaMol run directories..."

# Check if the analysis script exists
if [ ! -f "$ANALYSIS_SCRIPT" ]; then
    echo "Error: Analysis script '$ANALYSIS_SCRIPT' not found."
    exit 1
fi

# Check if the base results directory exists
if [ ! -d "$BASE_RESULTS_DIR" ]; then
    echo "Error: Base results directory '$BASE_RESULTS_DIR' not found."
    exit 1
fi

# Find all the individual 'reading_*' directories and loop through them
# The '-print0' and 'while read -r -d' combo safely handles paths with spaces
find "$BASE_RESULTS_DIR" -type d -name "reading_*" -print0 | while IFS= read -r -d $'\0' dir; do
    dir="${dir}/run"
    echo "--> Launching analysis for: $dir"
    # Run the python script in the background for parallel execution
    python "$ANALYSIS_SCRIPT" "$dir" &
done

# Wait for all background analysis jobs to complete
echo ""
echo "All analysis jobs have been launched. Waiting for completion..."
wait
echo "All analyses are complete. Each run directory now contains an 'analysis_summary.txt' file."
