#!/bin/bash

# --- Experiment Configuration ---
TRAINING_LOOPS=(0 1 2 3 4 5 6 "inf")
NUM_READINGS=5
BASE_RESULTS_DIR="experiment_results"

# --- Script Logic ---
echo "Starting ExaMol experiment suite in PARALLEL mode..."
# Create the top-level directory for all results
mkdir -p "$BASE_RESULTS_DIR"

# Loop through each training configuration
for loops in "${TRAINING_LOOPS[@]}"; do
    TRAINING_DIR="$BASE_RESULTS_DIR/training_${loops}"
    mkdir -p "$TRAINING_DIR"
    echo "Setting up runs for ${loops} training loops..."

    # Loop for each repeated reading
    for reading in $(seq 0 $((NUM_READINGS - 1))); do
        # Define the final destination for this run's results
        RESULTS_DIR="$TRAINING_DIR/reading_${reading}"
        
        # Define the main run directory for this specific job
        # This is the path that will be passed to spec.py
        RUN_DIR_ABSOLUTE=$(realpath "$RESULTS_DIR")
        mkdir -p "$RUN_DIR_ABSOLUTE"

        echo "--> Launching run: training_${loops}/reading_${reading}"

        # Group the commands and run them in the background with '&'
        {
            # Export environment variables for the upcoming run
            export EXAMOL_RUN_DIR="$RUN_DIR_ABSOLUTE"
            export EXAMOL_MAX_LOOPS="$loops"

            # Execute the ExaMol run
            examol run ExaMol/examples/redoxmers/spec.py:spec

            # # Check if the run was successful before processing results
            # if [ $? -eq 0 ]; then
            #     echo "    Run successful for training_${loops}/reading_${reading}."
            #     # Note: Results are already in the correct directory, so no copying is needed.
            #     # You could add cleanup steps here if desired.
            # else
            #     echo "    ERROR: Run failed for training_${loops}/reading_${reading}. Check logs in $RUN_DIR_ABSOLUTE"
            # fi
        } &
    done
done

# Wait for all background jobs to complete
echo ""
echo "All experiment runs have been launched. Waiting for completion..."
wait
echo "All experiments have completed. Results are in the '$BASE_RESULTS_DIR' directory."