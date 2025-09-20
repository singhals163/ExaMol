
export JOBNAME=parsl.HighThroughputExecutor.block-0.1758331596.5099158
set -e
export CORES=$(getconf _NPROCESSORS_ONLN)
[[ "1" == "1" ]] && echo "Found cores : $CORES"
WORKERCOUNT=1
FAILONANY=0
PIDS=""

CMD() {
process_worker_pool.py  --max_workers_per_node=1 -a 127.0.0.1,130.127.133.71 -p 0 -c 1.0 -m None --poll 10 --port=54491 --cert_dir None --logdir=/users/vthurime/experiment_overall_profiling/experiment_results/training_2/reading_1/parsl-logs/000/HighThroughputExecutor --block_id=0 --hb_period=30  --hb_threshold=120 --drain_period=None --cpu-affinity none  --mpi-launcher=mpiexec --available-accelerators 
}
for COUNT in $(seq 1 1 $WORKERCOUNT); do
    [[ "1" == "1" ]] && echo "Launching worker: $COUNT"
    CMD $COUNT &
    PIDS="$PIDS $!"
done

ALLFAILED=1
ANYFAILED=0
for PID in $PIDS ; do
    wait $PID
    if [ "$?" != "0" ]; then
        ANYFAILED=1
    else
        ALLFAILED=0
    fi
done

[[ "1" == "1" ]] && echo "All workers done"
if [ "$FAILONANY" == "1" ]; then
    exit $ANYFAILED
else
    exit $ALLFAILED
fi
