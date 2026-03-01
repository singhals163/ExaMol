from .version import __version__  # noqa: 401

import time
import json
import fcntl
from pathlib import Path

class SimpleProfiler:
    def __init__(self, log_file: Path, name="unknown_task"):
        """
        name: Identifier for this run (e.g., 'training_run_1', 'sim_run_5')
        log_file: The common file path to write results to.
        """
        self.stats = {}
        self.name = name
        self.task_start_time = time.time()  # Captures when the overall task started
        
        # create a file in log_file by the name "profile_stats.jsonl" if it does not exist, otherwise use it
        self.log_file = Path(log_file) / "profile_stats.jsonl"

    def __call__(self, label):
        parent = self
        class Timer:
            def __enter__(self):
                self.start_perf = time.perf_counter()
                self.start_timestamp = time.time()  # Capture absolute start time
                
            def __exit__(self, *args):
                elapsed = time.perf_counter() - self.start_perf
                # Store both the start timestamp and the precise duration
                parent.stats[label] = {
                    "start_timestamp": self.start_timestamp,
                    "duration": elapsed
                }
        return Timer()

    def save(self):
        """Writes the collected stats to the common log file safely."""
        if not self.stats:
            return

        # Structure the log entry
        data = {
            "timestamp": time.time(),          # Time when the log was saved/task finished
            "task_start_time": self.task_start_time,
            "task_type": self.name,
            "metrics": self.stats
        }

        # Safe append with locking
        try:
            with open(self.log_file, "a") as f:
                # Acquire an exclusive lock to prevent write collisions from other workers
                fcntl.flock(f, fcntl.LOCK_EX) 
                f.write(json.dumps(data) + "\n")
                # Unlock happens automatically when file is closed, but good practice to release
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            print(f"Profiler failed to write to {self.log_file}: {e}")

    def __del__(self):
        """Automatically save when the profiler instance is destroyed."""
        self.save()