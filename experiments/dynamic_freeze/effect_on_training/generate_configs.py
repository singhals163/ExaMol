import os
import json

# Define the 10 strategies
strategies = {
    "baseline_always": [str(i) for i in range(19)],
    "freeze_after_6": [str(i) for i in range(7)],
    "freeze_before_6": [str(i) for i in range(6, 19)],
    "freeze_4_and_8": [str(i) for i in range(19) if i not in [4, 8]],
    "exponential_decay": ["0", "1", "3", "7", "15"],
    "periodic_even": [str(i) for i in range(0, 19, 2)],
    "periodic_sparse": [str(i) for i in range(0, 19, 3)],
    "bookends": ["0", "1", "2", "16", "17", "18"],
    "middle_heavy": ["6", "7", "8", "9", "10", "11", "12"],
    "one_shot": ["0"]
}

# Base template
base_config = {
    "num_random": 10,
    "num_total": 200,
    "train_freq": 10,
    "infer_freq": 10,
    "train_policy": "linear",
    "max_loops": -1,  
    "num_workers": 6,        # 6 individual simulation workers
    "learning_workers": 1    # 1 learning worker
}

# Master directory
master_dir = "dynamic_freezing_experiments"
os.makedirs(master_dir, exist_ok=True)

for i, (strategy_name, cycles) in enumerate(strategies.items()):
    folder_path = os.path.join(master_dir, f"exp_{i:02d}_{strategy_name}")
    os.makedirs(folder_path, exist_ok=True)
    
    # Isolate a block of 6 cores per experiment (0-5, 6-11, 12-17, ..., 54-59)
    start_core = i * 6
    
    # 1. Simulation Affinity: All 6 cores separated by colons (e.g., list:0:1:2:3:4:5)
    sim_cores = [str(start_core + j) for j in range(6)]
    sim_affinity_str = "list:" + ":".join(sim_cores)
    
    # 2. Training Affinity: The LAST 3 cores of the block as a hyphenated range (e.g., list:3-5)
    train_start = start_core + 3
    train_end = start_core + 5
    train_affinity_str = f"list:{train_start}-{train_end}"
    
    # Build config
    current_config = base_config.copy()
    current_config["run_training_cycles"] = cycles
    current_config["sim_affinity"] = sim_affinity_str
    current_config["train_score_affinity"] = train_affinity_str
    
    # Save config
    config_file_path = os.path.join(folder_path, "run_config.json")
    with open(config_file_path, "w") as f:
        json.dump(current_config, f, indent=4)
        
    print(f"Generated: {strategy_name:<18} | Sim: {sim_affinity_str:<20} | Train: {train_affinity_str}")

print("\nAll 10 configurations successfully generated. Ready to launch!")