import os
import json

# The values you want to sweep across
loop_values = [-1, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

# Base template (without the affinities, as we will inject them dynamically)
base_config = {
    "num_random": 10,
    "num_total": 200,
    "train_freq": 10,
    "infer_freq": 10,
    "train_policy": "linear",
    "max_loops": -1,
    "num_workers": 5,
    "learning_workers": 1
}

# Master directory to hold all the experiment folders
master_dir = "active_learning_sweep"
os.makedirs(master_dir, exist_ok=True)

for i, val in enumerate(loop_values):
    # Map -1 to "infinite", otherwise use the number string
    folder_name = "infinite" if val == -1 else str(val)
    folder_path = os.path.join(master_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Calculate the starting core for this specific run (shifts by 8 each time)
    start_core = i * 8
    
    # 1. Generate Simulation Affinity (First 5 cores: e.g., "list:8:9:10:11:12")
    sim_cores = [str(start_core + j) for j in range(5)]
    sim_affinity_str = "list:" + ":".join(sim_cores)
    
    # 2. Generate Training Affinity (Next 3 cores: e.g., "list:13-15")
    train_start = start_core + 5
    train_end = start_core + 7
    train_affinity_str = f"list:{train_start}-{train_end}"
    
    # Copy the base config and apply our specific values
    current_config = base_config.copy()
    current_config["max_training_loops"] = val
    current_config["sim_affinity"] = sim_affinity_str
    current_config["train_score_affinity"] = train_affinity_str
    
    # Save the config file
    config_file_path = os.path.join(folder_path, "run_config.json")
    with open(config_file_path, "w") as f:
        json.dump(current_config, f, indent=4)
        
    print(f"Generated: {folder_name:>8} | Sim Cores: {sim_affinity_str:<20} | Train Cores: {train_affinity_str}")

print("\nAll configurations successfully generated with unique core affinities!")