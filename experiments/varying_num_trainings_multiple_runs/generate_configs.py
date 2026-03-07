import os
import json

def generate_experiments():
    base_dir = "batch_experiments"
    total_experiments = 10
    total_cores = 64
    
    # Create the base directory
    os.makedirs(base_dir, exist_ok=True)

    cores_allocated = 0
    
    for i in range(total_experiments):
        # Distribute the 64 cores (4 experiments get 7 cores, 6 get 6 cores)
        num_cores = (total_cores // total_experiments) + (1 if i < (total_cores % total_experiments) else 0)
        
        # Slice the specific CPU IDs for this experiment
        experiment_cores = list(range(cores_allocated, cores_allocated + num_cores))
        
        # Format the sim_affinity string as "list:C1:C2:C3..." (multiple workers, 1 core each)
        sim_affinity_str = "list:" + ":".join(map(str, experiment_cores))
        
        # Format the train_score_affinity string as "list:start-end" (1 worker, multiple cores)
        start_core = cores_allocated
        end_core = cores_allocated + num_cores - 1
        train_score_affinity_str = f"list:{start_core}-{end_core}"
        
        # Increment tracker for the next experiment
        cores_allocated += num_cores
        
        # Vary max_training_loops from 0 to 18 (2*i)
        max_training_loops = 2*i
        
        # Build the configuration dictionary
        config = {
            "num_random": 10,
            "num_total": 300,
            "max_training_loops": max_training_loops,
            "train_freq": 20,
            "infer_freq": 10,
            "sim_affinity": sim_affinity_str,
            "train_score_affinity": train_score_affinity_str,
            "num_workers": num_cores,
            "learning_workers": 1
        }
        
        # Create a clearly named subdirectory
        exp_dir = os.path.join(base_dir, f"exp_{i+1:02d}_loops_{max_training_loops}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # Write the JSON file
        config_path = os.path.join(exp_dir, "run_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
            
        print(f"Created {exp_dir} -> sim: ({sim_affinity_str}) | train: ({train_score_affinity_str})")

if __name__ == "__main__":
    generate_experiments()
    print("\nAll 10 experiment configurations generated successfully.")