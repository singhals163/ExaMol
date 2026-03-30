import os
import sys
import json
import time
import argparse
import subprocess
import traceback
import pandas as pd
import numpy as np

def parse_database(run_dir):
    """Extracts SMILES and energies directly from database.json"""
    db_path = os.path.join(run_dir, "database.json")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Cannot find {db_path}")
        
    smiles_list = []
    energies = []
    
    with open(db_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                record = json.loads(line)
                
                smiles = record.get("identifier", {}).get("smiles")
                
                # Navigate the nested dictionary as seen in your database.json
                props = record.get("properties", {})
                energy = None
                if "oxidation_potential" in props:
                    ox_pot = props["oxidation_potential"]
                    if "mopac_pm7-acn-adiabatic" in ox_pot:
                        energy = ox_pot["mopac_pm7-acn-adiabatic"]
                        
                if smiles and energy is not None:
                    smiles_list.append(smiles)
                    energies.append(energy)
                    
            except json.JSONDecodeError:
                continue

    print(f"Successfully loaded {len(smiles_list)} valid molecules from database.json")
    return smiles_list, energies

def get_core_list(case, iteration):
    """
    iteration: 0 to 9 (for 10 to 100 molecules)
    NUMA 0: 0-31, 64-95
    NUMA 1: 32-63, 96-127
    """
    if case == 1:
        # Same SMP cores
        return "0,1,2"
    elif case == 2:
        # Walking SMP cores on NUMA 0
        start = iteration * 3
        return f"{start},{start+1},{start+2}"
    elif case == 3:
        # Alternating NUMA nodes
        if iteration % 2 == 0:
            start = (iteration // 2) * 3
        else:
            start = 32 + ((iteration // 2) * 3)
        return f"{start},{start+1},{start+2}"
    else:
        raise ValueError("Invalid case number")

def run_worker(smiles_file, energy_file, size, run_dir):
    """The isolated worker process that trains the model EXACTLY as ExaMol does"""
    try:
        # 1. Read the slice of data
        with open(smiles_file, 'r') as f:
            smiles = [line.strip() for line in f.readlines()][:size]
        with open(energy_file, 'r') as f:
            energies = np.array([float(line.strip()) for line in f.readlines()][:size])
            
        # 2. Import ExaMol classes
        from examol.score.rdkit import make_knn_model, RDKitScorer
        
        # 3. Initialize model and scorer using EXACT ExaMol defaults.
        # MUST pass run_dir so ExaMol's internal SimpleProfiler doesn't crash on NoneType!
        pipeline = make_knn_model()
        scorer = RDKitScorer(run_dir=os.getcwd())
        
        # --- CRITICAL TIMING BLOCK ---
        t_start = time.perf_counter()
        
        # 4. Trigger the exact retrain function
        scorer.retrain(pipeline, smiles, energies)
        
        t_end = time.perf_counter()
        # -----------------------------
        
        pure_retrain_time = t_end - t_start
        print(f"DEBUG_METRIC;pure_retrain_time;{pure_retrain_time}")
        
    except Exception as e:
        print(f"WORKER_ERROR;{str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

def run_orchestrator(run_dir):
    """Manages the experiment, CPU pinning, and perf stat collection"""
    smiles, energies = parse_database(run_dir)
    
    if len(smiles) < 100:
        print(f"Warning: Found {len(smiles)} molecules. Stopping loops early if needed.")

    smiles_file = os.path.join(run_dir, "tmp_smiles.txt")
    energy_file = os.path.join(run_dir, "tmp_energies.txt")
    with open(smiles_file, 'w') as f:
        f.write('\n'.join(smiles))
    with open(energy_file, 'w') as f:
        f.write('\n'.join(map(str, energies)))

    results = []
    events = "task-clock,cycles,instructions,L1-dcache-load-misses,LLC-loads,LLC-load-misses,minor-faults"

    try:
        for case in [1, 2, 3]:
            print(f"\n=======================================================")
            print(f" Starting Case {case} (10 to 100 molecules)")
            print(f"=======================================================")
            
            for iteration, size in enumerate(range(10, 101, 10)):
                if size > len(smiles):
                    break 
                    
                cores = get_core_list(case, iteration)
                
                cmd = [
                    "sudo", "perf", "stat", "-x", ";", "-e", events,
                    "taskset", "-c", cores,
                    sys.executable, __file__, "--worker", 
                    "--smiles", smiles_file, "--energies", energy_file, "--size", str(size),
                    "--run_dir", run_dir  # Passing run_dir down to the worker
                ]
                
                process = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
                
                metrics = {
                    'Case': case, 
                    'Molecules': size, 
                    'Cores': cores
                }
                
                # Check for worker errors
                for line in process.stderr.split('\n'):
                    if line.startswith("WORKER_ERROR;"):
                        print(f"  [!] Worker crashed: {line.split(';', 1)[1]}")
                
                # Get the internal timer
                for line in process.stdout.split('\n'):
                    if line.startswith("DEBUG_METRIC;"):
                        parts = line.split(';')
                        metrics[parts[1]] = float(parts[2])

                # Get hardware counters
                for line in process.stderr.split('\n'):
                    parts = line.split(';')
                    if len(parts) >= 3:
                        val = parts[0].strip()
                        event = parts[2].strip()
                        if val != '<not supported>':
                            try:
                                metrics[event] = float(val)
                            except ValueError:
                                pass
                                
                results.append(metrics)
                fit_time = metrics.get('pure_retrain_time', 0)
                print(f"Case {case} | Mols: {size:<3} | Cores: {cores:<8} | ExaMol Retrain Time: {fit_time:.4f} sec")
                
    finally:
        if os.path.exists(smiles_file): os.remove(smiles_file)
        if os.path.exists(energy_file): os.remove(energy_file)
        
        df = pd.DataFrame(results)
        out_csv = os.path.join(run_dir, "examol_affinity_profiling.csv")
        df.to_csv(out_csv, index=False)
        print(f"\nProfiling complete! Data saved to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, help="Path to ExaMol run directory")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--smiles", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--energies", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--size", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        run_worker(args.smiles, args.energies, args.size, args.run_dir)
    else:
        if not args.run_dir:
            print("Error: --run_dir required")
            sys.exit(1)
        run_orchestrator(args.run_dir)