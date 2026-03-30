import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration & Paper-Grade Setup
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.5,
    'grid.linestyle': '--'
})

def parse_profile_stats(filepath):
    """Parses profile_stats.jsonl into a structured DataFrame."""
    print(f"Parsing {filepath}...")
    records = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip(): 
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            task_type = data.get('task_type', 'unknown')
            start_time = data.get('task_start_time', 0)
            end_time = data.get('timestamp', 0) 
            
            duration = 0
            if 'metrics' in data:
                for k, v in data['metrics'].items():
                    if 'total' in k:
                        duration = v['duration']
                        break
                if duration == 0 and len(data['metrics']) > 0:
                    start_ts = min([v['start_timestamp'] for v in data['metrics'].values()])
                    end_ts = max([v['start_timestamp'] + v['duration'] for v in data['metrics'].values()])
                    duration = end_ts - start_ts
            
            records.append({
                'task_type': task_type,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration
            })
            
    df = pd.DataFrame(records)
    if df.empty:
        print("Error: No valid records found in the provided file.")
        sys.exit(1)
        
    df = df.sort_values(by='start_time').reset_index(drop=True)
    return df

def calculate_decoupled_trends(df):
    """
    Separates Training and Simulation tracks. 
    Simulations are binned strictly in 10s.
    Training is mapped to whenever it triggered dynamically.
    """
    train_df = df[df['task_type'] == 'training_retrain'].copy().sort_values('start_time')
    sim_df = df[df['task_type'].str.contains('mopac_pm7')].copy().sort_values('end_time')
    
    if train_df.empty:
        print("No 'training_retrain' tasks found in the log.")
        sys.exit(1)
        
    # Restrict to first 10 training events (Geometric scaling)
    train_df = train_df.head(10)
    
    # 1. Establish the "Molecule Completion" Timeline
    # Since 1 molecule = 2 compute_energy events, every 2nd end_time marks a completion
    compute_energies = sim_df[sim_df['task_type'].str.contains('compute_energy')]
    mol_end_times = compute_energies['end_time'].values[1::2] # indices 1, 3, 5...
    
    # ==========================================
    # 2. Build Training Stats
    # ==========================================
    train_records = []
    cum_train_time = 0
    for _, row in train_df.iterrows():
        t_start = row['start_time']
        t_dur = row['duration']
        # How many molecules finished BEFORE this training triggered?
        mols_completed = np.sum(mol_end_times <= t_start)
        
        cum_train_time += t_dur
        train_records.append({
            'molecules_completed': mols_completed,
            'training_duration': t_dur,
            'cumulative_training_time': cum_train_time
        })
    train_stats = pd.DataFrame(train_records)
    
    # Extract the maximum X-axis bound to cap the simulation bins
    max_mols_evaluated = train_stats['molecules_completed'].max()
    if max_mols_evaluated == 0:
        max_mols_evaluated = len(mol_end_times) # Fallback if training triggered instantly
        
    # ==========================================
    # 3. Build Simulation Stats (Batches of 10)
    # ==========================================
    sim_records = []
    batch_size = 5
    cum_sim_time = 0
    prev_time = 0
    
    # Iterate over the timeline in chunks of 10
    for i in range(batch_size - 1, len(mol_end_times), batch_size):
        mols_completed = i + 1
        end_time = mol_end_times[i]
        
        if mols_completed > max_mols_evaluated + batch_size:
            break # Stop processing bins far beyond the 10th training loop
            
        # Get all simulation pieces (optimize + compute_energy) in this time window
        window_sims = sim_df[(sim_df['end_time'] > prev_time) & (sim_df['end_time'] <= end_time)]
        batch_sim_time = window_sims['duration'].sum()
        
        cum_sim_time += batch_sim_time
        avg_mol_time = batch_sim_time / batch_size
        
        sim_records.append({
            'molecules_completed': mols_completed,
            'batch_simulation_time': batch_sim_time,
            'avg_molecule_sim_time': avg_mol_time,
            'cumulative_simulation_time': cum_sim_time
        })
        prev_time = end_time
        
    sim_stats = pd.DataFrame(sim_records)
    
    return train_stats, sim_stats

def calculate_r2(y_true, y_pred):
    """Calculates the R-squared goodness-of-fit metric."""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

def plot_trends(train_stats, sim_stats, output_dir):
    """Generates the decoupled comparison plots with fitted curves."""
    print("Generating plots...")
    
    x_train = train_stats['molecules_completed'].values
    x_sim = sim_stats['molecules_completed'].values
    
    max_x = max(x_train.max() if len(x_train)>0 else 0, x_sim.max() if len(x_sim)>0 else 0)
    
    # ---------------------------------------------------------
    # Graph 1: Training Event Time vs. Single Molecule Simulation Time
    # ---------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(x_train, train_stats['training_duration'], marker='X', color='#d62728', 
             linestyle='-', linewidth=2.5, markersize=10, label='Training Task Duration')
    ax1.plot(x_sim, sim_stats['avg_molecule_sim_time'], marker='o', color='#1f77b4', 
             linestyle='-', linewidth=2.5, markersize=8, label='Avg Single Molecule Sim Time')
    
    ax1.set_xlabel('Total Molecules Simulated')
    ax1.set_ylabel('Duration (Seconds)')
    ax1.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trend_01_training_vs_single_sim.png'))
    plt.close(fig1)

    # ---------------------------------------------------------
    # Graph 2: Training Event Time vs. Batch (10) Simulation Time
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(x_train, train_stats['training_duration'], marker='X', color='#d62728', 
             linestyle='-', linewidth=2.5, markersize=10, label='Training Task Duration')
    ax2.plot(x_sim, sim_stats['batch_simulation_time'], marker='s', color='#2ca02c', 
             linestyle='-', linewidth=2.5, markersize=8, label='Batch Simulation Time (10 Mols)')
    
    ax2.set_title('Training vs Batch Simulation Time (Geometric Scaling)', pad=15)
    ax2.set_xlabel('Total Molecules Simulated')
    ax2.set_ylabel('Duration (Seconds)')
    ax2.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trend_02_training_vs_batch_sim.png'))
    plt.close(fig2)

    # ---------------------------------------------------------
    # Graph 3: Cumulative Training Time vs. Cumulative Simulation Time (Curve Fitting)
    # ---------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    y_train = train_stats['cumulative_training_time'].values
    y_sim = sim_stats['cumulative_simulation_time'].values
    
    # Generate smooth curves
    x_smooth_train = np.linspace(min(x_train), max(x_train), 200) if len(x_train)>1 else x_train
    x_smooth_sim = np.linspace(min(x_sim), max(x_sim), 200) if len(x_sim)>1 else x_sim
    
    # Fit Linear Line for Simulation: y = mx + c
    if len(x_sim) > 1:
        sim_coeffs = np.polyfit(x_sim, y_sim, 1)
        sim_poly = np.poly1d(sim_coeffs)
        sim_r2 = calculate_r2(y_sim, sim_poly(x_sim))
        ax3.plot(x_smooth_sim, sim_poly(x_smooth_sim), color='#1f77b4', linestyle='--', linewidth=2.5, label='Simulation Fit (Linear)')
    else:
        sim_coeffs = [0, 0]
        sim_r2 = 0

    # Fit Quadratic Polynomial for Training: y = mx + c
    if len(x_train) > 2:
        train_coeffs = np.polyfit(x_train, y_train, 1)
        train_poly = np.poly1d(train_coeffs)
        train_r2 = calculate_r2(y_train, train_poly(x_train))
        ax3.plot(x_smooth_train, train_poly(x_smooth_train), color='#d62728', linestyle='--', linewidth=2.5, label='Training Fit (Linear)')
    else:
        train_coeffs = [0, 0]
        train_r2 = 0

    # Plot original scatter data points
    ax3.plot(x_train, y_train, marker='X', color='#d62728', linestyle='', markersize=10, label='Cumulative Training Time')
    ax3.plot(x_sim, y_sim, marker='o', color='#1f77b4', linestyle='', markersize=8, label='Cumulative Simulation Time')
    
    # Text block with equations (using :+ formatting to handle negative signs cleanly)
    eq_text = (
        "Fit Statistics:\n"
        f"Training: $y = {train_coeffs[0]:.2f}x {train_coeffs[1]:+.2f}$ ($R^2 = {train_r2:.3f}$)\n"
        f"Simulation: $y = {sim_coeffs[0]:.2f}x {sim_coeffs[1]:+.2f}$ ($R^2 = {sim_r2:.3f}$)"
    )
    
    ax3.text(0.05, 0.85, eq_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='black', alpha=0.9))

    ax3.set_xlabel('Total Molecules Simulated')
    ax3.set_ylabel('Cumulative Duration (Seconds)')
    ax3.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trend_03_cumulative_scaling.png'))
    plt.close(fig3)
    
    print("Successfully saved 3 trend plots to the log directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze geometric training vs batch simulation times.")
    parser.add_argument("log_file", type=str, help="Path to profile_stats.jsonl file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"Error: Could not find file {args.log_file}")
        sys.exit(1)
        
    output_dir = os.path.dirname(os.path.abspath(args.log_file))
    
    df = parse_profile_stats(args.log_file)
    train_stats, sim_stats = calculate_decoupled_trends(df)
    plot_trends(train_stats, sim_stats, output_dir)