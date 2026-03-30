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

def calculate_training_trends(df):
    """Calculates batch metrics, single molecule metrics, and cumulative times for the first 10 loops."""
    train_df = df[df['task_type'] == 'training_retrain'].copy().sort_values('start_time')
    sim_df = df[df['task_type'].str.contains('mopac_pm7')].copy().sort_values('end_time')
    
    if train_df.empty:
        print("No 'training_retrain' tasks found in the log.")
        sys.exit(1)
        
    # Restrict to the first 10 training events
    train_df = train_df.head(10)
    
    trend_records = []
    prev_train_start = 0
    cumulative_mols = 0
    cumulative_sim_time = 0
    cumulative_train_time = 0
    
    for i, train_row in train_df.iterrows():
        train_start = train_row['start_time']
        train_duration = train_row['duration']
        
        # Isolate the simulation tasks that finished in the window BEFORE this training run
        window_sims = sim_df[(sim_df['end_time'] > prev_train_start) & (sim_df['end_time'] <= train_start)]
        
        batch_sim_time = window_sims['duration'].sum()
        compute_energy_count = len(window_sims[window_sims['task_type'].str.contains('compute_energy')])
        mols_in_batch = compute_energy_count // 2
        
        avg_mol_sim_time = batch_sim_time / mols_in_batch if mols_in_batch > 0 else 0
        
        cumulative_mols += mols_in_batch
        cumulative_sim_time += batch_sim_time
        cumulative_train_time += train_duration
        
        trend_records.append({
            'loop_idx': len(trend_records) + 1,
            'molecules_completed': cumulative_mols,
            'training_duration': train_duration,
            'batch_simulation_time': batch_sim_time,
            'avg_molecule_sim_time': avg_mol_sim_time,
            'cumulative_training_time': cumulative_train_time,
            'cumulative_simulation_time': cumulative_sim_time
        })
        
        prev_train_start = train_start
        
    return pd.DataFrame(trend_records)

def calculate_r2(y_true, y_pred):
    """Calculates the R-squared goodness-of-fit metric."""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

def plot_trends(trend_df, output_dir):
    """Generates the 3 requested comparison plots, including curve fitting on Graph 3."""
    print("Generating plots...")
    x_data = trend_df['molecules_completed'].values
    
    # ---------------------------------------------------------
    # Graph 1: Training Event Time vs. Single Molecule Simulation Time
    # ---------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(x_data, trend_df['training_duration'], marker='X', color='#d62728', 
             linestyle='-', linewidth=2.5, markersize=10, label='Training Task Duration')
    ax1.plot(x_data, trend_df['avg_molecule_sim_time'], marker='o', color='#1f77b4', 
             linestyle='-', linewidth=2.5, markersize=8, label='Batch Avg Simulation Duration')
    
    # ax1.set_title('Training vs Single Molecule Simulation Time', pad=15)
    ax1.set_xlabel('Total Molecules Simulated')
    ax1.set_ylabel('Duration (Seconds)')
    ax1.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trend_01_training_vs_single_sim.png'))
    plt.close(fig1)

    # ---------------------------------------------------------
    # Graph 2: Training Event Time vs. Batch Simulation Time
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(x_data, trend_df['training_duration'], marker='X', color='#d62728', 
             linestyle='-', linewidth=2.5, markersize=10, label='Training Task Duration')
    ax2.plot(x_data, trend_df['batch_simulation_time'], marker='s', color='#2ca02c', 
             linestyle='-', linewidth=2.5, markersize=8, label='Batch Simulation Time')
    
    ax2.set_title('Training vs Batch Simulation Time (First 10 Loops)', pad=15)
    ax2.set_xlabel('Total Molecules Simulated')
    ax2.set_ylabel('Duration (Seconds)')
    ax2.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trend_02_training_vs_batch_sim.png'))
    plt.close(fig2)

    # ---------------------------------------------------------
    # Graph 3: Cumulative Training Time vs. Cumulative Simulation Time (With Curve Fitting)
    # ---------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    y_train = trend_df['cumulative_training_time'].values
    y_sim = trend_df['cumulative_simulation_time'].values
    
    # Generate smooth x values for the continuous curves
    x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
    
    # Fit Linear Line for Simulation: y = mx + c
    sim_coeffs = np.polyfit(x_data, y_sim, 1)
    sim_poly = np.poly1d(sim_coeffs)
    sim_r2 = calculate_r2(y_sim, sim_poly(x_data))
    
    # Fit Quadratic Polynomial for Training: y = ax^2 + bx + c
    train_coeffs = np.polyfit(x_data, y_train, 2)
    train_poly = np.poly1d(train_coeffs)
    train_r2 = calculate_r2(y_train, train_poly(x_data))

    # Plot original scatter data points (no lines, just markers)
    ax3.plot(x_data, y_train, marker='X', color='#d62728', linestyle='', markersize=10, label='Cumulative Training Time')
    ax3.plot(x_data, y_sim, marker='o', color='#1f77b4', linestyle='', markersize=8, label='Cumulative Simulation Time')
    
    # Plot the fitted curves
    ax3.plot(x_smooth, train_poly(x_smooth), color='#d62728', linestyle='--', linewidth=2.5, label='Training Fit')
    ax3.plot(x_smooth, sim_poly(x_smooth), color='#1f77b4', linestyle='--', linewidth=2.5, label='Simulation Fit')
    
    # Create the text block with equations and R^2 values
    eq_text = (
        "Fit Statistics:\n"
        f"Training: $y = {train_coeffs[0]:.4f}x^2 {train_coeffs[1]:.2f}x + {train_coeffs[2]:.2f}$ ($R^2 = {train_r2:.3f}$)\n"
        f"Simulation: $y = {sim_coeffs[0]:.2f}x {sim_coeffs[1]:.2f}$ ($R^2 = {sim_r2:.3f}$)"
    )
    
    # Anchor the text box to the top left of the graph, slightly shifted right of the y-axis
    ax3.text(0.05, 0.85, eq_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='black', alpha=0.9))

    # ax3.set_title('Training v/s Simulation Cumulative Time Comparison (ExaMol)', pad=15)
    ax3.set_xlabel('Total Molecules Simulated')
    ax3.set_ylabel('Cumulative Duration (Seconds)')
    
    # Move legend to a spot where it won't obscure the text box
    ax3.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trend_03_cumulative_scaling.png'))
    plt.close(fig3)
    
    print("Successfully saved 3 trend plots to the log directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze training vs simulation times (First 10 loops).")
    parser.add_argument("log_file", type=str, help="Path to profile_stats.jsonl file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"Error: Could not find file {args.log_file}")
        sys.exit(1)
        
    output_dir = os.path.dirname(os.path.abspath(args.log_file))
    
    df = parse_profile_stats(args.log_file)
    trend_df = calculate_training_trends(df)
    plot_trends(trend_df, output_dir)