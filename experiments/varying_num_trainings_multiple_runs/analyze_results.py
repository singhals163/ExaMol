import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# ==========================================
# 1. Configuration & Paper-Grade Setup
# ==========================================
BASE_DIR = "batch_experiments"
RESULTS_DIR = "analysis_results"
GANTT_DIR = os.path.join(RESULTS_DIR, "gantt_charts")
TIME_EVOL_DIR = os.path.join(RESULTS_DIR, "task_evolution")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GANTT_DIR, exist_ok=True)
os.makedirs(TIME_EVOL_DIR, exist_ok=True)

# High-quality paper-style configurations
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

# Distinct color and marker maps for the 4 separated tasks
COLOR_MAP = {
    'Optimize Structure': '#1f77b4', # Strong Blue
    'Compute Energy': '#17becf',     # Cyan/Light Blue
    'ML Inference': '#2ca02c',       # Green
    'Training': '#d62728'            # Crimson Red
}

MARKER_MAP = {
    'Optimize Structure': 'o',       # Dot
    'Compute Energy': 's',           # Square
    'ML Inference': '^',             # Triangle
    'Training': 'X'                  # Cross
}

def categorize_task(task_type):
    if 'retrain' in task_type:
        return 'Training'
    elif 'score' in task_type or 'inference' in task_type:
        return 'ML Inference'
    elif 'optimize' in task_type:
        return 'Optimize Structure'
    elif 'energy' in task_type:
        return 'Compute Energy'
    return 'Other'

# ==========================================
# 2. Data Parsing
# ==========================================
def parse_experiment_data():
    """Parses profile_stats.jsonl to get hardware task timings."""
    print(f"Scanning for profile_stats.jsonl in {BASE_DIR}...")
    records = []
    
    search_pattern = os.path.join(BASE_DIR, "*", "*", "*", "profile_stats.jsonl")
    file_paths = glob.glob(search_pattern)
    
    if not file_paths:
        print(f"Warning: No profile_stats.jsonl files found in {search_pattern}")
        return pd.DataFrame()
        
    for filepath in file_paths:
        parts = filepath.split(os.sep)
        exp_name = parts[-4]
        run_name = parts[-3]
        
        with open(filepath, 'r') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                task_type = data.get('task_type', 'unknown')
                start_time = data.get('task_start_time', 0)
                cpus = data.get('cpus', [])
                
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
                
                category = categorize_task(task_type)
                if category == 'Other': continue
                
                records.append({
                    'exp_name': exp_name,
                    'run_name': run_name,
                    'task_type': task_type,
                    'category': category,
                    'start_time': start_time,
                    'duration': duration,
                    'cpus': cpus
                })
                
    df = pd.DataFrame(records)
    
    if not df.empty:
        df['rel_start_time'] = df.groupby(['exp_name', 'run_name'])['start_time'].transform(lambda x: x - x.min())
        
    print(f"Successfully parsed {len(df)} task timing records.")
    return df

def parse_simulation_results():
    """Parses simulation_results.json to extract the final energies of discovered molecules."""
    print(f"Scanning for simulation_results.json in {BASE_DIR}...")
    records = []
    
    search_pattern = os.path.join(BASE_DIR, "**", "simulation-results.json")
    file_paths = glob.glob(search_pattern, recursive=True)
    
    for filepath in file_paths:
        parts = filepath.split(os.sep)
        exp_name = "Unknown"
        for part in parts:
            if part.startswith("exp_"):
                exp_name = part
                break
                
        with open(filepath, 'r') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    
                    if data.get('method') == 'compute_energy':
                        task_info = data.get('task_info', {})
                        
                        if task_info.get('status') == 'finished':
                            result_val = task_info.get('result')
                            
                            if isinstance(result_val, list) and len(result_val) > 0:
                                energy = result_val[0]
                                records.append({
                                    'exp_name': exp_name,
                                    'energy': energy
                                })
                except json.JSONDecodeError:
                    continue
                    
    df = pd.DataFrame(records)
    print(f"Successfully parsed {len(df)} molecule energy records.")
    return df

# ==========================================
# 3. Plotting Functions
# ==========================================
def plot_energy_distribution(energy_df):
    """Plot the normalized frequency distribution of simulated molecule energies (pooled)."""
    print("Plotting energy distribution curves...")
    if energy_df.empty:
        print("No energy data to plot.")
        return
        
    fig, ax = plt.subplots(figsize=(12, 7))
    
    experiments = energy_df['exp_name'].unique()
    for exp in sorted(experiments):
        subset = energy_df[energy_df['exp_name'] == exp]
        sns.kdeplot(data=subset, x='energy', label=f"{exp} (N={len(subset)})", linewidth=2.5, fill=True, alpha=0.1, ax=ax)
        
    ax.set_title('Normalized Frequency Distribution of Discovered Molecule Energies')
    ax.set_xlabel('Simulated Redox Energy')
    ax.set_ylabel('Density (Normalized Frequency)')
    ax.legend(title="Experiment Setup", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'molecule_energy_distribution.png'))
    plt.close()

def plot_gantt_charts(df):
    """Plot Gantt charts mapping task execution to CPUs."""
    print("Generating Gantt charts...")
    grouped = df.groupby(['exp_name', 'run_name'])
    
    for (exp_name, run_name), run_df in grouped:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for _, row in run_df.iterrows():
            color = COLOR_MAP.get(row['category'], 'gray')
            for cpu in row['cpus']:
                ax.barh(y=cpu, width=row['duration'], left=row['rel_start_time'], 
                        color=color, edgecolor='none', height=0.8)
                
        ax.set_title(f'Hardware Execution Timeline: {exp_name} ({run_name})', pad=15)
        ax.set_xlabel('Wall-clock Time (Seconds)')
        ax.set_ylabel('Hardware CPU Core ID')
        
        all_cpus = [cpu for cpus in run_df['cpus'] for cpu in cpus]
        if all_cpus:
            ax.set_ylim(min(all_cpus) - 1, max(all_cpus) + 1)
            ax.set_yticks(range(min(all_cpus), max(all_cpus) + 1))
            
        legend_elements = [Patch(facecolor=color, label=cat) for cat, color in COLOR_MAP.items()]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1.05))
        
        plt.tight_layout()
        plt.savefig(os.path.join(GANTT_DIR, f"gantt_{exp_name}_{run_name}.png"))
        plt.close(fig)

def plot_task_evolution(df):
    """Plot line-and-marker duration evolution EXCLUSIVELY for exp_10."""
    print("Plotting line-and-marker task duration evolution for Exp 10...")
    
    exp_10_df = df[df['exp_name'].str.contains('exp_10', na=False, case=False)]
    
    if exp_10_df.empty:
        print("Could not find 'exp_10' in data. Skipping evolution plot.")
        return
    
    run_df = exp_10_df[exp_10_df['run_name'] == 'run_1'].copy()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for category, color in COLOR_MAP.items():
        cat_df = run_df[run_df['category'] == category].sort_values('rel_start_time')
        if not cat_df.empty:
            ax.plot(cat_df['rel_start_time'], cat_df['duration'], 
                    marker=MARKER_MAP[category], color=color, 
                    linestyle='-', linewidth=2, markersize=9, 
                    alpha=0.85, label=category)
            
    ax.set_title('Task Duration Evolution Over Time (18 Loops Scenario)', pad=15)
    ax.set_xlabel('Experiment Wall-clock Time (Seconds)')
    ax.set_ylabel('Individual Task Duration (Seconds)')
    ax.legend(title="Task Type")
    
    plt.tight_layout()
    plt.savefig(os.path.join(TIME_EVOL_DIR, 'duration_evolution_exp_10.png'))
    plt.close(fig)

def plot_cumulative_quadratic(df):
    """Plot cumulative time growth to show quadratic vs linear scaling for exp_10."""
    print("Plotting cumulative task time curves for Exp 10...")
    
    exp_10_df = df[df['exp_name'].str.contains('exp_10', na=False, case=False)]
    if exp_10_df.empty: return
    
    run_df = exp_10_df[exp_10_df['run_name'] == 'run_1'].copy()
    run_df = run_df.sort_values('rel_start_time')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for category in ['Training', 'Optimize Structure']:
        cat_df = run_df[run_df['category'] == category].copy()
        if not cat_df.empty:
            cat_df['cumulative_time'] = cat_df['duration'].cumsum()
            ax.plot(cat_df['rel_start_time'], cat_df['cumulative_time'], 
                    marker=MARKER_MAP[category], color=COLOR_MAP[category],
                    linestyle='--', linewidth=2.5, markersize=8, 
                    label=f'Cumulative {category}')
            
    ax.set_title('Cumulative Component Time vs Experiment Timeline', pad=15)
    ax.set_xlabel('Experiment Wall-clock Time (Seconds)')
    ax.set_ylabel('Total Cumulative Duration (Seconds)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(TIME_EVOL_DIR, 'cumulative_quadratic_curve_exp_10.png'))
    plt.close(fig)

def plot_cumulative_average_bar(df):
    """Plot extrapolated stacked bar charts with embedded percentage labels."""
    print("Calculating and plotting cumulative average makespan with percentages...")
    
    summary_records = []
    grouped = df.groupby(['exp_name', 'run_name'])
    
    for (exp_name, run_name), run_df in grouped:
        opt_time = run_df[run_df['category'] == 'Optimize Structure']['duration'].sum()
        eng_time = run_df[run_df['category'] == 'Compute Energy']['duration'].sum()
        inf_time = run_df[run_df['category'] == 'ML Inference']['duration'].sum()
        train_time = run_df[run_df['category'] == 'Training']['duration'].sum()
        
        mol_count = len(run_df[run_df['category'] == 'Optimize Structure'])
        scale_factor = 1.0
        if 'exp_01' in exp_name and 0 < mol_count < 300:
            scale_factor = 300.0 / mol_count
        
        summary_records.append({
            'exp_name': exp_name,
            'run_name': run_name,
            'Optimize Structure': opt_time * scale_factor,
            'Compute Energy': eng_time * scale_factor,
            'ML Inference': inf_time * scale_factor,
            'Training': train_time * scale_factor
        })
        
    summary_df = pd.DataFrame(summary_records)
    avg_df = summary_df.groupby('exp_name')[
        ['Optimize Structure', 'Compute Energy', 'ML Inference', 'Training']
    ].mean().reset_index().sort_values('exp_name')
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    bottom_eng = avg_df['Optimize Structure'].values
    bottom_inf = bottom_eng + avg_df['Compute Energy'].values
    bottom_train = bottom_inf + avg_df['ML Inference'].values
    
    ax.bar(avg_df['exp_name'], avg_df['Optimize Structure'], label='Optimize Structure', color=COLOR_MAP['Optimize Structure'])
    ax.bar(avg_df['exp_name'], avg_df['Compute Energy'], bottom=bottom_eng, label='Compute Energy', color=COLOR_MAP['Compute Energy'])
    ax.bar(avg_df['exp_name'], avg_df['ML Inference'], bottom=bottom_inf, label='ML Inference', color=COLOR_MAP['ML Inference'])
    ax.bar(avg_df['exp_name'], avg_df['Training'], bottom=bottom_train, label='Training', color=COLOR_MAP['Training'])
    
    for i, row in avg_df.iterrows():
        total_time = row['Optimize Structure'] + row['Compute Energy'] + row['ML Inference'] + row['Training']
        if total_time > 0 and row['Training'] > 0:
            train_pct = (row['Training'] / total_time) * 100
            y_pos = row['Optimize Structure'] + row['Compute Energy'] + row['ML Inference'] + (row['Training'] / 2)
            
            ax.text(i, y_pos, f'{train_pct:.1f}%', 
                    ha='center', va='center', 
                    color='white', fontweight='bold', fontsize=12)
    
    ax.set_title('Average Total Makespan Distribution (Extrapolated to 300 Molecules)', pad=20)
    ax.set_ylabel('Cumulative Wall-clock Time (Seconds)')
    ax.set_xlabel('Experiment Case')
    
    plt.xticks(rotation=45, ha='right')
    ax.legend(title="Execution Component", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'cumulative_average_makespan.png'))
    plt.close(fig)

# ==========================================
# 4. Main Execution Block
# ==========================================
if __name__ == "__main__":
    df_logs = parse_experiment_data()
    df_energy = parse_simulation_results()
    
    if not df_energy.empty:
        plot_energy_distribution(df_energy)
    
    if df_logs.empty:
        print("No valid JSON logs parsed for timing data. Please check the BASE_DIR path.")
    else:
        # plot_gantt_charts(df_logs)
        # plot_task_evolution(df_logs)
        # plot_cumulative_quadratic(df_logs)
        # plot_cumulative_average_bar(df_logs)
        print(f"Analysis complete. All high-res charts saved in the '{RESULTS_DIR}' directory.")