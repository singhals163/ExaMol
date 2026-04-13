import os
import glob
import json
import math
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

# Vibrant, high-contrast academic colors
INDIGO_COLOR = '#4A148C'       # Baseline Always (Deep Richter Purple)
TEAL_COLOR = '#00ACC1'         # Front Loaded (Vibrant Teal)
YELLOW_GREEN_COLOR = '#7CB342' # No Training (Crisp Green)

# Gantt Chart Background Color
GANTT_NO_TRAIN = '#E0E0E0'     # Light Gray for "Sim Only" background

def setup_focused_academic_plot_style():
    """Configures matplotlib and seaborn for highly polished, borderless figures."""
    plt.rcParams.update({
        'font.size': 16,            
        'axes.labelsize': 20,       
        'axes.labelweight': 'bold', 
        'axes.titlesize': 0,        
        'xtick.labelsize': 16,      
        'ytick.labelsize': 16,
        'legend.fontsize': 14,      
        'legend.frameon': False,    # Default to False for Figs 7 & 8; explicitly enabled in Fig 9
        'legend.title_fontsize': 0, 
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.5,     
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'sans-serif', 
    })
    sns.set_style("ticks")

# --- PARSERS ---

def parse_logs(base_dir):
    """Traverses the directory structure and extracts the simulation energies."""
    experiment_data = defaultdict(list)
    search_pattern = os.path.join(base_dir, "exp_*", "run_*", "run", "run_sequence.log")
    log_files = glob.glob(search_pattern)
    
    if not log_files:
        print(f"Warning: No log files found in {base_dir}")
        return experiment_data

    for filepath in log_files:
        path_parts = os.path.normpath(filepath).split(os.sep)
        exp_name = next(part for part in path_parts if part.startswith("exp_"))
        clean_name = "_".join(exp_name.split("_")[2:]).replace("_", " ").title()
        
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith("Simulation result |"):
                    try:
                        val_str = line.strip().split("value: ")[-1]
                        experiment_data[clean_name].append(float(val_str))
                    except (IndexError, ValueError):
                        pass
                        
    return experiment_data

def parse_profiling(base_dir):
    """Parses both profile_stats and train-results to extract task timestamps and training intervals."""
    timing_data = defaultdict(lambda: defaultdict(dict))
    
    search_pattern = os.path.join(base_dir, "exp_*", "run_*", "run")
    run_dirs = glob.glob(search_pattern)
    
    if not run_dirs:
        print(f"Warning: No run directories found in {base_dir}")
        return timing_data
        
    for run_dir in run_dirs:
        path_parts = os.path.normpath(run_dir).split(os.sep)
        exp_name = next(part for part in path_parts if part.startswith("exp_"))
        run_name = next(part for part in path_parts if part.startswith("run_"))
        clean_name = "_".join(exp_name.split("_")[2:]).replace("_", " ").title()
        
        min_start = float('inf')
        max_end = float('-inf')
        opt_end_times = []
        raw_train_intervals = []
        
        prof_files = glob.glob(os.path.join(run_dir, "profile_stats.json*"))
        for filepath in prof_files:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        metrics = data.get('metrics', {})
                        task_type = data.get('task_type', '')
                        
                        for m_val in metrics.values():
                            if isinstance(m_val, dict):
                                st = m_val.get('start_timestamp')
                                dur = m_val.get('duration', 0.0)
                                if st is not None:
                                    min_start = min(min_start, st)
                                    max_end = max(max_end, st + dur)
                        
                        if task_type.startswith('optimize_structure'):
                            if 'total_optimization_time' in metrics:
                                dur = metrics['total_optimization_time'].get('duration', 0.0)
                                st = metrics['total_optimization_time'].get('start_timestamp')
                                if st is not None:
                                    opt_end_times.append(st + dur)
                    except json.JSONDecodeError:
                        pass
        
        train_results_path = os.path.join(run_dir, "train-results.json")
        if os.path.exists(train_results_path):
            with open(train_results_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data.get('method') == 'retrain':
                            t_start = data.get('time_compute_started')
                            t_end = data.get('time_compute_ended')
                            if t_start is not None and t_end is not None:
                                raw_train_intervals.append((t_start, t_end))
                                min_start = min(min_start, t_start)
                                max_end = max(max_end, t_end)
                    except json.JSONDecodeError:
                        pass
                        
        if min_start != float('inf'):
            timing_data[clean_name][run_name] = {
                'global_start': min_start,
                'global_end': max_end,
                'opt_end_times': sorted(opt_end_times),
                'raw_train_intervals': raw_train_intervals
            }
            
    return timing_data

def parse_run_data(base_dir):
    """Parses both logs and profiles to link Quality and Time per run for Pareto analysis."""
    run_data = defaultdict(lambda: defaultdict(lambda: {'energies': [], 'total_time': 0}))
    
    log_search = os.path.join(base_dir, "exp_*", "run_*", "run", "run_sequence.log")
    for filepath in glob.glob(log_search):
        path_parts = os.path.normpath(filepath).split(os.sep)
        run_name = next(part for part in path_parts if part.startswith("run_"))
        exp_name = next(part for part in path_parts if part.startswith("exp_"))
        policy = "_".join(exp_name.split("_")[2:]).replace("_", " ").title()
            
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith("Simulation result |"):
                    try:
                        val = float(line.strip().split("value: ")[-1])
                        run_data[policy][run_name]['energies'].append(val)
                    except ValueError:
                        pass

    prof_search = os.path.join(base_dir, "exp_*", "run_*", "run", "profile_stats.json*")
    for filepath in glob.glob(prof_search):
        path_parts = os.path.normpath(filepath).split(os.sep)
        run_name = next(part for part in path_parts if part.startswith("run_"))
        exp_name = next(part for part in path_parts if part.startswith("exp_"))
        policy = "_".join(exp_name.split("_")[2:]).replace("_", " ").title()
            
        min_start, max_end = float('inf'), float('-inf')
        with open(filepath, 'r') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    for m_val in data.get('metrics', {}).values():
                        if isinstance(m_val, dict) and 'start_timestamp' in m_val:
                            st = m_val['start_timestamp']
                            dur = m_val.get('duration', 0.0)
                            min_start = min(min_start, st)
                            max_end = max(max_end, st + dur)
                except json.JSONDecodeError:
                    pass
                    
        if min_start != float('inf'):
            run_data[policy][run_name]['total_time'] = max_end - min_start

    return run_data

# --- PLOTTERS ---

def plot_focused_energy_frequency(energy_data, output_dir):
    """Generates Figure 7: Focused Energy Frequency KDE Plot."""
    fig, ax = plt.subplots(figsize=(6, 4.5)) 
    target_policies = ["Baseline Always", "Front Loaded", "No Training"]
    color_map = {target_policies[0]: INDIGO_COLOR, target_policies[1]: TEAL_COLOR, target_policies[2]: YELLOW_GREEN_COLOR}

    focused_values = []
    for p in target_policies:
        focused_values.extend(energy_data.get(p, []))
    
    if not focused_values:
        return

    for policy in target_policies:
        if policy in energy_data:
            energies = energy_data[policy]
            color = color_map[policy]
            
            sns.kdeplot(
                data=energies, label=policy, color=color, 
                fill=True, alpha=0.35, linewidth=2.5, ax=ax, warn_singular=False
            )

    ax.set_xlabel("Redox Energy", labelpad=10)
    ax.set_ylabel("Frequency", labelpad=10)
    
    ax.yaxis.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2)
    ax.xaxis.grid(False)
    sns.despine(ax=ax, top=True, right=True)
    
    ax.legend(loc='upper left', fontsize=13, frameon=False) 
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig7_focused_energy_frequency.png"))
    plt.savefig(os.path.join(output_dir, "fig7_focused_energy_frequency.pdf"))
    plt.close()

def plot_combined_rolling_throughput(timing_data, output_dir):
    """Generates Figure 8: Rolling Throughput with Run-1 Gantt Timeline."""
    target_policies = ["Baseline Always", "Front Loaded", "No Training"]
    color_map = {target_policies[0]: INDIGO_COLOR, target_policies[1]: TEAL_COLOR, target_policies[2]: YELLOW_GREEN_COLOR}

    max_global_time = 0
    for policy in target_policies:
        runs_dict = timing_data.get(policy, {})
        for run_info in runs_dict.values():
            if run_info['opt_end_times']:
                run_max = run_info['global_end'] - run_info['global_start']
                if run_max > max_global_time:
                    max_global_time = run_max
                    
    if max_global_time == 0:
        print("Warning: No timing data found for focused optimize_structure tasks. Skipped Fig 8.")
        return

    fig, (ax_main, ax_gantt) = plt.subplots(
        nrows=2, ncols=1, figsize=(10, 5.5), 
        gridspec_kw={'height_ratios': [4, 0.6], 'hspace': 0.05}, 
        sharex=True
    )

    max_t = int(math.ceil(max_global_time))
    t_grid = np.arange(0, max_t + 1, 1)
    gantt_target_run = 'run_1'
    
    for policy in target_policies:
        if policy in timing_data:
            color = color_map[policy]
            runs_dict = timing_data[policy]
            all_throughputs = []
            
            for run_name, run_info in runs_dict.items():
                raw_ends = run_info['opt_end_times']
                if not raw_ends: continue
                    
                global_start = run_info['global_start']
                rel_ends = np.array(raw_ends) - global_start
                
                counts_at_t = np.searchsorted(rel_ends, t_grid, side='right')
                counts_at_t_minus_60 = np.searchsorted(rel_ends, t_grid - 60, side='right')
                
                throughput = counts_at_t - counts_at_t_minus_60
                all_throughputs.append(throughput)
                
            if all_throughputs:
                all_throughputs = np.array(all_throughputs)
                mean_tp = np.mean(all_throughputs, axis=0)
                std_tp = np.std(all_throughputs, axis=0)
                
                ax_main.plot(t_grid, mean_tp, label=policy, color=color, linewidth=3.0)
                ax_main.fill_between(t_grid, np.maximum(0, mean_tp - std_tp), mean_tp + std_tp, color=color, alpha=0.15)

    ax_main.set_ylabel("Throughput\n(Sim Tasks / Min)", labelpad=15)
    ax_main.yaxis.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2)
    ax_main.xaxis.grid(False)
    sns.despine(ax=ax_main, top=True, right=True, bottom=True) 
    ax_main.tick_params(axis='x', length=0) 
    ax_main.legend(loc='upper right', fontsize=14, frameon=False) 
    ax_main.set_ylim(bottom=0)

    for idx, policy in enumerate(target_policies):
        if policy not in timing_data: 
            continue
            
        runs_dict = timing_data[policy]
        color = color_map[policy]
        
        run_name = gantt_target_run if gantt_target_run in runs_dict else sorted(runs_dict.keys())[0]
        run_info = runs_dict[run_name]
        
        global_start = run_info['global_start']
        global_end = run_info['global_end']
        run_duration = global_end - global_start
        
        ax_gantt.barh(idx, run_duration, left=0, height=0.15, color=GANTT_NO_TRAIN)
        
        train_intervals = run_info.get('raw_train_intervals', [])
        for i, (t_start, t_end) in enumerate(train_intervals):
            rel_start = t_start - global_start
            rel_dur = t_end - t_start
            ax_gantt.barh(idx, rel_dur, left=rel_start, height=0.25, color=color)

    ax_gantt.set_yticks([]) 
    ax_gantt.invert_yaxis() 
    ax_gantt.set_ylim(2.5, -0.5)
    ax_gantt.set_xlabel("Time since Experiment Start (seconds)", labelpad=15)
    
    sns.despine(ax=ax_gantt, top=True, right=True, left=True)
    ax_gantt.tick_params(axis='y', length=0) 

    plt.savefig(os.path.join(output_dir, "fig8_combined_rolling_throughput.png"), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "fig8_combined_rolling_throughput.pdf"), bbox_inches='tight')
    plt.close()

def plot_pareto_efficiency(run_data, output_dir):
    """Generates Figure 9: Cost vs Quality Pareto Scatter Plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    target_policies = ["Baseline Always", "Front Loaded", "No Training"]
    color_map = {target_policies[0]: INDIGO_COLOR, target_policies[1]: TEAL_COLOR, target_policies[2]: YELLOW_GREEN_COLOR}
    marker_map = {target_policies[0]: 'o', target_policies[1]: 's', target_policies[2]: '^'}

    TOP_N = 10 

    for policy in target_policies:
        if policy not in run_data: continue
        
        times = []
        qualities = []
        
        for run_name, metrics in run_data[policy].items():
            energies = metrics['energies']
            total_time = metrics['total_time']
            
            if len(energies) >= TOP_N and total_time > 0:
                top_energies = sorted(energies)[-TOP_N:]
                quality = np.mean(top_energies)
                
                times.append(total_time)
                qualities.append(quality)
                
        if times:
            ax.scatter(times, qualities, label=policy, color=color_map[policy], 
                       marker=marker_map[policy], s=150, alpha=0.8, edgecolor='black', linewidth=1)
            ax.scatter(np.mean(times), np.mean(qualities), color=color_map[policy], 
                       marker='X', s=400, alpha=0.3)

    ax.set_xlabel("Total Wall Clock Time (seconds)", labelpad=10)
    ax.set_ylabel(f"Quality (Avg Top {TOP_N} Energy)", labelpad=10)
    
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    sns.despine(ax=ax, top=True, right=True)
    
    # Explicitly turn on the frame just for this legend to maintain clarity over scatter points
    ax.legend(title="Strategy", loc='lower right', fontsize=13, title_fontsize=13, frameon=True) 
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig9_pareto_efficiency.png"))
    plt.savefig(os.path.join(output_dir, "fig9_pareto_efficiency.pdf"))
    plt.close()

# --- MAIN EXECUTOR ---

def main():
    parser = argparse.ArgumentParser(description="Generate focused figures (Figs 7, 8, 9) for academic paper.")
    parser.add_argument("base_dir", type=str, help="Path to the master directory containing the exp_* folders")
    args = parser.parse_args()
    
    base_dir = os.path.abspath(args.base_dir)
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist.")
        return

    output_dir = os.path.join(base_dir, "analysis_results")
    os.makedirs(output_dir, exist_ok=True)

    print("Setting up academic plot styles...")
    setup_focused_academic_plot_style()

    print("\n[Fig 7] Parsing energy sequence logs...")
    experiment_data = parse_logs(base_dir)
    if experiment_data:
        plot_focused_energy_frequency(experiment_data, output_dir)
        print("  -> Generated fig7_focused_energy_frequency")

    print("\n[Fig 8] Parsing JSON profiling and training stats...")
    timing_data = parse_profiling(base_dir)
    if timing_data:
        plot_combined_rolling_throughput(timing_data, output_dir)
        print("  -> Generated fig8_combined_rolling_throughput")

    print("\n[Fig 9] Parsing pareto run data...")
    pareto_data = parse_run_data(base_dir)
    if pareto_data:
        plot_pareto_efficiency(pareto_data, output_dir)
        print("  -> Generated fig9_pareto_efficiency")
    
    print("\nSuccess! All focused, highly polished figures generated.")

if __name__ == "__main__":
    main()