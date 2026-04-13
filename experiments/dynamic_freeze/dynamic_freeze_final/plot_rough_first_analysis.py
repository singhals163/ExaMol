import os
import glob
import json
import math
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap

def setup_academic_plot_style():
    """Configures matplotlib and seaborn for clean, publication-ready figures."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'legend.frameon': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.2,
        'lines.linewidth': 2.5,
        'figure.figsize': (10, 6),
        'figure.dpi': 300,
        'savefig.bbox': 'tight'
    })
    sns.set_style("ticks")

# Standard academic color palette for initial plots
ACADEMIC_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', 
                   '#ff7f00', '#ffff33', '#a65628', '#f781bf']

# Distinct academic colorblind-friendly Viridis colors (indigo/teal) from image_7.png
# and as commonly used for density-style plots.
INDIGO_COLOR = '#440154' # Baseline Always
TEAL_COLOR = '#21908d'   # Front Loaded

def parse_logs(base_dir):
    """Traverses the directory structure and extracts the simulation energies."""
    experiment_data = defaultdict(list)
    search_pattern = os.path.join(base_dir, "exp_*", "run_*", "run", "run_sequence.log")
    log_files = glob.glob(search_pattern)
    
    if not log_files:
        print(f"Warning: No log files found matching pattern {search_pattern}")
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
    """Parses profile_stats.json to extract runtimes, resource usage, and task timestamps."""
    timing_data = defaultdict(lambda: defaultdict(dict))
    
    search_pattern = os.path.join(base_dir, "exp_*", "run_*", "run", "profile_stats.jsonl")
    prof_files = glob.glob(search_pattern)
    
    if not prof_files:
        print(f"Warning: No profile stats found matching {search_pattern}")
        return timing_data
        
    for filepath in prof_files:
        path_parts = os.path.normpath(filepath).split(os.sep)
        exp_name = next(part for part in path_parts if part.startswith("exp_"))
        run_name = next(part for part in path_parts if part.startswith("run_"))
        clean_name = "_".join(exp_name.split("_")[2:]).replace("_", " ").title()
        
        min_start = float('inf')
        max_end = float('-inf')
        
        cum_retrain_wall_time = 0.0
        cum_sim_resources = 0.0
        cum_retrain_resources = 0.0
        
        opt_end_times = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    metrics = data.get('metrics', {})
                    task_type = data.get('task_type', '')
                    
                    # Search metrics to find global min start and global max end
                    for m_val in metrics.values():
                        if isinstance(m_val, dict):
                            st = m_val.get('start_timestamp')
                            dur = m_val.get('duration', 0.0)
                            if st is not None:
                                min_start = min(min_start, st)
                                max_end = max(max_end, st + dur)
                    
                    # Accumulate CPU Resources & Capture End Times
                    if task_type.startswith('optimize_structure'):
                        if 'total_optimization_time' in metrics:
                            dur = metrics['total_optimization_time'].get('duration', 0.0)
                            st = metrics['total_optimization_time'].get('start_timestamp')
                            cum_sim_resources += dur
                            if st is not None:
                                opt_end_times.append(st + dur)
                                
                    elif task_type.startswith('compute_energy'):
                        if 'total_compute_energy_time' in metrics:
                            cum_sim_resources += metrics['total_compute_energy_time'].get('duration', 0.0)
                            
                    elif task_type == 'training_retrain':
                        if 'total_retrain_time' in metrics:
                            retrain_dur = metrics['total_retrain_time'].get('duration', 0.0)
                            cum_retrain_wall_time += retrain_dur
                            cum_retrain_resources += (retrain_dur * 3) 
                            
                except json.JSONDecodeError:
                    pass
                    
        if min_start != float('inf') and max_end != float('-inf'):
            total_run_time = max_end - min_start
            sim_time = total_run_time - cum_retrain_wall_time
            
            timing_data[clean_name][run_name] = {
                'sim_time': sim_time,
                'retrain_time': cum_retrain_wall_time,
                'sim_resources': cum_sim_resources,
                'retrain_resources': cum_retrain_resources,
                'global_start': min_start,
                'opt_end_times': sorted(opt_end_times)
            }
            
    return timing_data

def prepare_dataframe(data):
    records = []
    for policy, energies in data.items():
        for energy in energies:
            records.append({"Policy": policy, "Redox Energy": energy})
    return pd.DataFrame(records)

# --- Standard plots ---

def plot_frequency_distributions(data, output_dir, color_map):
    fig, ax = plt.subplots()
    all_values = [val for exp_vals in data.values() for val in exp_vals]
    global_min = np.floor(min(all_values) * 10) / 10.0
    global_max = np.ceil(max(all_values) * 10) / 10.0
    bins = np.arange(global_min, global_max + 0.1, 0.1)
    
    for exp_name, energies in sorted(data.items()):
        color = color_map[exp_name]
        counts, bin_edges = np.histogram(energies, bins=bins)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        ax.plot(bin_centers, counts, label=exp_name, color=color, alpha=0.9, linewidth=2.5)
        ax.fill_between(bin_centers, 0, counts, color=color, alpha=0.15)

    ax.set_xlabel("Redox Energy")
    ax.set_ylabel("Frequency")
    ax.set_title("Frequency Distribution of Discovered Molecules")
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(title="Training Policy", loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    ax.set_ylim(bottom=0)
    plt.savefig(os.path.join(output_dir, "fig1_energy_frequency.png"))
    plt.savefig(os.path.join(output_dir, "fig1_energy_frequency.pdf"))
    plt.close()

def plot_violin(df, output_dir, descending_order, color_map):
    fig, ax = plt.subplots()
    sns.violinplot(data=df, x="Policy", y="Redox Energy", order=descending_order,
                   palette=color_map, inner="quartile", linewidth=1.5, ax=ax)
    ax.set_title("Distribution Density of Discovered Redox Energies")
    ax.set_xlabel("Training Policy")
    ax.set_ylabel("Redox Energy")
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(output_dir, "fig2_energy_violin.png"))
    plt.savefig(os.path.join(output_dir, "fig2_energy_violin.pdf"))
    plt.close()

def plot_bar_chart(df, output_dir, descending_order, color_map):
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="Policy", y="Redox Energy", order=descending_order,
                palette=color_map, errorbar="sd", capsize=0.1, 
                err_kws={'linewidth': 1.5, 'color': 'black'}, edgecolor='black', linewidth=1.5, ax=ax)
    ax.set_title("Average Discovered Redox Energy (± Standard Deviation)")
    ax.set_xlabel("Training Policy")
    ax.set_ylabel("Average Redox Energy")
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(output_dir, "fig3_energy_bar_std.png"))
    plt.savefig(os.path.join(output_dir, "fig3_energy_bar_std.pdf"))
    plt.close()

def plot_stacked_timings(timing_data, output_dir, descending_order):
    fig, ax = plt.subplots(figsize=(12, 6))
    n_policies = len(descending_order)
    max_runs = max((len(timing_data.get(p, {})) for p in descending_order), default=0)
    if max_runs == 0: return

    total_group_width = 0.8
    bar_width = total_group_width / max_runs
    x_base = np.arange(n_policies)
    color_sim = '#377eb8'      
    color_retrain = '#e41a1c'  
    
    sim_added, ret_added = False, False
    
    for i, policy in enumerate(descending_order):
        runs_dict = timing_data.get(policy, {})
        sorted_runs = sorted(runs_dict.keys())
        start_offset = -total_group_width/2 + bar_width/2
        for j, run_name in enumerate(sorted_runs):
            x_pos = x_base[i] + start_offset + (j * bar_width)
            sim_time = runs_dict[run_name].get('sim_time', 0)
            ret_time = runs_dict[run_name].get('retrain_time', 0)
            ax.bar(x_pos, sim_time, width=bar_width*0.9, color=color_sim, edgecolor='black', linewidth=0.5,
                   label='Simulation (Wall Time)' if not sim_added else "")
            sim_added = True
            ax.bar(x_pos, ret_time, width=bar_width*0.9, bottom=sim_time, color=color_retrain, edgecolor='black', linewidth=0.5,
                   label='Retrain (Wall Time)' if not ret_added else "")
            ret_added = True

    ax.set_xticks(x_base)
    ax.set_xticklabels(descending_order, rotation=45, ha='right')
    ax.set_ylabel("Wall-clock Time (seconds)")
    ax.set_title("Total Wall-clock Time Breakdown per Experiment")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.savefig(os.path.join(output_dir, "fig4_timing_breakdown.png"))
    plt.savefig(os.path.join(output_dir, "fig4_timing_breakdown.pdf"))
    plt.close()

def plot_stacked_resources(timing_data, output_dir, descending_order):
    fig, ax = plt.subplots(figsize=(12, 6))
    n_policies = len(descending_order)
    max_runs = max((len(timing_data.get(p, {})) for p in descending_order), default=0)
    if max_runs == 0: return

    total_group_width = 0.8
    bar_width = total_group_width / max_runs
    x_base = np.arange(n_policies)
    color_sim_res = '#4daf4a'      
    color_retrain_res = '#ff7f00'  
    
    sim_added, ret_added = False, False
    
    for i, policy in enumerate(descending_order):
        runs_dict = timing_data.get(policy, {})
        sorted_runs = sorted(runs_dict.keys())
        start_offset = -total_group_width/2 + bar_width/2
        for j, run_name in enumerate(sorted_runs):
            x_pos = x_base[i] + start_offset + (j * bar_width)
            sim_res = runs_dict[run_name].get('sim_resources', 0)
            ret_res = runs_dict[run_name].get('retrain_resources', 0)
            ax.bar(x_pos, sim_res, width=bar_width*0.9, color=color_sim_res, edgecolor='black', linewidth=0.5,
                   label='Simulation (CPU-s)' if not sim_added else "")
            sim_added = True
            ax.bar(x_pos, ret_res, width=bar_width*0.9, bottom=sim_res, color=color_retrain_res, edgecolor='black', linewidth=0.5,
                   label='Retrain (CPU-s)' if not ret_added else "")
            ret_added = True

    ax.set_xticks(x_base)
    ax.set_xticklabels(descending_order, rotation=45, ha='right')
    ax.set_ylabel("Compute Resources (CPU-seconds)")
    ax.set_title("Total Compute Resource Breakdown per Experiment\n(Simulation vs. Retraining)")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.savefig(os.path.join(output_dir, "fig5_resource_breakdown.png"))
    plt.savefig(os.path.join(output_dir, "fig5_resource_breakdown.pdf"))
    plt.close()

def plot_rolling_throughput(timing_data, output_dir, descending_order, color_map):
    max_global_time = 0
    for policy in descending_order:
        for run_info in timing_data.get(policy, {}).values():
            if run_info['opt_end_times']:
                run_max = run_info['opt_end_times'][-1] - run_info['global_start']
                if run_max > max_global_time:
                    max_global_time = run_max
                    
    if max_global_time == 0: return

    max_t = int(math.ceil(max_global_time))
    t_grid = np.arange(0, max_t + 1, 1)

    n_policies = len(descending_order)
    ncols = 2
    nrows = math.ceil(n_policies / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.5 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, policy in enumerate(descending_order):
        ax = axes[idx]
        color = color_map.get(policy, '#333333')
        runs_dict = timing_data.get(policy, {})
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
            
            ax.plot(t_grid, mean_tp, color=color, linewidth=2)
            ax.fill_between(t_grid, np.maximum(0, mean_tp - std_tp), mean_tp + std_tp, color=color, alpha=0.2)
            
        ax.set_title(policy)
        ax.grid(True, linestyle='--', alpha=0.3)
        if idx % ncols == 0:
            ax.set_ylabel("Throughput\n(Tasks / Min)")
        if idx >= n_policies - ncols:
            ax.set_xlabel("Time since Experiment Start (seconds)")

    for idx in range(n_policies, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig6_throughput_subplots.png"))
    plt.savefig(os.path.join(output_dir, "fig6_throughput_subplots.pdf"))
    plt.close()


# --- NEW FOCUSED PLOTS ---

def setup_focused_academic_plot_style():
    """Configures matplotlib and seaborn for smaller, borderless figures with large fonts."""
    plt.rcParams.update({
        'font.size': 16,            # Larger base font
        'axes.labelsize': 20,       # Larger labels
        'axes.titlesize': 0,        # Explicitly remove main title space
        'xtick.labelsize': 16,      # Larger ticks
        'ytick.labelsize': 16,
        'legend.fontsize': 16,      # Larger legend
        'legend.frameon': False,     # No legend box surrounding box
        'legend.title_fontsize': 0, # Remove legend title space
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.0,     # user requested reduced width for *frequency* lines specifically. I'll make these thinner.
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'sans-serif', # Cleaner academic look
    })
    sns.set_style("ticks")

def plot_focused_energy_frequency(energy_data, output_dir):
    """Generates a smaller, styled Figure focusing ONLY on Baseline Always and Front Loaded with a Viridis-like palette."""
    setup_focused_academic_plot_style()
    
    fig, ax = plt.subplots(figsize=(8, 5)) # Smaller figure
    target_policies = ["Baseline Always", "Front Loaded"]
    
    # Custom color map using INDIGO_COLOR and TEAL_COLOR
    color_map = {
        target_policies[0]: INDIGO_COLOR, 
        target_policies[1]: TEAL_COLOR
    }

    # Bins: must calculate based on the focused data only for correct ranges
    focused_values = []
    for p in target_policies:
        focused_values.extend(energy_data.get(p, []))
    
    if not focused_values:
        print("Warning: No energy data found for focused policies. Skipped.")
        plt.close()
        return

    global_min = np.floor(min(focused_values) * 10) / 10.0
    global_max = np.ceil(max(focused_values) * 10) / 10.0
    bins = np.arange(global_min, global_max + 0.1, 0.1)
    
    for policy in target_policies:
        if policy in energy_data:
            energies = energy_data[policy]
            color = color_map[policy]
            counts, bin_edges = np.histogram(energies, bins=bins)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            
            # Use smaller linewidth for lines
            ax.plot(bin_centers, counts, label=policy, color=color, alpha=0.9, linewidth=1.5)
            # Apply same distinct color for fill
            ax.fill_between(bin_centers, 0, counts, color=color, alpha=0.15)

    ax.set_xlabel("Redox Energy")
    ax.set_ylabel("Frequency")
    
    # Grid: keep academic tick style, grid can remain subtle
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Legend: remove surrounding box, large fonts, inside
    ax.legend(loc='best', fontsize=16) 
    
    # Ensure zero is the bottom of the y-axis
    ax.set_ylim(bottom=0)
    
    # No top/right spines (box removed)
    
    plt.savefig(os.path.join(output_dir, "fig7_focused_energy_frequency.png"))
    plt.savefig(os.path.join(output_dir, "fig7_focused_energy_frequency.pdf"))
    plt.close()

def plot_combined_rolling_throughput(timing_data, output_dir):
    """Generates a smaller, borderless Figure plotting Baseline and Front Loaded rolling throughput on a single shared axis."""
    setup_focused_academic_plot_style()
    
    # Smaller figure, single shared axis
    fig, ax = plt.subplots(figsize=(9, 6)) 
    target_policies = ["Baseline Always", "Front Loaded"]
    
    # Custom color map for consistency
    color_map = {
        target_policies[0]: INDIGO_COLOR, 
        target_policies[1]: TEAL_COLOR
    }

    # Determine maximum global time across *focused* policies to set common X grid
    max_global_time = 0
    for policy in target_policies:
        runs_dict = timing_data.get(policy, {})
        for run_info in runs_dict.values():
            if run_info['opt_end_times']:
                run_max = run_info['opt_end_times'][-1] - run_info['global_start']
                if run_max > max_global_time:
                    max_global_time = run_max
                    
    if max_global_time == 0:
        print("Warning: No timing data found for focused optimize_structure tasks. Skipped.")
        plt.close()
        return

    # 1-second grid from 0 to common max time
    max_t = int(math.ceil(max_global_time))
    t_grid = np.arange(0, max_t + 1, 1)
    
    for policy in target_policies:
        if policy in timing_data:
            color = color_map[policy]
            runs_dict = timing_data[policy]
            
            all_throughputs = []
            
            for run_name, run_info in runs_dict.items():
                raw_ends = run_info['opt_end_times']
                if not raw_ends:
                    continue
                    
                global_start = run_info['global_start']
                # Normalize end times to t=0
                rel_ends = np.array(raw_ends) - global_start
                
                #counts_at_t = total tasks completed by time t
                counts_at_t = np.searchsorted(rel_ends, t_grid, side='right')
                #counts_at_t_minus_60 = total tasks completed by time t-60
                counts_at_t_minus_60 = np.searchsorted(rel_ends, t_grid - 60, side='right')
                
                throughput = counts_at_t - counts_at_t_minus_60
                all_throughputs.append(throughput)
                
            if all_throughputs:
                all_throughputs = np.array(all_throughputs)
                mean_tp = np.mean(all_throughputs, axis=0)
                std_tp = np.std(all_throughputs, axis=0)
                
                # Plot mean line and shade standard deviation
                # Thrughput lines can remain distinct thickness, not thinned like frequency lines.
                ax.plot(t_grid, mean_tp, label=policy, color=color, linewidth=2.5)
                ax.fill_between(t_grid, np.maximum(0, mean_tp - std_tp), mean_tp + std_tp, color=color, alpha=0.15)

    ax.set_ylabel("Throughput\n(Tasks / Min)")
    ax.set_xlabel("Time since Experiment Start (seconds)")
    ax.set_title("Total Rolling 1-Minute Throughput\n(Baseline vs. Front Loaded)")
    
    # Styling consistent with request
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc='best', fontsize=16) # Legend inside, no box surrounding box
    
    # zero on y-axis
    ax.set_ylim(bottom=0)
    
    # No top/right spines (box removed)

    plt.savefig(os.path.join(output_dir, "fig8_combined_rolling_throughput.png"))
    plt.savefig(os.path.join(output_dir, "fig8_combined_rolling_throughput.pdf"))
    plt.close()

# --- Main function ---

def main():
    parser = argparse.ArgumentParser(description="Parse logs and plot energy/timing metrics.")
    parser.add_argument("base_dir", type=str, help="Path to the master directory containing the exp_* folders")
    
    args = parser.parse_args()
    base_dir = os.path.abspath(args.base_dir)
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist.")
        return

    print("Parsing energy sequence logs...")
    experiment_data = parse_logs(base_dir)
    if not experiment_data:
        print("No energy data parsed. Exiting.")
        return
    for exp, vals in sorted(experiment_data.items()):
        print(f"Parsed {len(vals)} simulations for {exp}")

    print("\nParsing JSON profiling stats...")
    timing_data = parse_profiling(base_dir)
    for exp, runs in sorted(timing_data.items()):
        print(f"Parsed {len(runs)} profiling runs for {exp}")

    # Convert to pandas DataFrame for Seaborn
    df = prepare_dataframe(experiment_data)

    # Establish strict color mapping based on alphabetical order for standard plots
    sorted_policies = sorted(experiment_data.keys())
    standard_color_map = {policy: ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)] for i, policy in enumerate(sorted_policies)}

    # Calculate the descending order of the means for consistent plotting
    descending_mean_order = df.groupby('Policy')['Redox Energy'].mean().sort_values(ascending=False).index.tolist()

    output_dir = os.path.join(base_dir, "analysis_results")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating plots in {output_dir}...")
    setup_academic_plot_style()
    
    # Pass standard color mapping and sorted orders to standard plots
    plot_frequency_distributions(experiment_data, output_dir, standard_color_map)
    print("- Frequency Distribution plotted (fig1)")
    
    plot_violin(df, output_dir, descending_mean_order, standard_color_map)
    print("- Violin Plot (Descending) plotted (fig2)")
    
    plot_bar_chart(df, output_dir, descending_mean_order, standard_color_map)
    print("- Bar Chart with StdDev (Descending) plotted (fig3)")
    
    if timing_data:
        plot_stacked_timings(timing_data, output_dir, descending_mean_order)
        print("- Stacked Timing Breakdown plotted (fig4)")
        
        plot_stacked_resources(timing_data, output_dir, descending_mean_order)
        print("- Stacked Resource Breakdown plotted (fig5)")
        
        plot_rolling_throughput(timing_data, output_dir, descending_mean_order, standard_color_map)
        print("- Rolling Throughput Subplots plotted (fig6)")
        
        # --- ADD NEW FOCUSED PLOTS HERE ---
        plot_focused_energy_frequency(experiment_data, output_dir)
        print("- Focused Energy Frequency plotted (fig7)")
        
        plot_combined_rolling_throughput(timing_data, output_dir)
        print("- Combined Rolling Throughput plotted (fig8)")
        
    else:
        print("- Skipped Timing and Resource Plots (No profile_stats.json files found)")
    
    print("\nSuccess! All academic figures generated.")

if __name__ == "__main__":
    main()