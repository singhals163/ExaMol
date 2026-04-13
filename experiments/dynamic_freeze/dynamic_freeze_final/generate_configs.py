import os
import glob
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

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
        'figure.figsize': (10, 6),
        'figure.dpi': 300,
        'savefig.bbox': 'tight'
    })
    sns.set_style("ticks")

# Cohesive academic color palette
ACADEMIC_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', 
                   '#ff7f00', '#ffff33', '#a65628', '#f781bf']

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
        
        # Clean up name: 'exp_00_baseline_always' -> 'Baseline Always'
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

def prepare_dataframe(data):
    """Converts the dictionary into a flat Pandas DataFrame."""
    records = []
    for policy, energies in data.items():
        for energy in energies:
            records.append({"Policy": policy, "Redox Energy": energy})
    return pd.DataFrame(records)

def plot_frequency_distributions(data, output_dir, color_map):
    """Creates an overlaid frequency distribution line curve."""
    fig, ax = plt.subplots()
    
    all_values = [val for exp_vals in data.values() for val in exp_vals]
    global_min = np.floor(min(all_values) * 10) / 10.0
    global_max = np.ceil(max(all_values) * 10) / 10.0
    bins = np.arange(global_min, global_max + 0.1, 0.1)
    
    # Sort alphabetically here to keep the legend neat
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
    """Creates a violin plot sorted in descending order of the mean."""
    fig, ax = plt.subplots()
    
    sns.violinplot(
        data=df, 
        x="Policy", 
        y="Redox Energy", 
        order=descending_order,  # Forces the descending sort
        palette=color_map,       # Ensures colors match Fig 1 perfectly
        inner="quartile", 
        linewidth=1.5,
        ax=ax
    )
    
    ax.set_title("Distribution Density of Discovered Redox Energies")
    ax.set_xlabel("Training Policy")
    ax.set_ylabel("Redox Energy")
    plt.xticks(rotation=45, ha='right')
    
    plt.savefig(os.path.join(output_dir, "fig2_energy_violin.png"))
    plt.savefig(os.path.join(output_dir, "fig2_energy_violin.pdf"))
    plt.close()

def plot_bar_chart(df, output_dir, descending_order, color_map):
    """Creates a bar chart sorted in descending order of the mean with std dev error bars."""
    fig, ax = plt.subplots()
    
    sns.barplot(
        data=df, 
        x="Policy", 
        y="Redox Energy", 
        order=descending_order,  # Forces the descending sort
        palette=color_map,       # Ensures colors match Fig 1 perfectly
        errorbar="sd", 
        capsize=0.1, 
        err_kws={'linewidth': 1.5, 'color': 'black'},
        edgecolor='black',
        linewidth=1.5,
        ax=ax
    )
    
    ax.set_title("Average Discovered Redox Energy (± Standard Deviation)")
    ax.set_xlabel("Training Policy")
    ax.set_ylabel("Average Redox Energy")
    plt.xticks(rotation=45, ha='right')
    
    plt.savefig(os.path.join(output_dir, "fig3_energy_bar_std.png"))
    plt.savefig(os.path.join(output_dir, "fig3_energy_bar_std.pdf"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Parse logs and plot energy metrics.")
    parser.add_argument("base_dir", type=str, help="Path to the master directory containing the exp_* folders")
    
    args = parser.parse_args()
    base_dir = os.path.abspath(args.base_dir)
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist.")
        return

    print("Parsing logs...")
    experiment_data = parse_logs(base_dir)
    
    if not experiment_data:
        print("No data parsed. Exiting.")
        return
        
    for exp, vals in sorted(experiment_data.items()):
        print(f"Parsed {len(vals)} simulations for {exp}")

    # Convert to pandas DataFrame for Seaborn
    df = prepare_dataframe(experiment_data)

    # 1. Establish strict color mapping based on alphabetical order
    sorted_policies = sorted(experiment_data.keys())
    color_map = {policy: ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)] for i, policy in enumerate(sorted_policies)}

    # 2. Calculate the descending order of the means for Seaborn plotting
    descending_mean_order = df.groupby('Policy')['Redox Energy'].mean().sort_values(ascending=False).index.tolist()

    output_dir = os.path.join(base_dir, "analysis_results")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating plots in {output_dir}...")
    setup_academic_plot_style()
    
    # Pass the color mapping and sorting orders to the plot functions
    plot_frequency_distributions(experiment_data, output_dir, color_map)
    print("- Frequency Distribution plotted (fig1)")
    
    plot_violin(df, output_dir, descending_mean_order, color_map)
    print("- Violin Plot (Descending) plotted (fig2)")
    
    plot_bar_chart(df, output_dir, descending_mean_order, color_map)
    print("- Bar Chart with StdDev (Descending) plotted (fig3)")
    
    print("\nSuccess! All academic figures generated.")

if __name__ == "__main__":
    main()