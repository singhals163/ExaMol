import os
import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd

# Apply Grade-A paper aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif', 
    'font.size': 14, 
    'axes.titlesize': 18, 
    'axes.labelsize': 16,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'figure.dpi': 300,
    'savefig.dpi': 300
})

def extract_energies(run_dir, max_mols=80):
    """Helper function to parse the log file and return up to max_mols energies."""
    # Check directly in the dir, or inside a 'run' subdir
    log_file = os.path.join(run_dir, "run_sequence.log")
    if not os.path.exists(log_file):
        log_file = os.path.join(run_dir, "run", "run_sequence.log")
        
    if not os.path.exists(log_file):
        print(f"Error: Log file not found at {log_file}")
        return []

    print(f"Parsing {log_file}...")
    energies = []
    
    # Parse the sequential true energies
    with open(log_file, 'r') as f:
        for line in f:
            if "Simulation result" in line:
                try:
                    # Extract the value from "Simulation result | key: XYZ, value: 6.322"
                    val_str = line.split("value:")[-1].strip()
                    # Handle potential list formatting like "[6.322]"
                    val_str = val_str.replace('[', '').replace(']', '')
                    energies.append(float(val_str))
                    
                    # Stop exactly at the requested limit
                    if len(energies) == max_mols:
                        break
                except ValueError:
                    continue
                    
    return energies

def print_summary_statistics(energies_0, energies_18):
    """Calculates and prints the mean, median, std dev, etc. for both runs."""
    if not energies_0 or not energies_18:
        return

    df = pd.DataFrame({
        'Baseline (0 Loops)': energies_0,
        'Active Learning (18 Loops)': energies_18
    })

    print("\n" + "="*55)
    print(" SUMMARY STATISTICS (First 80 Molecules)")
    print("="*55)
    # .describe() automatically calculates count, mean, std, min, 25%, 50% (median), 75%, max
    print(df.describe().round(4).to_string())
    print("="*55 + "\n")


def plot_discovery_trajectory(dir_0, dir_18):
    energies_0 = extract_energies(dir_0, max_mols=100)
    energies_18 = extract_energies(dir_18, max_mols=100)

    if not energies_0 or not energies_18:
        print("Missing simulation results in one or both log files.")
        sys.exit(1)

    print(f"Found {len(energies_0)} molecules for Baseline and {len(energies_18)} for Active Learning. Generating plot...")

    # Prepare DataFrames and calculate running maximums
    df_0 = pd.DataFrame({'energy': energies_0})
    df_0['running_max'] = df_0['energy'].cummax()
    
    df_18 = pd.DataFrame({'energy': energies_18})
    df_18['running_max'] = df_18['energy'].cummax()

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Baseline (0 Loops) in Blue
    ax.scatter(df_0.index, df_0['energy'], color='#1f77b4', alpha=0.4, label='Baseline Random Simulation', s=25)
    ax.plot(df_0.index, df_0['running_max'], color='#1f77b4', linewidth=3.5, label='Baseline Best Discovered Molecule Trend', linestyle='--')

    # Plot Active Learning (18 Loops) in Red
    ax.scatter(df_18.index, df_18['energy'], color='#d62728', alpha=0.4, label='AI Simulation (batchSize = 10)', s=25)
    ax.plot(df_18.index, df_18['running_max'], color='#d62728', linewidth=3.5, label='AI Best Discovered Molecule Trend')

    # ax.set_title('Active Learning vs Baseline Discovery Trajectory (First 80 Mols)')
    ax.set_xlabel('Molecules Simulated')
    ax.set_ylabel('Redox Energy')
    
    # Organize legend
    ax.legend(loc='lower right', frameon=True, edgecolor='black', fontsize=11)
    
    plt.tight_layout()
    
    # Save to the current working directory
    output_path = 'exploration_comparison.png'
    plt.savefig(output_path)
    plt.close()
    
    print(f"Success! Plot saved to: {output_path}")

def plot_violin_distribution(energies_0, energies_18):
    if not energies_0 or not energies_18:
        return

    print("Generating violin distribution plot...")
    
    data = [energies_0, energies_18]
    labels = ['Baseline\n(0 Loops)', 'Active Learning\n(18 Loops)']

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create violin plot
    parts = ax.violinplot(data, showmeans=True, showextrema=True, showmedians=True)
    
    # Color customization to match the trajectory plot
    parts['bodies'][0].set_facecolor('#1f77b4') # Blue for baseline
    parts['bodies'][1].set_facecolor('#d62728') # Red for AL
    
    for pc in parts['bodies']:
        pc.set_alpha(0.6)
        
    # Customize edges and lines for clean aesthetic
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
        vp = parts[partname]
        vp.set_edgecolor('#333333')
        vp.set_linewidth(1.5)

    # Adding a legend-like label for means/medians
    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('white')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel('True Redox Energy')
    ax.set_title('Distribution of Discovered Energies (First 80 Mols)')
    
    plt.tight_layout()
    
    output_path = 'exploration_violin.png'
    plt.savefig(output_path)
    plt.close()
    
    print(f"Success! Violin plot saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot and compare active learning discovery trajectories.")
    parser.add_argument("dir_0", type=str, help="Path to the directory containing the baseline (0 loop) run_sequence.log")
    parser.add_argument("dir_18", type=str, help="Path to the directory containing the Active Learning (18 loop) run_sequence.log")
    
    args = parser.parse_args()
    
    # Ensure the paths are absolute or correctly resolved
    dir_0_path = os.path.abspath(args.dir_0)
    dir_18_path = os.path.abspath(args.dir_18)
    
    # Extract once to save disk I/O
    e_0 = extract_energies(dir_0_path, max_mols=80)
    e_18 = extract_energies(dir_18_path, max_mols=80)

    # Generate stats and both plots
    print_summary_statistics(e_0, e_18)
    plot_discovery_trajectory(e_0, e_18)
    plot_violin_distribution(e_0, e_18)