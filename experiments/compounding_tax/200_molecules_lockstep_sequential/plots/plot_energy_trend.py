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

def plot_discovery_trajectory(run_dir):
    log_file = os.path.join(run_dir, "run_sequence.log")
    
    if not os.path.exists(log_file):
        print(f"Error: Log file not found at {log_file}")
        sys.exit(1)

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
                except ValueError:
                    continue

    if not energies:
        print("No simulation results found in the log file.")
        sys.exit(1)

    print(f"Found {len(energies)} simulated molecules. Generating plot...")

    df = pd.DataFrame({'energy': energies})
    
    # Calculate the running maximum (the best molecule found up to step N)
    df['running_max'] = df['energy'].cummax()

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot every individual evaluation as a faint dot
    ax.scatter(df.index, df['energy'], color='gray', alpha=0.4, label='Simulated Candidate', s=25)
    
    # Plot the running maximum as a bold crimson line
    ax.plot(df.index, df['running_max'], color='#d62728', linewidth=3.5, label='Best Discovered (Running Max)')

    ax.set_title('Active Learning Discovery Trajectory')
    ax.set_xlabel('Molecules Simulated (Sequential)')
    ax.set_ylabel('True Redox Energy')
    ax.legend(loc='lower right', frameon=True, edgecolor='black')
    
    plt.tight_layout()
    
    # Save to the specific run directory
    output_path = os.path.join(run_dir, 'discovery_trajectory.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Success! Plot saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the active learning discovery trajectory from a run_sequence.log file.")
    parser.add_argument("run_dir", type=str, help="Path to the directory containing run_sequence.log")
    
    args = parser.parse_args()
    
    # Ensure the path is absolute or correctly resolved
    target_dir = os.path.abspath(args.run_dir)
    plot_discovery_trajectory(target_dir)