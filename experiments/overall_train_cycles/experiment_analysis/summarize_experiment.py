import argparse
import re
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def find_summary_files(base_dir: Path) -> List[Path]:
    """Finds all analysis_summary.txt files within the base directory."""
    print(f"Searching for summary files in: {base_dir.resolve()}...")
    files = list(base_dir.glob("**/analysis_summary.txt"))
    print(f"Found {len(files)} summary files.")
    return files

def parse_summary_file(file_path: Path) -> List[Dict[str, Any]]:
    """Parses the 'Overall Task Performance Summary' table from a summary file."""
    
    # Extract training_loops and reading_id from the path
    # MODIFIED: Adjusted path indices to account for the extra 'run' directory.
    # e.g., .../training_inf/reading_4/run/analysis_summary.txt
    try:
        # Before: file_path.parts[-3] -> was 'reading_4'
        # After: file_path.parts[-4] -> is now 'training_inf'
        training_loops = file_path.parts[-4].split('_')[1]
        
        # Before: file_path.parts[-2] -> was 'run'
        # After: file_path.parts[-3] -> is now 'reading_4'
        reading_id = int(file_path.parts[-3].split('_')[1])
    except (IndexError, ValueError):
        print(f"Warning: Could not parse training/reading info from path: {file_path}")
        return []

    # Regex to capture the data rows from the summary table
    pattern = re.compile(
        r"^(?P<method>\w+)\s+(?P<count>\d+)\s+(?P<total_compute>[\d.]+)\s+(?P<avg_compute>[\d.]+)"
    )

    records = []
    in_summary_section = False
    with file_path.open('r') as f:
        for line in f:
            if "Overall Task Performance Summary" in line:
                in_summary_section = True
                continue
            
            if in_summary_section:
                match = pattern.match(line)
                if match:
                    data = match.groupdict()
                    records.append({
                        'training_loops': training_loops,
                        'reading_id': reading_id,
                        'method': data['method'],
                        'count': int(data['count']),
                        'total_compute_s': float(data['total_compute']),
                        'avg_compute_s': float(data['avg_compute']),
                    })
    return records

def create_charts(df: pd.DataFrame, output_dir: Path):
    """Generates and saves summary charts from the aggregated data."""
    if df.empty:
        print("DataFrame is empty, skipping chart creation.")
        return

    # Set plot style
    sns.set_theme(style="whitegrid")

    # --- Chart 1: Average Compute Time per Method ---
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df, x='training_loops', y='avg_compute_s', hue='method', palette='viridis')
    plt.title('Average Compute Time vs. Training Loops', fontsize=16)
    plt.xlabel('Number of Training Loops', fontsize=12)
    plt.ylabel('Average Compute Time (s)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    avg_time_chart_path = output_dir / "summary_avg_time.png"
    plt.savefig(avg_time_chart_path)
    plt.close()
    print(f"Saved average time chart to: {avg_time_chart_path}")

    # --- Chart 2: Average Task Count per Method ---
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df, x='training_loops', y='count', hue='method', palette='plasma')
    plt.title('Average Task Count vs. Training Loops', fontsize=16)
    plt.xlabel('Number of Training Loops', fontsize=12)
    plt.ylabel('Average Number of Tasks per Run', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    task_count_chart_path = output_dir / "summary_task_counts.png"
    plt.savefig(task_count_chart_path)
    plt.close()
    print(f"Saved task count chart to: {task_count_chart_path}")


def main(base_dir: Path, output_dir: Path):
    """Main function to find, parse, aggregate, and summarize results."""
    
    # 1. Find and parse all individual summary files
    summary_files = find_summary_files(base_dir)
    if not summary_files:
        print("No summary files found. Exiting.")
        return
        
    all_records = []
    for file in summary_files:
        all_records.extend(parse_summary_file(file))

    # 2. Convert to DataFrame and aggregate results
    if not all_records:
        print("Error: No valid records were parsed from the summary files. Cannot generate report.")
        return
        
    df = pd.DataFrame(all_records)
    
    # Aggregate by training_loops and method, calculating the mean across readings
    agg_df = df.groupby(['training_loops', 'method']).mean().reset_index()
    # Ensure a logical sort order for the 'training_loops' column
    agg_df['sort_key'] = agg_df['training_loops'].apply(lambda x: float('inf') if x == 'inf' else int(x))
    agg_df = agg_df.sort_values(by=['sort_key', 'method']).drop(columns=['sort_key'])


    # 3. Generate summary charts
    create_charts(agg_df, output_dir)

    # 4. Write the final Markdown report
    report_path = output_dir / "experiment_summary.md"
    with report_path.open('w') as f:
        f.write("# ExaMol Experiment Summary\n\n")
        f.write(f"This report summarizes the results from the `{base_dir.name}` directory.\n\n")
        
        f.write("## Average Task Performance Across Readings\n\n")
        f.write("The table below shows the average performance for each task type, grouped by the number of training loops performed.\n\n")
        f.write(agg_df.to_markdown(index=False))
        
        f.write("\n\n## Average Compute Time vs. Training Loops\n\n")
        f.write("![Average Compute Time](summary_avg_time.png)\n\n")

        f.write("## Average Task Count vs. Training Loops\n\n")
        f.write("![Average Task Count](summary_task_counts.png)\n\n")

    print(f"\nExperiment summary report created at:\n{report_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize an entire ExaMol experiment from individual run analyses.")
    parser.add_argument("base_directory", type=str,
                        help="Path to the base experiment directory (e.g., 'experiment_results').")
    parser.add_argument("-o", "--output_directory", type=str, default=".",
                        help="Directory to save the final summary report and charts (defaults to current directory).")
    
    args = parser.parse_args()
    main(Path(args.base_directory), Path(args.output_directory))

