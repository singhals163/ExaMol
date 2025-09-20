import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

def format_timestamp(ts: float) -> str:
    """Converts a UNIX timestamp to a human-readable string."""
    if ts is None or ts == 0:
        return "N/A"
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

def parse_report_counts(report_path: Path) -> dict:
    """Extracts task counts from the report.md file."""
    counts = {
        'simulation': 0,
        'train': 0,
        'inference': 0
    }
    if not report_path.exists():
        return counts

    pattern = re.compile(r"\|\s*(simulation|train|inference)\s*\|\s*(\d+)\s*\|")
    
    with report_path.open('r') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                task_type, count = match.groups()
                counts[task_type.strip()] = int(count)
    return counts

def analyze_json_log(file_path: Path) -> list:
    """Analyzes a JSON Lines file (e.g., simulation-results.json)."""
    if not file_path.exists():
        return []

    records = []
    with file_path.open('r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                
                time_created = data.get("time_created", 0)
                time_received = data.get("time_result_received", 0)
                time_compute_start = data.get("time_compute_started", 0)
                time_compute_end = data.get("time_compute_ended", 0)

                total_duration = time_received - time_created if time_received and time_created else 0
                compute_duration = time_compute_end - time_compute_start if time_compute_end and time_compute_start else 0
                
                total_duration = max(0, total_duration)
                compute_duration = max(0, compute_duration)

                records.append({
                    'id': data.get('task_id', 'N/A'),
                    'method': data.get('method', 'Unknown'),
                    'success': data.get('success', False),
                    'start_time': format_timestamp(time_created),
                    'compute_start': format_timestamp(time_compute_start),
                    'compute_end': format_timestamp(time_compute_end),
                    'end_time': format_timestamp(time_received),
                    'total_duration_s': total_duration,
                    'compute_duration_s': compute_duration,
                    'overhead_s': total_duration - compute_duration,
                })
            except (json.JSONDecodeError, TypeError):
                print(f"Warning: Skipping malformed line {line_num} in {file_path.name}")
                continue
    return records

def write_task_details(f, task_name: str, tasks: List[Dict[str, Any]]):
    """Writes a detailed, formatted section for a list of tasks."""
    f.write("\n" + "="*25 + "\n")
    f.write(f"  {task_name.upper()} TASK ANALYSIS\n")
    f.write("="*25 + "\n\n")

    if not tasks:
        f.write("No tasks found.\n")
        return

    # --- Overall Summary Statistics for the task type ---
    total_durations = [t['total_duration_s'] for t in tasks]
    compute_durations = [t['compute_duration_s'] for t in tasks]
    
    f.write(f"Total Tasks: {len(tasks)}\n")
    f.write(f"Overall Total Duration (s): Min: {min(total_durations):.3f}, Max: {max(total_durations):.3f}, Avg: {sum(total_durations)/len(tasks):.3f}\n")
    f.write(f"Overall Compute Duration (s): Min: {min(compute_durations):.3f}, Max: {max(compute_durations):.3f}, Avg: {sum(compute_durations)/len(tasks):.3f}\n")

    # --- MODIFICATION: Add breakdown for simulation methods ---
    if task_name == 'Simulation':
        methods = {
            'optimize_structure': [t for t in tasks if t['method'] == 'optimize_structure'],
            'compute_energy': [t for t in tasks if t['method'] == 'compute_energy']
        }
        for method_name, method_tasks in methods.items():
            if method_tasks:
                f.write(f"\n--- Breakdown for '{method_name}' ---\n")
                comp_durs = [t['compute_duration_s'] for t in method_tasks]
                tot_durs = [t['total_duration_s'] for t in method_tasks]
                f.write(f"Total Tasks: {len(method_tasks)}\n")
                f.write(f"Total Duration (s): Min: {min(tot_durs):.3f}, Max: {max(tot_durs):.3f}, Avg: {sum(tot_durs)/len(tot_durs):.3f}\n")
                f.write(f"Compute Duration (s): Min: {min(comp_durs):.3f}, Max: {max(comp_durs):.3f}, Avg: {sum(comp_durs)/len(comp_durs):.3f}\n")

    f.write("\n--- Individual Task Log ---\n")
    header = f"{'Task ID':<38}{'Method':<20}{'Success':<9}{'Start Time':<23}{'End Time':<23}{'Total (s)':>12}{'Compute (s)':>14}\n"
    f.write(header)
    f.write('-' * len(header) + '\n')
    for task in tasks:
        f.write(
            f"{task['id']:<38}"
            f"{task['method']:<20}"
            f"{str(task['success']):<9}"
            f"{task['start_time']:<23}"
            f"{task['end_time']:<23}"
            f"{task['total_duration_s']:>12.3f}"
            f"{task['compute_duration_s']:>14.3f}\n"
        )

def main(run_dir: Path):
    """Main function to orchestrate the analysis."""
    if not run_dir.is_dir():
        print(f"Error: Directory not found at '{run_dir}'")
        return

    output_file = run_dir / 'analysis_summary.txt'

    report_counts = parse_report_counts(run_dir / 'report.md')
    simulation_tasks = analyze_json_log(run_dir / 'simulation-results.json')
    training_tasks = analyze_json_log(run_dir / 'train-results.json')
    inference_tasks = analyze_json_log(run_dir / 'inference-results.json')

    with output_file.open('w') as f:
        f.write("========================================\n")
        f.write("      ExaMol Run Analysis Summary\n")
        f.write("========================================\n")
        f.write(f"Analyzed Directory: {run_dir.resolve()}\n")
        f.write(f"Report Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("--- Task Count Verification ---\n")
        for task_type in ['simulation', 'train', 'inference']:
            report_count = report_counts.get(task_type, 0)
            json_count = len(locals().get(f"{task_type}_tasks", []))
            status = "OK" if report_count == json_count else "MISMATCH"
            f.write(f"{task_type.capitalize():<12}: Report.md -> {report_count:<4} | JSON Lines -> {json_count:<4} | Status: {status}\n")

        write_task_details(f, 'Simulation', simulation_tasks)
        write_task_details(f, 'Training', training_tasks)
        write_task_details(f, 'Inference', inference_tasks)

        # --- MODIFICATION: Add final overall summary section ---
        f.write("\n\n" + "="*35 + "\n")
        f.write("  Overall Task Performance Summary\n")
        f.write("="*35 + "\n\n")

        all_tasks = simulation_tasks + training_tasks + inference_tasks
        tasks_by_method = {}
        for task in all_tasks:
            method = task['method']
            if method not in tasks_by_method:
                tasks_by_method[method] = []
            tasks_by_method[method].append(task)
        
        header = f"{'Method':<25}{'Count':>10}{'Total Compute (s)':>20}{'Average Compute (s)':>22}\n"
        f.write(header)
        f.write('-' * len(header) + '\n')

        summary_methods = ['optimize_structure', 'compute_energy', 'retrain', 'score']
        for method in summary_methods:
            if method in tasks_by_method:
                method_tasks = tasks_by_method[method]
                count = len(method_tasks)
                total_compute = sum(t['compute_duration_s'] for t in method_tasks)
                avg_compute = total_compute / count if count > 0 else 0
                f.write(f"{method:<25}{count:>10}{total_compute:>20.3f}{avg_compute:>22.3f}\n")
            else:
                f.write(f"{method:<25}{0:>10}{0.0:>20.3f}{0.0:>22.3f}\n")


    print(f"\nAnalysis complete. Summary written to:\n{output_file.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the output files of an ExaMol run.")
    parser.add_argument("run_directory", type=str,
                        help="Path to the ExaMol run directory to be analyzed.")
    
    args = parser.parse_args()
    main(Path(args.run_directory))

