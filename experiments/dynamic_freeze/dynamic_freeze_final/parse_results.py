import json
import argparse
import os

def parse_simulations(input_path):
    # Ensure the file exists
    if not os.path.isfile(input_path):
        print(f"Error: The file '{input_path}' does not exist.")
        return

    # Get the directory of the input file to save the output file in the same location
    target_dir = os.path.dirname(os.path.abspath(input_path))
    output_path = os.path.join(target_dir, "run_sequence.log")

    processed_count = 0

    # Helper function to process a single JSON record
    def process_record(data, outfile):
        nonlocal processed_count
        
        # The fix: access the nested 'task_info' dictionary first
        task_info = data.get("task_info", {})
        
        # Check status and result inside task_info
        if task_info.get("status") == "finished" and "result" in task_info:
            task_key = task_info.get("key")
            result_list = task_info.get("result")
            
            # Extract the first item from the result array
            result_value = result_list[0] if (result_list and isinstance(result_list, list)) else None
            
            if task_key is not None and result_value is not None:
                log_line = f"Simulation result | key: {task_key} value: {result_value}\n"
                outfile.write(log_line)
                processed_count += 1

    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            # Check the first character to determine if it's JSONL or a JSON array
            first_char = infile.read(1).strip()
            infile.seek(0) # Reset to start of file
            
            if first_char == '{':
                # Parse as JSONL (one JSON object per line)
                for line in infile:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        process_record(data, outfile)
                    except json.JSONDecodeError:
                        continue # Skip malformed lines quietly
            
            elif first_char == '[':
                # Fallback: Parse as a standard JSON array
                full_data = json.load(infile)
                if isinstance(full_data, list):
                    for data in full_data:
                        if isinstance(data, dict):
                            process_record(data, outfile)

        print(f"Successfully extracted {processed_count} results.")
        print(f"Output saved to: {output_path}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse simulation results from a JSON/JSONL file.")
    parser.add_argument("filepath", help="Path to the simulation-results.json file")
    
    args = parser.parse_args()
    parse_simulations(args.filepath)