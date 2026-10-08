"""Idealized simulation time (at N-way parallelism) + measured training time = "training tax"

For each run: sum the recorded (real-world) simulation time of every molecule that finished during
the campaign, divide by `--cores` to get an idealized fully-parallel simulation time, then add the
actually-measured total training time. Training tax is the training share of that idealized total.

Usage: python compute_training_tax.py runs/<run-1> runs/<run-2> ... [--cores 16] [--out table.csv]
"""
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from analyze import load_jsonl, pool_from_run


def compute(run_dir: Path, cores: int) -> dict:
    truth = pd.read_csv(pool_from_run(run_dir) / 'ground-truth.csv')
    sim_time = dict(zip(truth['key'], truth['sim_time']))

    results = load_jsonl(run_dir / 'simulation-results.json')
    finished_keys = {r['task_info']['key'] for r in results if r['task_info'].get('status') == 'finished'}
    sim_total = sum(sim_time[k] for k in finished_keys)
    sim_at_cores = sim_total / cores

    rounds = load_jsonl(run_dir / 'training-rounds.json')
    train_total = sum(r['runtime'] for r in rounds)

    idealized_total = sim_at_cores + train_total
    return {
        'run': run_dir.name,
        'n_molecules': len(finished_keys),
        'sim_total_s': sim_total,
        f'sim_at_{cores}x_s': sim_at_cores,
        'train_total_s': train_total,
        'idealized_total_s': idealized_total,
        'training_tax': train_total / idealized_total,
    }


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--cores', type=int, default=16)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    rows = [compute(Path(r), args.cores) for r in args.runs]
    table = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(table.to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f'\nWrote {args.out}')
