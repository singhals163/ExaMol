"""Summarize campaigns: time to solution, quality of what was found, and model quality/forgetting per training round

Usage: python analyze.py runs/<run-1> runs/<run-2> ... [--top-frac 0.01] [--out analysis]
"""
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def retime_at_true_cost(run_dir: Path, traj: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Re-time a campaign as if simulations had cost their full recorded runtime

    Campaigns are run with ``TIME_SCALE`` < 1 to save wall-clock time. Training rounds are triggered by
    molecule counts rather than elapsed time, so the sequence of decisions is unchanged by the scale;
    only the durations differ. This replays the observed sequence of molecules and training rounds
    through the same number of workers, charging each molecule its full recorded simulation time and
    each training/inference round its measured (unscaled) duration.

    Returns the trajectory with an added ``time_true`` column.
    """
    settings = json.loads((run_dir / 'run-settings.json').read_text())
    workers, ensemble = settings['workers'], settings['ensemble_size']
    sim_time = dict(zip(truth['key'], truth['sim_time']))  # Full recorded cost per molecule

    # Measured cost of each training round and of the inference that follows it
    rounds = load_jsonl(run_dir / 'training-rounds.json')
    train_costs = [r['runtime'] for r in rounds]
    infs = load_jsonl(run_dir / 'inference-results.json')
    infer_cost = float(np.median([r['time_running'] for r in infs])) if infs else 0.0
    retrain_every, num_initial = settings['retrain_every'], settings['num_initial']
    infer_every = settings.get('infer_every')

    def _hold_workers(duration: float):
        """Occupy `ensemble` workers for a learning task"""
        slots = np.argsort(free)[-ensemble:]
        free[slots] = free[slots].max() + duration

    free = np.zeros(workers)  # Time at which each worker is next free
    times, completed_since_train, completed_since_infer, round_id = [], 0, 0, 0
    for key in traj['key']:
        # Learning tasks hold `ensemble` workers for their duration before the next batch of molecules
        if (round_id == 0 and len(times) >= num_initial) or (round_id > 0 and completed_since_train >= retrain_every):
            if round_id < len(train_costs):
                _hold_workers(train_costs[round_id] + infer_cost)  # Training always ends with inference
                round_id += 1
                completed_since_train = completed_since_infer = 0
        elif infer_every is not None and round_id > 0 and completed_since_infer >= infer_every:
            _hold_workers(infer_cost)  # Re-ranking pass without retraining
            completed_since_infer = 0

        worker = int(np.argmin(free))
        free[worker] += sim_time.get(key, np.nan)
        times.append(free[worker])
        completed_since_train += 1
        completed_since_infer += 1

    traj = traj.copy()
    traj['time_true'] = sorted(times)  # Completion times, in the order molecules completed
    return traj


def pool_from_run(run_dir: Path) -> Path:
    """Get the pool used by a run from the search space settings the thinker cached"""
    settings = json.loads((run_dir / 'search-space' / 'settings.json').read_text())
    return Path(settings['paths'][0]).parent


def analyze_run(run_dir: Path, top_frac: float) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Returns: summary metrics, per-molecule trajectory, per-round model quality"""
    truth = pd.read_csv(pool_from_run(run_dir) / 'ground-truth.csv')
    value_by_smiles = dict(zip(truth['smiles'], truth['value']))
    value_by_key = dict(zip(truth['key'], truth['value']))
    k = max(1, int(round(top_frac * len(truth))))
    top_keys = set(truth.nlargest(k, 'value')['key'])

    # Timing of every task
    results = {topic: load_jsonl(run_dir / f'{topic}-results.json') for topic in ['simulation', 'train', 'inference']}
    t0 = min(r['time_created'] for rs in results.values() for r in rs)

    # Trajectory of completed molecules
    finished = sorted((r for r in results['simulation'] if r['task_info'].get('status') == 'finished'), key=lambda r: r['time_result_received'])
    traj = pd.DataFrame({
        'time': [r['time_result_received'] - t0 for r in finished],
        'key': [r['task_info']['key'] for r in finished],
        'value': [value_by_key[r['task_info']['key']] for r in finished],
    })
    traj['n'] = np.arange(1, len(traj) + 1)
    traj['best'] = traj['value'].cummax()
    traj['top_found'] = traj['key'].isin(top_keys).cumsum()
    traj['top_recall'] = traj['top_found'] / k

    traj = retime_at_true_cost(run_dir, traj, truth)

    # Share of worker time spent on each type of task
    busy = {topic: sum(r['time_running'] or 0 for r in rs) for topic, rs in results.items()}
    wall = traj['time'].iloc[-1] if len(traj) else np.nan

    # Model quality after each training round
    rounds = load_jsonl(run_dir / 'training-rounds.json')
    trained_by_round = {r['round']: set(r['trained_keys']) for r in rounds}
    key_by_smiles = dict(zip(truth['smiles'], truth['key']))
    quality = []
    seen = set()  # Keys used in any training round up to (and including) the current one
    last_round = 0
    pred_files = sorted((run_dir / 'predictions').glob('pass-*-chunk-000.npz'))
    if not pred_files:  # Runs made before predictions were versioned per inference pass
        pred_files = sorted((run_dir / 'predictions').glob('round-*-chunk-000.npz'))
    for pass_id, pred_file in enumerate(pred_files, start=1):
        parts = pred_file.name.split('-')
        rnd = int(parts[3]) if parts[0] == 'pass' else int(parts[1])
        latest = trained_by_round.get(rnd, set()) if rnd != last_round else set()  # Empty when no training since the last pass
        seen_before = set(seen)  # Used in training before this round
        seen |= latest
        last_round = rnd
        data = np.load(pred_file)
        smiles = list(data['smiles'])
        mean = data['predictions'][0].mean(axis=1)
        pred = pd.DataFrame({'key': [key_by_smiles[s] for s in smiles], 'pred': mean, 'true': [value_by_smiles[s] for s in smiles]})

        def _rmse(mask):
            sub = pred[mask]
            return float(np.sqrt(((sub['pred'] - sub['true']) ** 2).mean())) if len(sub) else np.nan

        unseen = ~pred['key'].isin(seen)
        model_top = set(pred.nlargest(k, 'pred')['key'])
        quality.append({
            'pass': pass_id,
            'round': rnd,
            'time': float(data['time']) - t0,
            'rmse_all': _rmse(slice(None)),
            'rmse_unseen': _rmse(unseen),  # Molecules never used in training
            'rmse_top': _rmse(pred['key'].isin(top_keys)),  # The true best molecules
            'rmse_old_train': _rmse(pred['key'].isin(seen_before)),  # Used in any earlier round: measures forgetting
            'rmse_latest_train': _rmse(pred['key'].isin(latest)),  # Trained on in this round
            'spearman_all': float(spearmanr(pred['pred'], pred['true']).statistic),
            'model_top_precision': len(model_top & top_keys) / k,  # How many of the model's top-k are truly top-k
        })
    quality = pd.DataFrame(quality)

    # Final-model error on the first batch and on the newest batch, defined identically for both policies
    # (the newest batch is what was acquired since the previous round, not what the round trained on)
    final_rmse = {}
    if rounds and len(pred_files) > 0:
        data = np.load(pred_files[-1])
        pred = pd.DataFrame({'key': [key_by_smiles[s] for s in data['smiles']],
                             'pred': data['predictions'][0].mean(axis=1),
                             'true': [value_by_smiles[s] for s in data['smiles']]})
        subsets = {'first_batch': set(rounds[0]['trained_keys'])}
        if len(rounds) > 1:
            subsets['newest_batch'] = set(rounds[-1]['trained_keys']) - set(rounds[-2]['trained_keys'])
        for name, keys in subsets.items():
            sub = pred[pred['key'].isin(keys)]
            final_rmse[f'final_rmse_{name}'] = float(np.sqrt(((sub['pred'] - sub['true']) ** 2).mean())) if len(sub) else np.nan

    summary = {
        'run': run_dir.name,
        'completed': len(traj),
        'time_to_solution_s': wall,
        'time_to_solution_true_s': traj['time_true'].iloc[-1] if len(traj) else np.nan,
        'top_k': k,
        'top_recall': traj['top_recall'].iloc[-1] if len(traj) else np.nan,
        'best_found': traj['best'].iloc[-1] if len(traj) else np.nan,
        'mean_top10_found': traj['value'].nlargest(10).mean() if len(traj) else np.nan,
        'recall_auc_by_n': traj['top_recall'].mean() if len(traj) else np.nan,  # Average recall over the campaign
        'train_rounds': len(rounds),
        'train_time_total_s': sum(r['runtime'] for r in rounds),
        'busy_sim_s': busy['simulation'],
        'busy_train_s': busy['train'],
        'busy_inference_s': busy['inference'],
        'final_rmse_all': quality['rmse_all'].iloc[-1] if len(quality) else np.nan,
        'final_rmse_old_train': quality['rmse_old_train'].iloc[-1] if len(quality) else np.nan,
        **final_rmse,
    }
    return summary, traj, quality


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--top-frac', type=float, default=0.01, help='Fraction of the pool counted as "top" molecules')
    parser.add_argument('--out', default='analysis')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)
    summaries, trajs, qualities = [], [], []
    for run in args.runs:
        summary, traj, quality = analyze_run(Path(run), args.top_frac)
        summaries.append(summary)
        trajs.append(traj.assign(run=summary['run']))
        qualities.append(quality.assign(run=summary['run']))

    summaries = pd.DataFrame(summaries).set_index('run')
    trajs = pd.concat(trajs)
    qualities = pd.concat(qualities)
    summaries.to_csv(out_dir / 'summary.csv')
    trajs.to_csv(out_dir / 'trajectories.csv', index=False)
    qualities.to_csv(out_dir / 'model-quality.csv', index=False)

    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    print(summaries.T.to_string())
    print()
    print(qualities.round(3).to_string(index=False))

    # Plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for name, traj in trajs.groupby('run'):
        axs[0, 0].plot(traj['n'], traj['top_recall'], label=name)
        axs[0, 1].plot(traj['time_true'] / 3600, traj['top_recall'], label=name)
        axs[0, 2].plot(traj['time_true'] / 3600, traj['n'], label=name)
    for name, quality in qualities.groupby('run'):
        axs[1, 0].plot(quality['pass'], quality['rmse_all'], marker='o', label=name)
        axs[1, 1].plot(quality['pass'], quality['rmse_old_train'], marker='o', label=name)
        axs[1, 2].plot(quality['pass'], quality['model_top_precision'], marker='o', label=name)
    labels = [('Molecules evaluated', f'Recall of top {args.top_frac:.0%}'), ('Simulation time at true cost (h)', f'Recall of top {args.top_frac:.0%}'),
              ('Simulation time at true cost (h)', 'Molecules evaluated'), ('Inference pass', 'RMSE, whole pool (eV)'),
              ('Inference pass', 'RMSE, molecules from earlier rounds (eV)'), ('Inference pass', 'Precision of model top-k')]
    for ax, (xl, yl) in zip(axs.flat, labels):
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / 'comparison.png', dpi=120)
    print(f'\nWrote results to {out_dir}')
