"""Compare incremental retraining warm-started from a checkpoint against always reinitializing from
scratch, at replay=0.

Both arms use the same seeds, pool, schedule, and worker count (only ``RETRAIN_FROM_SCRATCH`` differs),
so any difference in recall or training time comes from the weight-initialization policy alone.

Usage: python plot_scratch_vs_checkpoint.py --analysis analysis/<name> --out figures/<name>
"""
import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {'checkpoint': '#0072B2', 'scratch': '#D55E00'}
LABELS = {'checkpoint': 'Warm-started from checkpoint', 'scratch': 'Reinitialized from scratch'}


def arm_of(run: str) -> str:
    return 'scratch' if '_scratch_' in f'_{run}_' else 'checkpoint'


def band(ax, x, ys, arm, label=None):
    mean, sd = np.nanmean(ys, axis=0), np.nanstd(ys, axis=0)
    ax.fill_between(x, mean - sd, mean + sd, color=COLORS[arm], alpha=0.15, linewidth=0)
    ax.plot(x, mean, color=COLORS[arm], label=label or LABELS[arm])
    return mean


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--analysis', default='analysis/scratch-vs-checkpoint')
    parser.add_argument('--out', default='figures/scratch-vs-checkpoint')
    args = parser.parse_args()

    analysis, out = Path(args.analysis), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    traj = pd.read_csv(analysis / 'trajectories.csv')
    quality = pd.read_csv(analysis / 'model-quality.csv')
    summary = pd.read_csv(analysis / 'summary.csv')
    for frame in (traj, quality, summary):
        frame['arm'] = frame['run'].map(arm_of)
    summary['seed'] = summary['run'].str.extract(r'_s(\d+)$').astype(int)

    # Headline numbers
    agg = summary.groupby('arm').agg(
        recall_mean=('top_recall', 'mean'), recall_std=('top_recall', 'std'),
        train_time_mean=('train_time_total_s', 'mean'), train_time_std=('train_time_total_s', 'std'),
        time_true_mean=('time_to_solution_true_s', 'mean'), time_true_std=('time_to_solution_true_s', 'std'),
        best_mean=('best_found', 'mean'), best_std=('best_found', 'std'),
        n=('run', 'count'),
    )
    speedup = agg.loc['checkpoint', 'train_time_mean'] / agg.loc['scratch', 'train_time_mean']
    recall_drop = agg.loc['checkpoint', 'recall_mean'] - agg.loc['scratch', 'recall_mean']
    agg.to_csv(out / 'headline.csv')
    print(agg.to_string())
    print(f"\nTraining-time speedup (checkpoint / scratch): {speedup:.2f}x")
    print(f"Recall drop (checkpoint - scratch): {recall_drop:+.1%}")

    # Per-seed paired comparison
    wide = summary.pivot(index='seed', columns='arm', values='top_recall')
    wide.to_csv(out / 'paired_recall.csv')

    # Figure: recall trajectory
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    grid = np.arange(1, int(traj['n'].max()) + 1)
    for arm, group in traj.groupby('arm'):
        ys = [np.interp(grid, g['n'], g['top_recall']) for _, g in group.groupby('run')]
        mean = band(axs[0], grid, ys, arm)
        axs[0].annotate(f'{mean[-1]:.0%}', (grid[-1], mean[-1]), xytext=(4, -2), textcoords='offset points',
                        color=COLORS[arm], fontsize=8, fontweight='bold', annotation_clip=False)
    axs[0].set_xlabel('Molecules evaluated')
    axs[0].set_ylabel('Recall of true top 1%')
    axs[0].set_ylim(0, None)
    axs[0].legend(loc='upper left', frameon=False, fontsize=8)
    axs[0].set_title('Search quality')

    # Figure: training time per round (model-quality.csv has no runtime column; read training-rounds.json directly)
    rows = []
    for run_dir in sorted((analysis.parent.parent / 'runs').glob('*')):
        if not (run_dir / 'training-rounds.json').exists():
            continue
        if run_dir.name not in set(summary['run']):
            continue
        arm = arm_of(run_dir.name)
        for line in (run_dir / 'training-rounds.json').read_text().splitlines():
            r = json.loads(line)
            rows.append({'run': run_dir.name, 'arm': arm, 'round': r['round'], 'runtime': r['runtime']})
    rows = pd.DataFrame(rows)
    if len(rows):
        for arm, group in rows.groupby('arm'):
            rounds = sorted(group['round'].unique())
            ys = [[g[g['round'] == r]['runtime'].mean() for r in rounds] for _, g in group.groupby('run')]
            band(axs[1], rounds, ys, arm)
        axs[1].set_xlabel('Training round')
        axs[1].set_ylabel('Training time per round (s)')
        axs[1].set_title('Training cost')
        axs[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out / 'comparison.png', dpi=150)
    plt.close(fig)
    print(f'\nWrote figures and tables to {out}')
