"""Two summary figures across every training policy

1. Cost-quality frontier: search quality against training cost, one point per policy
2. Discovery curves: recall against simulation budget for selected policies

Usage: python plot_summary.py [--out figures/summary]
"""
import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Replay size is an ordered magnitude, so it gets a sequential ramp (viridis: perceptually uniform and
# colorblind-safe). Full retraining is the reference, drawn in neutral black.
SIZES = [0, 16, 32, 64, 128]
REFERENCE = '#000000'


MARKERS = {0: 'o', 16: 's', 32: '^', 64: 'D', 128: 'v', 'full': 'P'}


def size_colors() -> dict[int, tuple]:
    """One viridis step per replay size, trimmed to avoid the palest yellow on white"""
    ramp = plt.get_cmap('viridis')
    return {size: ramp(x) for size, x in zip(SIZES, np.linspace(0.05, 0.8, len(SIZES)))}


def shade(color, amount: float):
    """Blend a colour toward white (amount > 0) or black (amount < 0)"""
    rgb = np.array(matplotlib.colors.to_rgb(color))
    target = np.ones(3) if amount > 0 else np.zeros(3)
    return tuple(rgb + (target - rgb) * abs(amount))


def drop_outliers(values: pd.Series, n_sd: float = 2.5) -> pd.Series:
    """Flag runs more than n_sd standard deviations from their group's mean"""
    if len(values) < 4:
        return pd.Series(False, index=values.index)
    return (values - values.mean()).abs() > n_sd * values.std()


PLOT_STYLE = {
    'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.alpha': 0.25, 'grid.linewidth': 0.5, 'figure.dpi': 200, 'savefig.bbox': 'tight',
    'axes.axisbelow': True, 'lines.linewidth': 1.8,
}


def policy_of(name: str) -> str | None:
    """Label each run directory by training policy"""
    if 'k32' not in name or 'lr0.0001' in name:
        return None  # Pilot runs and the fine-tuning variant are not part of this comparison
    if 'full' in name:
        return 'full'
    if '_r' in name:
        size = int(name.split('_r')[1].split('_')[0])
        return f'replay-{size}' if size in SIZES else None
    return 'replay-0'


def collect(pool: Path) -> pd.DataFrame:
    """Recall, training cost and discovery trajectory for every run"""
    truth = pd.read_csv(pool / 'ground-truth.csv')
    k = max(1, int(round(0.01 * len(truth))))
    top_keys = set(truth.nlargest(k, 'value')['key'])

    rows = []
    for run_dir in sorted(Path('runs').glob('dev-snapshot_*')):
        policy = policy_of(run_dir.name)
        if policy is None or not (run_dir / 'training-rounds.json').exists():
            continue
        sims = [json.loads(line) for line in (run_dir / 'simulation-results.json').read_text().splitlines()]
        finished = sorted((r for r in sims if r['task_info'].get('status') == 'finished'), key=lambda r: r['time_result_received'])
        if len(finished) < 495:
            continue
        keys = [r['task_info']['key'] for r in finished]
        found = np.cumsum([key in top_keys for key in keys])
        rounds = [json.loads(line) for line in (run_dir / 'training-rounds.json').read_text().splitlines()]
        rows.append({'run': run_dir.name, 'policy': policy, 'recall': len(set(keys) & top_keys) / k,
                     'train_s': sum(r['runtime'] for r in rounds),
                     # Deterministic measure of training work: molecules fed to the model over the campaign
                     'examples': sum(r['num_trained_on'] + r.get('num_replayed', 0) for r in rounds),
                     'curve': found / k})
    return pd.DataFrame(rows)


def fig_frontier(data: pd.DataFrame, out: Path, cost: str = 'examples'):
    """Search quality against training cost, one point per policy with error bars on both axes

    ``cost`` selects the x-axis: 'examples' counts the molecules fed to the model over a campaign
    (deterministic), 'train_s' uses measured wall-clock time (noisy below ~32 replayed molecules).
    """
    label_x = {'examples': 'Training examples processed per campaign',
               'train_s': 'Training time per campaign (s)'}[cost]
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        colors = size_colors()
        replay = data[data['policy'].str.startswith('replay')].copy()
        replay['size'] = replay['policy'].str.split('-').str[1].astype(int)
        stats = replay.groupby('size').agg(recall=('recall', 'mean'), recall_sd=('recall', 'std'),
                                           cost=(cost, 'mean'), cost_sd=(cost, 'std')).sort_index()
        ax.plot(stats['cost'], stats['recall'] * 100, color='#999999', linewidth=1.2, zorder=1)  # Connects the sizes
        for size, row in stats.iterrows():
            ax.errorbar(row['cost'], row['recall'] * 100, xerr=row['cost_sd'], yerr=row['recall_sd'] * 100,
                        color=colors[size], marker='o', markersize=7, capsize=2, elinewidth=0.9, capthick=0.9,
                        zorder=2, label=f'Replay {size}' if size else 'Replay 0 (incremental)')
        group = data[data['policy'] == 'full']
        ax.errorbar(group[cost].mean(), group['recall'].mean() * 100, xerr=group[cost].std(),
                    yerr=group['recall'].std() * 100, color=REFERENCE, marker='s', markersize=7, capsize=2,
                    elinewidth=0.9, capthick=0.9, zorder=3, label='Full retraining')
        ax.set_xlabel(label_x)
        ax.set_ylabel('Recall of true top 1% (%)')
        ax.set_ylim(0, 100)
        ax.legend(frameon=False, fontsize=8, loc='lower right')
        suffix = '' if cost == 'examples' else '_walltime'
        for ext in ('png', 'pdf'):
            fig.savefig(out / f'frontier{suffix}.{ext}')
        plt.close(fig)


def fig_curves(data: pd.DataFrame, out: Path):
    """Recall against simulation budget for the policies worth contrasting"""
    colors = size_colors()
    selection = [(f'replay-{size}', colors[size], f'Replay {size}' if size else 'Replay 0 (incremental)')
                 for size in SIZES] + [('full', REFERENCE, 'Full retraining')]
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        for policy, color, label in selection:
            group = data[data['policy'] == policy]
            if not len(group):
                continue
            length = min(len(c) for c in group['curve'])
            curves = np.array([c[:length] for c in group['curve']])
            x = np.arange(1, length + 1)
            mean = curves.mean(axis=0)
            style = '--' if policy == 'full' else '-'
            ax.plot(x, mean * 100, style, color=color, label=label)
            ax.annotate(f'{mean[-1]:.0%}', (x[-1], mean[-1] * 100), xytext=(3, -3), textcoords='offset points',
                        color=color, fontsize=9, fontweight='bold', annotation_clip=False)
        ax.set_xlabel('Molecules evaluated')
        ax.set_ylabel('Recall of true top 1% (%)')
        ax.set_ylim(0, 100)
        ax.set_xlim(0, 530)
        ax.legend(frameon=False, fontsize=8, loc='upper left', labelspacing=0.3)
        for ext in ('png', 'pdf'):
            fig.savefig(out / f'discovery_curves.{ext}')
        plt.close(fig)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pool', default='pools/dev-snapshot')
    parser.add_argument('--out', default='figures/summary')
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = collect(Path(args.pool))
    fig_frontier(data, out, cost='examples')
    fig_frontier(data, out, cost='train_s')  # Wall-clock version for the supplement
    fig_curves(data, out)

    for policy, group in data.groupby('policy'):
        flagged = group[drop_outliers(group['recall'])]
        for _, row in flagged.iterrows():
            print(f"excluded from {policy} mean (>2.5 sd): {row['run']} recall {row['recall']:.3f}")

    summary = data.groupby('policy').agg(runs=('recall', 'size'), recall=('recall', 'mean'),
                                         recall_sd=('recall', 'std'), train_s=('train_s', 'mean'),
                                         examples=('examples', 'mean')).round(3)
    print(summary.to_string())
    data.drop(columns='curve').to_csv(out / 'policy-summary.csv', index=False)
    print(f'\nWrote figures to {out}')
