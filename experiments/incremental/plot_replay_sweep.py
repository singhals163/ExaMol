"""Plot search quality and training cost against replay-buffer size

The sweep runs from replay=0 (pure incremental) to replay=all (every earlier molecule, i.e. the same
data full retraining sees, but warm-started). Full retraining from scratch is drawn as a reference band.

Usage: python plot_replay_sweep.py [--out figures/replay-sweep]
"""
import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLOR = '#E69F00'      # Incremental family
COLOR_FULL = '#0072B2'  # Full retraining reference


def recall_and_cost(run_dir: Path, top_keys: set, k: int) -> tuple[float, float] | None:
    sims = [json.loads(line) for line in (run_dir / 'simulation-results.json').read_text().splitlines()]
    keys = {r['task_info']['key'] for r in sims if r['task_info'].get('status') == 'finished'}
    if len(keys) < 495:  # Still running
        return None
    rounds = [json.loads(line) for line in (run_dir / 'training-rounds.json').read_text().splitlines()]
    return len(keys & top_keys) / k, sum(r['runtime'] for r in rounds)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pool', default='pools/dev-snapshot')
    parser.add_argument('--out', default='figures/replay-sweep')
    args = parser.parse_args()

    truth = pd.read_csv(Path(args.pool) / 'ground-truth.csv')
    k = max(1, int(round(0.01 * len(truth))))
    top_keys = set(truth.nlargest(k, 'value')['key'])

    rows = []
    for run_dir in sorted(Path('runs').glob('dev-snapshot_*_n500_k32_w8_ts0.05*')):
        name = run_dir.name
        if 'lr' in name or 'e8' in name:  # Fine-tuning variants belong to a different comparison
            continue
        if not (run_dir / 'training-rounds.json').exists():
            continue
        if 'incremental' in name:
            replay = int(name.split('_r')[1].split('_')[0]) if '_r' in name else 0
        else:
            replay = None  # Full retraining from scratch
        result = recall_and_cost(run_dir, top_keys, k)
        if result is None:
            continue
        rows.append({'run': name, 'replay': replay, 'recall': result[0], 'train_s': result[1]})

    data = pd.DataFrame(rows)
    data.to_csv(Path(args.out).parent / 'replay-sweep.csv', index=False) if Path(args.out).parent.exists() else None
    full = data[data['replay'].isna()]
    sweep = data[data['replay'].notna()].copy()
    # REPLAY=100000 means "every earlier molecule"; label that point "all"
    all_sentinel = 100000
    order = sorted(sweep['replay'].unique())
    positions = {r: i for i, r in enumerate(order)}
    labels = [('all' if r >= all_sentinel else str(int(r))) for r in order]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': 0.25, 'figure.dpi': 200, 'savefig.bbox': 'tight',
                         'axes.axisbelow': True, 'lines.linewidth': 1.8}):
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
        for ax, col, ylabel in [(axs[0], 'recall', 'Recall of top 1%'), (axs[1], 'train_s', 'Training time (s)')]:
            stats = sweep.groupby('replay')[col].agg(['mean', 'std', 'count'])
            x = [positions[r] for r in stats.index]
            ax.errorbar(x, stats['mean'], yerr=stats['std'].fillna(0), color=COLOR, marker='o',
                        markersize=5, capsize=3, label='Incremental + replay')
            if len(full):
                ax.axhline(full[col].mean(), color=COLOR_FULL, linestyle='--', label='Full retraining')
                ax.fill_between([-0.4, len(order) - 0.6], full[col].mean() - full[col].std(),
                                full[col].mean() + full[col].std(), color=COLOR_FULL, alpha=0.12, linewidth=0)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(labels)
            ax.set_xlim(-0.4, len(order) - 0.6)
            ax.set_xlabel('Replayed molecules per round')
            ax.set_ylabel(ylabel)
        axs[0].set_ylim(0, None)
        axs[0].legend(frameon=False, loc='lower right', fontsize=9)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(out / f'replay_sweep.{ext}')
        plt.close(fig)

    print(sweep.groupby('replay')[['recall', 'train_s']].agg(['mean', 'std', 'count']).round(3).to_string())
    if len(full):
        print(f"\nfull retraining: recall {full['recall'].mean():.3f} ± {full['recall'].std():.3f}, "
              f"train {full['train_s'].mean():.0f} ± {full['train_s'].std():.0f} s (n={len(full)})")
    print(f'\nWrote {out}/replay_sweep.png')
