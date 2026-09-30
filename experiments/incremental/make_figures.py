"""Build candidate figures comparing full and incremental retraining

Usage: python make_figures.py --analysis analysis/seeds0-4 --runs 'runs/dev-snapshot_*_n500_k32_*' --out figures
"""
import json
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Okabe-Ito: colorblind-safe pair
COLORS = {'full': '#0072B2', 'incremental': '#E69F00'}
LABELS = {'full': 'Full retraining', 'incremental': 'Incremental retraining'}

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.alpha': 0.25, 'grid.linewidth': 0.5, 'figure.dpi': 200, 'savefig.bbox': 'tight',
    'axes.axisbelow': True, 'lines.linewidth': 2,
})


def mode_of(run: str) -> str:
    return 'incremental' if 'incremental' in run else 'full'


def band(ax, x, ys, mode, label=None):
    """Mean line with a +/- 1 standard deviation band across seeds"""
    mean, sd = np.nanmean(ys, axis=0), np.nanstd(ys, axis=0)
    ax.fill_between(x, mean - sd, mean + sd, color=COLORS[mode], alpha=0.15, linewidth=0)
    ax.plot(x, mean, color=COLORS[mode], label=label or LABELS[mode])
    return mean


def fig_recall_vs_budget(traj, out):
    """Figure 1: recall of the true top 1% against molecules evaluated"""
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    grid = np.arange(1, int(traj['n'].max()) + 1)
    for mode, group in traj.groupby('mode'):
        ys = [np.interp(grid, g['n'], g['top_recall']) for _, g in group.groupby('run')]
        mean = band(ax, grid, ys, mode)
        ax.annotate(f'{mean[-1]:.0%}', (grid[-1], mean[-1]), xytext=(4, -2), textcoords='offset points',
                    color=COLORS[mode], fontsize=8, fontweight='bold')
    ax.set_xlabel('Molecules evaluated')
    ax.set_ylabel('Recall of true top 1%')
    ax.set_ylim(0, None)
    ax.legend(loc='upper left', frameon=False)
    fig.savefig(out / 'fig1_recall_vs_budget.png')
    plt.close(fig)


def fig_recall_paper(traj, out):
    """Figure 1, sized for print: small canvas with large type, so the labels match ~9pt body text

    Saved at the size it should appear in the paper (2.6 in wide, one-column-third). Place it without
    scaling; the vector PDF keeps the type crisp.
    """
    with plt.rc_context({'font.size': 11, 'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
                         'legend.fontsize': 10, 'lines.linewidth': 1.8}):
        fig, ax = plt.subplots(figsize=(2.6, 2.1))
        grid = np.arange(1, int(traj['n'].max()) + 1)
        for mode, group in traj.groupby('mode'):
            ys = np.array([np.interp(grid, g['n'], g['top_recall']) for _, g in group.groupby('run')])
            mean, sd = ys.mean(axis=0), ys.std(axis=0)
            ax.fill_between(grid, mean - sd, mean + sd, color=COLORS[mode], alpha=0.13, linewidth=0)
            ax.plot(grid, mean, color=COLORS[mode], label=LABELS[mode].replace(' retraining', ''))
            ax.annotate(f'{mean[-1]:.0%}', (grid[-1], mean[-1]), xytext=(3, -3), textcoords='offset points',
                        color=COLORS[mode], fontsize=10, fontweight='bold', annotation_clip=False)
        ax.set_xlabel('Molecules evaluated')
        ax.set_ylabel('Recall of top 1%')
        ax.set_ylim(0, None)
        ax.set_xticks([0, 250, 500])
        ax.set_xlim(0, grid[-1] * 1.02)  # Leave room for the end labels
        ax.legend(loc='upper left', frameon=False, handlelength=1.4, borderpad=0.1, labelspacing=0.25)
        for ext in ('png', 'pdf'):
            fig.savefig(out / f'fig1_recall_paper.{ext}')
        plt.close(fig)


def fig_model_quality(quality, out):
    """Figure 2: model error against training round, whole pool and earlier-round molecules"""
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.1), sharex=True)
    panels = [('rmse_all', 'RMSE, whole pool (eV)', 'Model accuracy'),
              ('rmse_old_train', 'RMSE, earlier rounds (eV)', 'Memory of earlier data')]
    for ax, (col, ylabel, title) in zip(axs, panels):
        for mode, group in quality.groupby('mode'):
            rounds = sorted(group['round'].unique())
            ys = [[g[g['round'] == r][col].mean() for r in rounds] for _, g in group.groupby('run')]
            band(ax, rounds, ys, mode)
        ax.set_xlabel('Training round')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axs[0].legend(loc='upper right', frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'fig2_model_quality.png')
    plt.close(fig)


def fig_paired_seeds(summary, out):
    """Figure 3: per-seed recall, paired by seed"""
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    wide = summary.pivot(index='seed', columns='mode', values='top_recall')
    for seed, row in wide.iterrows():
        ax.plot([0, 1], [row['full'], row['incremental']], color='#999999', linewidth=0.8, zorder=1)
    for i, mode in enumerate(['full', 'incremental']):
        ax.scatter([i] * len(wide), wide[mode], color=COLORS[mode], s=42, zorder=2,
                   edgecolor='white', linewidth=1.2, label=LABELS[mode])
        ax.scatter([i], [wide[mode].mean()], marker='_', s=420, color=COLORS[mode], linewidth=2.5, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Full', 'Incremental'])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel('Recall of true top 1%')
    ax.set_ylim(0, None)
    fig.savefig(out / 'fig3_paired_seeds.png')
    plt.close(fig)


def fig_forgetting_gap(quality, out):
    """Figure 4: for each mode, error on the newest batch vs on earlier rounds"""
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
    for ax, mode in zip(axs, ['full', 'incremental']):
        group = quality[quality['mode'] == mode]
        rounds = sorted(group['round'].unique())
        for col, style, label in [('rmse_latest_train', '-', 'Newest batch'), ('rmse_old_train', '--', 'Earlier rounds')]:
            ys = [[g[g['round'] == r][col].mean() for r in rounds] for _, g in group.groupby('run')]
            mean = np.nanmean(ys, axis=0)
            ax.plot(rounds, mean, style, color=COLORS[mode], label=label)
        ax.set_title(LABELS[mode])
        ax.set_xlabel('Training round')
        ax.legend(frameon=False)
    axs[0].set_ylabel('RMSE (eV)')
    fig.tight_layout()
    fig.savefig(out / 'fig4_forgetting_gap.png')
    plt.close(fig)


def fig_best_found(traj, out):
    """Figure 5: best oxidation potential found against molecules evaluated"""
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    grid = np.arange(1, int(traj['n'].max()) + 1)
    for mode, group in traj.groupby('mode'):
        ys = [np.interp(grid, g['n'], g['best']) for _, g in group.groupby('run')]
        band(ax, grid, ys, mode)
    ax.set_xlabel('Molecules evaluated')
    ax.set_ylabel('Best potential found (eV)')
    ax.legend(loc='lower right', frameon=False)
    fig.savefig(out / 'fig5_best_found.png')
    plt.close(fig)


def fig_training_cost(run_dirs, out):
    """Figure 6: training time per round, and its share of the campaign"""
    rows = []
    for run_dir in run_dirs:
        mode = mode_of(run_dir.name)
        for line in (run_dir / 'training-rounds.json').read_text().splitlines():
            r = json.loads(line)
            rows.append({'run': run_dir.name, 'mode': mode, 'round': r['round'],
                         'runtime': r['runtime'], 'trained_on': r['num_trained_on']})
    rows = pd.DataFrame(rows)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for mode, group in rows.groupby('mode'):
        rounds = sorted(group['round'].unique())
        ys = [[g[g['round'] == r]['runtime'].mean() for r in rounds] for _, g in group.groupby('run')]
        band(axs[0], rounds, ys, mode)
        sizes = [group[group['round'] == r]['trained_on'].mean() for r in rounds]
        axs[1].plot(rounds, sizes, color=COLORS[mode], label=LABELS[mode])
    axs[0].set_xlabel('Training round')
    axs[0].set_ylabel('Training time per round (s)')
    axs[0].set_title('Cost per round')
    axs[0].legend(frameon=False)
    axs[1].set_xlabel('Training round')
    axs[1].set_ylabel('Molecules trained on')
    axs[1].set_title('Training set size')
    fig.tight_layout()
    fig.savefig(out / 'fig6_training_cost.png')
    plt.close(fig)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--analysis', default='analysis/seeds0-4')
    parser.add_argument('--runs', default='runs/dev-snapshot_*_n500_k32_*')
    parser.add_argument('--out', default='figures')
    args = parser.parse_args()

    analysis, out = Path(args.analysis), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    traj = pd.read_csv(analysis / 'trajectories.csv')
    quality = pd.read_csv(analysis / 'model-quality.csv')
    summary = pd.read_csv(analysis / 'summary.csv')
    for frame in (traj, quality, summary):
        frame['mode'] = frame['run'].map(mode_of)
    summary['seed'] = summary['run'].str.extract(r'_s(\d+)$').astype(int)
    run_dirs = [p for p in map(Path, sorted(glob(args.runs))) if (p / 'training-rounds.json').exists()]

    fig_recall_vs_budget(traj, out)
    fig_recall_paper(traj, out)
    fig_model_quality(quality, out)
    fig_paired_seeds(summary, out)
    fig_forgetting_gap(quality, out)
    fig_best_found(traj, out)
    fig_training_cost(run_dirs, out)
    print(f'Wrote {len(list(out.glob("*.png")))} figures to {out}')
