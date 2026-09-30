# Incremental retraining trades result quality for training time

*10 seeds per arm (seeds 0–9). Regenerate with:*

```bash
python analyze.py $(ls -d runs/dev-snapshot_*_n500_k32_w8_ts0.05_s[0-9]) --out analysis/baseline-10seeds
python make_figures.py --analysis analysis/baseline-10seeds --runs 'runs/dev-snapshot_*_n500_k32_w8_ts0.05_s[0-9]' --out figures/baseline-10seeds
```

## Draft text

While incremental training appears to remove the cost of repeated retraining, it trades away
result quality for that time. We searched a pool of 3,286 molecules for the highest oxidation
potential, retraining after each batch of 32 completed simulations. In the first configuration,
*full retraining*, the model is retrained from scratch on all accumulated data at every round. In
the second, *incremental retraining*, the model is warm-started from the previous weights and
trained only on the 32 newest simulations. Both arms evaluate the same budget of 500 molecules and
draw from identical precomputed simulation results, so any difference comes from the training
policy alone. Search quality is measured as recall of the pool's true top 1% (33 molecules), which
is known exactly because every candidate was precomputed.

Full retraining performs consistently better, reaching **61.8% ± 5.6%** average recall over ten
seeds, indicating that the model reliably follows a trajectory toward high-potential molecules.
Incremental retraining reaches only **33.3% ± 12.9%** (Welch's t-test p < 0.001), and its more than
twofold larger spread (individual runs span 15.2%–63.6%, versus 51.5%–69.7% for full) shows that
its discoveries depend heavily on chance. Full retraining wins in nine of ten paired seeds and ties
in the tenth. Training time alone is **516 ± 53 s** for full versus **229 ± 24 s** for incremental:
incremental halves the training budget over the first 500 molecules, yet produces markedly poorer
results. The saving does not shorten the campaign, because at realistic simulation cost training
accounts for only ~2% of it (time to solution: 2.74 h ± 0.13 for full versus 2.77 h ± 0.07 for
incremental).

We further measured prediction error (RMSE, eV) of the final model on three subsets: the molecules
acquired since the previous training round ("newest batch"), the initial batch, and the whole pool
(Table X). The incrementally-trained model is least inaccurate on the batch it has just seen
(0.419 ± 0.134) and degrades on older data (first batch 0.608 ± 0.126; whole pool 0.582 ± 0.127),
whereas full retraining is uniformly accurate and is in fact most accurate on the oldest data
(0.288 ± 0.037). Incremental updates therefore neither fit the newest data as well as full
retraining nor retain earlier knowledge, and the model deteriorates globally after a few cycles,
losing the ability to rank regions of the search space it has already explored.

## Table X

Final-model RMSE (eV), mean ± standard deviation over ten seeds. "Newest batch" is defined
identically for both arms as the molecules acquired since the previous training round.

| Training | Newest batch | First batch | Whole pool |
|---|---|---|---|
| Full | 0.325 ± 0.040 | 0.288 ± 0.037 | 0.319 ± 0.008 |
| Incremental | 0.419 ± 0.134 | 0.608 ± 0.126 | 0.582 ± 0.127 |

## Figure X

`figures/baseline-10seeds/fig1_recall_paper.pdf`

> **Figure X.** Recall of the pool's true top 1% against simulation budget, averaged over ten
> seeds; shaded bands show ±1 standard deviation. The two policies are indistinguishable until
> roughly 200 molecules, after which full retraining pulls away and ends at 62% versus 33%.

## Supporting numbers

| Metric (10 seeds) | Full | Incremental |
|---|---|---|
| Recall of true top 1% | 61.8% ± 5.6% | 33.3% ± 12.9% |
| Recall range across seeds | 51.5%–69.7% | 15.2%–63.6% |
| Paired seed wins | 9 | 0 (1 tie) |
| Best potential found (eV) | 7.537 ± 0.030 | 7.452 ± 0.046 |
| Training time (s) | 516 ± 53 | 229 ± 24 |
| Time to solution at true simulation cost (h) | 2.74 ± 0.13 | 2.77 ± 0.07 |
| Training rounds | 11.3 ± 0.5 | 12.9 ± 0.3 |

Welch's t-test on recall: p < 0.001 (Mann–Whitney: p = 0.0005).

Per-seed recall (%), seeds 0–9:

| Seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Full | 60.6 | 63.6 | 51.5 | 60.6 | 63.6 | 66.7 | 69.7 | 66.7 | 54.5 | 60.6 |
| Incremental | 39.4 | 36.4 | 15.2 | 30.3 | 63.6 | 27.3 | 27.3 | 27.3 | 27.3 | 39.4 |

## Method details worth stating

- Property: adiabatic oxidation potential (+1) in acetonitrile, MOPAC PM7, computed in advance for
  every candidate so both arms replay identical results.
- Model: ensemble of four message-passing neural networks (NFP), mean for prediction and spread for
  the expected-improvement acquisition function.
- Schedule: 32 randomly-chosen starting molecules, retraining every 32 completed simulations, and a
  re-ranking inference pass every 8.
- Campaigns ran with simulations replayed at 0.05× their recorded duration to save wall-clock time;
  all reported times are recomputed at the full recorded cost.

## Caveats to keep in the text

- One seed of ten tied (63.6% for both arms), so the claim is that incremental is much worse on
  average and far more variable, not that it always loses.
- The null result on time depends on the ratio of training cost to simulation cost. With cheaper
  simulations or a larger ensemble, incremental would save meaningful time; with DFT-scale
  simulations the saving would be even smaller than measured here.
- Full ran 11.3 training rounds on average and incremental 12.9, because incremental's faster
  training was blocked by in-flight inference less often. Worth a footnote so the unequal round
  counts are not mistaken for truncated data.
- These results use one acquisition function (expected improvement) and one model family; the
  mechanism (narrow, acquisition-biased batches) should generalize, but we have not tested that.

---

# Replay recovers full retraining's quality at bounded cost

*15 seeds for replay 0 and 16, 10 seeds for the other settings. Regenerate with:*

```bash
python plot_summary.py --out figures/summary        # frontier, frontier_walltime, discovery_curves
python plot_replay_sweep.py --out figures/replay-sweep
```



## Figures

- `figures/summary/frontier_walltime.pdf` — recall against measured training time (cited as fig y above)
- `figures/summary/frontier.pdf` — the same points against training examples processed, which is
  deterministic and monotone; preferred if the main text needs a hardware-independent cost axis
- `figures/summary/discovery_curves.pdf` — recall against simulation budget, one line per replay size
- `figures/replay-sweep/replay_sweep.pdf` — recall and training time against replay size, side by side

## Caveats to keep in the text

- Seed counts are uneven: 15 for R = 0 and R = 16, 10 for the others. State this, or run five more
  seeds of each remaining setting to make it uniform.
- R = 256 was also run (10 seeds): 63.3% ± 8.5 recall at 381 s. It adds nothing over R = 128, so it is
  omitted from the figures.
- A gentler fine-tuning variant (learning rate 1e-4, 8 epochs instead of 1e-3 and 32) was tested as an
  alternative explanation for the failure of pure incremental training. It performed *worse*
  (**18.2% ± 14.2%**, 5 seeds), so the deficit is not over-fitting to the newest batch; the model needs
  more representative data, not gentler updates.
- Replay samples uniformly from all earlier molecules. Prioritized or diversity-aware sampling was not
  tested and may reach the same quality with a smaller buffer.
- "Examples processed" counts training-set size per round, not gradient steps: each round runs up to
  32 epochs with early stopping, so the number of passes varies slightly between runs.
