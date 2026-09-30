# Incremental vs full retraining

Does retraining only on new data (constant cost per round) match retraining on everything
(cost growing with the database)? These campaigns answer that, and test a replay buffer as a fix.

## Layout

| Path | What it is |
|---|---|
| `spec.py` | ExaMol campaign, configured through environment variables (see its docstring) |
| `analyze.py` | Per-run metrics: recall, time to solution, model error, forgetting |
| `make_figures.py` | Baseline figures, including the print-sized recall figure |
| `plot_summary.py` | Cost/quality frontier and discovery curves across policies |
| `plot_replay_sweep.py` | Quality and cost against replay-buffer size |
| `make_pool.py` | Freezes precomputed results into a fixed search pool with ground truth |
| `run_campaigns.sh`, `run_until_complete.sh` | Run campaigns sequentially; the latter retries failures |
| `paper-section.md` | Write-up of both results with the numbers |
| `figures/`, `analysis/` | Generated figures and metric tables |

Campaign outputs (`runs/`, `logs/`) are not tracked: they run to several GB. The simulation data is
tracked in compressed form:

| File | What it holds |
|---|---|
| `../precompute/data/energies.csv` | Oxidation potential and runtime for all 10,000 attempted molecules (6,647 succeeded) |
| `../precompute/data/results.jsonl.gz` | Raw records: relaxed geometries, all energies, per-step timings |
| `pools/dev-snapshot/results.jsonl.gz` | The 3,286-molecule subset the campaigns replay |
| `pools/dev-snapshot/ground-truth.csv` | Key, SMILES, potential and simulation time per pool molecule |

Decompress with `gunzip -k` before running campaigns; `ReplaySimulator` reads the plain `.jsonl`.

## Reproducing

```bash
# 1. Precompute MOPAC oxidation potentials for a sample of the search space (~20 h on 20 cores)
python ../precompute/run_precompute.py --num 10000 --workers 20

# 2. Freeze the successful results into a pool
python make_pool.py dev-snapshot

# 3. Run campaigns (each ~12 min at TIME_SCALE=0.05)
POOL=dev-snapshot NUM_TO_RUN=500 NUM_INITIAL=32 RETRAIN_EVERY=32 INFER_EVERY=8 \
  WORKERS=8 ENSEMBLE=4 EPOCHS=32 TIME_SCALE=0.05 \
  ./run_campaigns.sh "MODE=full SEED=0" "MODE=incremental SEED=0" "MODE=incremental SEED=0 REPLAY=64"

# 4. Analyze and plot
python analyze.py runs/dev-snapshot_*_s[0-9] --out analysis/baseline
python plot_summary.py --out figures/summary
```

Simulations are replayed from the precomputed results (`examol/simulate/replay.py`), so every
campaign sees identical ground truth and differs only in its training policy. `TIME_SCALE` shrinks
the replayed duration to save wall-clock time; `analyze.py` re-times each campaign at the full
recorded simulation cost.
