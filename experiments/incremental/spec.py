"""Active learning campaign comparing full and incremental retraining

Simulations are replayed from precomputed MOPAC results (see ``experiments/precompute``) so that
every campaign sees identical ground truth. All tasks (simulation, training, inference) share one
pool of workers, so time spent training is time not spent simulating.

Configure with environment variables, then run ``examol run spec.py:spec`` from this directory:

    MODE            full | incremental (default: full)
    SEED            Random seed for the initial selection (default: 0)
    POOL            Name of the pool under ``pools/`` (default: dev-snapshot)
    NUM_TO_RUN      Simulation budget: number of molecules to evaluate (default: 500)
    NUM_INITIAL     Number of randomly-selected molecules before the first training (default: 64)
    RETRAIN_EVERY   Number of new molecules between training rounds (default: 64)
    INFER_EVERY     Number of new molecules between inference passes, which re-rank the search space without
                    retraining. Unset means inference only runs after training
    WORKERS         Number of workers shared by all tasks (default: 8)
    ENSEMBLE        Number of models in the ensemble (default: 4)
    EPOCHS          Maximum training epochs per round (default: 32)
    LR              Learning rate for training (default: 1e-3)
    REPLAY          Incremental only: how many earlier molecules to mix into each round (default: 0)
    TIME_SCALE      Multiplier on recorded simulation times (default: 1)
    RUN_NAME        Name of the run directory under ``runs/`` (default: derived from settings)
"""
import json
import os
import random
import shutil
from pathlib import Path

from parsl import Config, HighThroughputExecutor
from proxystore.connectors.file import FileConnector
from proxystore.store import Store

from examol.score.nfp import NFPScorer, make_simple_network
from examol.select.bayes import ExpectedImprovement
from examol.simulate.replay import ReplaySimulator
from examol.solution import SingleFidelityActiveLearning
from examol.specify import ExaMolSpecification
from examol.start.base import Starter
from examol.steer.incremental import IncrementalThinker
from examol.store.recipes import RedoxEnergy

# Settings
mode = os.environ.get('MODE', 'full')
assert mode in ('full', 'incremental'), f'Unknown mode: {mode}'
incremental = mode == 'incremental'
seed = int(os.environ.get('SEED', 0))
pool = os.environ.get('POOL', 'dev-snapshot')
num_to_run = int(os.environ.get('NUM_TO_RUN', 500))
num_initial = int(os.environ.get('NUM_INITIAL', 64))
retrain_every = int(os.environ.get('RETRAIN_EVERY', 64))
infer_every = int(os.environ['INFER_EVERY']) if os.environ.get('INFER_EVERY') else None
workers = int(os.environ.get('WORKERS', 8))
ensemble_size = int(os.environ.get('ENSEMBLE', 4))
epochs = int(os.environ.get('EPOCHS', 32))
learning_rate = float(os.environ.get('LR', 1e-3))
replay_size = int(os.environ.get('REPLAY', 0))
time_scale = float(os.environ.get('TIME_SCALE', 1))

my_path = Path().absolute()  # ExaMol runs specs with exec, so __file__ is unavailable. Run from this directory
pool_dir = my_path / 'pools' / pool
run_name = os.environ.get('RUN_NAME', f'{pool}_{mode}_n{num_to_run}_k{retrain_every}_w{workers}_ts{time_scale:g}'
                          + (f'_r{replay_size}' if replay_size else '') + (f'_lr{learning_rate:g}' if learning_rate != 1e-3 else '')
                          + (f'_e{epochs}' if epochs != 32 else '') + f'_s{seed}')
run_dir = my_path / 'runs' / run_name
if run_dir.is_dir():
    shutil.rmtree(run_dir)
run_dir.mkdir(parents=True)
(run_dir / 'run-settings.json').write_text(json.dumps({
    'mode': mode, 'seed': seed, 'pool': pool, 'num_to_run': num_to_run, 'num_initial': num_initial,
    'retrain_every': retrain_every, 'infer_every': infer_every, 'workers': workers,
    'ensemble_size': ensemble_size, 'epochs': epochs, 'time_scale': time_scale,
    'learning_rate': learning_rate, 'replay_size': replay_size,
}, indent=2))


class SeededRandomStarter(Starter):
    """Select molecules at random with a fixed seed, so paired campaigns start from the same molecules"""

    def __init__(self, seed: int):
        super().__init__()
        self.seed = seed

    def _select(self, to_select: list[str], count: int) -> list[str]:
        return random.Random(self.seed).sample(sorted(to_select), count)


# Define the problem
recipe = RedoxEnergy(1, energy_config='mopac_pm7', solvent='acn')

# Ensemble of MPNNs. Separate model objects so warm-started members stay independent
models = [make_simple_network(atom_features=64, message_steps=4, output_layers=[128, 64]) for _ in range(ensemble_size)]
scorer = NFPScorer(retrain_from_scratch=not incremental)

solution = SingleFidelityActiveLearning(
    starter=SeededRandomStarter(seed),
    minimum_training_size=num_initial,
    selector=ExpectedImprovement(to_select=4 * retrain_every, maximize=True),  # Queue refills each inference pass
    scorer=scorer,
    models=[models],
    num_to_run=num_to_run,
    # The validation split (25%) must fill at least one batch, so the batch size has to fit the smallest
    # training round: the initial set on round 1, and each increment thereafter
    train_options=dict(num_epochs=epochs, batch_size=max(2, min(16, min(retrain_every, num_initial) // 4)), validation_split=0.25,
                       learning_rate=learning_rate, device_type='cpu'),
)

# One pool of workers for all tasks
config = Config(
    executors=[HighThroughputExecutor(label='htex', max_workers_per_node=workers, cpu_affinity='none')],
    run_dir=str(run_dir / 'parsl-logs'),
)

spec = ExaMolSpecification(
    database=run_dir / 'database.json',
    recipes=[recipe],
    search_space=[pool_dir / 'search-space.smi'],
    solution=solution,
    simulator=ReplaySimulator(pool_dir / 'results.jsonl', time_scale=time_scale),
    thinker=IncrementalThinker,
    thinker_options=dict(num_workers=workers, retrain_every=retrain_every, infer_every=infer_every, incremental=incremental,
                         replay_size=replay_size, replay_seed=seed),
    compute_config=config,
    proxystore=Store(name='file', connector=FileConnector(store_dir=str(run_dir / 'proxystore')), metrics=True),  # Keeps large messages out of the pipes
    run_dir=run_dir,
)
