"""Specification of the optimization problem"""
import json
import os
from pathlib import Path
import shutil
import sys

from parsl import Config, HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.launchers import WrappedLauncher
from proxystore.store import Store
from proxystore.connectors.file import FileConnector

from examol.reporting.markdown import MarkdownReporter
from examol.score.rdkit import make_knn_model, RDKitScorer
from examol.simulate.ase import ASESimulator
from examol.solution import SingleFidelityActiveLearning
from examol.start.fast import RandomStarter
from examol.steer.single import SingleStepThinker
from examol.store.recipes import RedoxEnergy
from examol.select.baseline import GreedySelector
from examol.specify import ExaMolSpecification

# Load run configuration dynamically
env_config_path = os.environ.get('EXAMOL_CONFIG_PATH')

if env_config_path:
    config_path = Path(env_config_path).resolve()
    run_base_dir = config_path.parent  # Isolate outputs to the config's directory
else:
    run_base_dir = Path().absolute()
    config_path = run_base_dir / 'run_config.json'

with open(config_path, 'r') as f:
    run_config = json.load(f)

# Parameters extracted from config
num_random: int = run_config.get('num_random', 2)
num_total: int = run_config.get('num_total', 8)

# Delete the old run
run_dir = run_base_dir / 'run'
if run_dir.is_dir():
    shutil.rmtree(run_dir)

# Make the recipe
recipe = RedoxEnergy(1, energy_config='mopac_pm7', solvent='acn')

# Make the scorer
pipeline = make_knn_model()
scorer = RDKitScorer(run_dir=run_dir)

# Define the tools needed to solve the problem
solution = SingleFidelityActiveLearning(
    starter=RandomStarter(),
    minimum_training_size=num_random,
    selector=GreedySelector(num_total, maximize=True),
    scorer=scorer,
    models=[[pipeline]],
    num_to_run=num_total,
)

# Mark how we report outcomes
reporter = MarkdownReporter()

# Make the parsl (compute) and proxystore (optional data fabric) configuration
is_mac = sys.platform == 'darwin'

sim_affinity = run_config.get('sim_affinity', 'none')
train_score_affinity = run_config.get('train_score_affinity', 'none')
num_cores = run_config.get('num_workers', 1)

config = Config(
    executors=[
        HighThroughputExecutor(
            label='simulation', 
            max_workers_per_node=num_cores,
            cpu_affinity=sim_affinity,
            provider=LocalProvider(
                launcher=WrappedLauncher(f"perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-load-misses,LLC-load-misses -I 1000 -o {run_dir}/perf_stat_simulation.data"),  # Use the default launcher
            ), 
        ),
        HighThroughputExecutor(
            label='learning', 
            max_workers_per_node=1, 
            cores_per_worker=num_cores,
            cpu_affinity=train_score_affinity,
            provider=LocalProvider(
                launcher=WrappedLauncher(f"perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-load-misses,LLC-load-misses -I 1000 -o {run_dir}/perf_stat_learning.data"),  # Use the default launcher
            ), 
        ),
    ],
    run_dir=str((run_base_dir / 'parsl-logs')),
)
store = Store(name='file', connector=FileConnector(store_dir=str(run_base_dir / 'proxystore')), metrics=True)

spec = ExaMolSpecification(
    database=(run_dir / 'database.json'),
    recipes=[recipe],
    search_space=[(Path().absolute() / 'search_space.smi')], # Assumes search_space.smi is also copied to the subdirectories
    solution=solution,
    simulator=ASESimulator(scratch_dir=(run_dir / 'tmp'), clean_after_run=False),
    thinker=SingleStepThinker,
    thinker_options={'run_config': run_config},
    compute_config=config,
    proxystore=store,
    reporters=[reporter],
    run_dir=run_dir,
)