"""Specification of the optimization problem"""
from pathlib import Path
import shutil
import sys
import os

from parsl import Config, HighThroughputExecutor
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

# Parameters you may want to configure
num_random: int = 2  # Number of randomly-selected molecules to run
num_total: int = 8  # Total number of molecules to run

run_dir_str = os.environ.get('EXAMOL_RUN_DIR')
max_loops_str = os.environ.get('EXAMOL_MAX_LOOPS')

if not run_dir_str or not max_loops_str:
    raise ValueError("Env variables EXAMOL_RUN_DIR and EXAMOL_MAX_LOOPS must be set.")

my_path = Path(run_dir_str)
if max_loops_str.lower() == 'inf':
    max_loops = -1
else:
    max_loops = int(max_loops_str)

# # Get my path. We'll want to provide everything as absolute paths, as they are relative to this file
# my_path = Path().absolute()

# # Delete the old run
run_dir = my_path / 'run'
if run_dir.is_dir():
    shutil.rmtree(run_dir)

# Make the recipe
recipe = RedoxEnergy(1, energy_config='mopac_pm7', solvent='acn')

# Make the scorer
pipeline = make_knn_model()
scorer = RDKitScorer()

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
config = Config(
    executors=[HighThroughputExecutor(max_workers_per_node=1)],
    run_dir=str((my_path / 'parsl-logs')),
)
store = Store(name='file', connector=FileConnector(store_dir=str(my_path / 'proxystore')), metrics=True)

spec = ExaMolSpecification(
    database=(run_dir / 'database.json'),
    recipes=[recipe],
    search_space=[('/users/vthurime/experiment_overall_profiling/ExaMol/examples/redoxmers/search_space.smi')],
    solution=solution,
    simulator=ASESimulator(scratch_dir=(run_dir / 'tmp'), clean_after_run=False),
    thinker=SingleStepThinker,
    compute_config=config,
    proxystore=store,
    reporters=[reporter],
    run_dir=run_dir,
    max_loops=max_loops,
)
