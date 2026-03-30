import argparse
import logging
import sys
from pathlib import Path
import os

from parsl import Config, HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.launchers import WrappedLauncher
from colmena.task_server import ParslTaskServer
from colmena.queue import PipeQueues
from colmena.thinker import BaseThinker, agent

from examol.store.db.memory import InMemoryStore
from examol.score.rdkit import make_knn_model, RDKitScorer
from examol.solution import SingleFidelityActiveLearning
from examol.store.recipes import RedoxEnergy
from examol.start.fast import RandomStarter
from examol.select.baseline import GreedySelector


class OfflineTrainingThinker(BaseThinker):
    """A minimal Thinker that just runs training on incrementally larger datasets."""
    
    def __init__(self, queues, run_dir, recipe, scorer, models, database, strategy):
        super().__init__(queues)
        self.run_dir = run_dir
        self.recipe = recipe
        self.scorer = scorer
        self.models = models
        self.database = database
        self.strategy = strategy
        self.train_sizes = list(range(10, 101, 10))  # 10, 20, 30... 100
        
        # Setup robust logging without overwriting the read-only BaseThinker property
        handler = logging.FileHandler(self.run_dir / f'offline_training_{strategy}.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    @agent(startup=True)
    def offline_training_loop(self):
        try:
            self.logger.info("=== Starting Offline Training Script ===")
            
            # 1. Read database ONCE and extract valid records
            valid_records = []
            for record in self.database.iterate_over_records():
                if self.recipe.lookup(record) is not None:
                    valid_records.append(record)
                    
            self.logger.info(f"Successfully loaded {len(valid_records)} valid, pre-simulated records from database.")

            # 2. Iterate through the compounding batch sizes
            for size in self.train_sizes:
                if size > len(valid_records):
                    self.logger.warning(f"Requested {size} molecules, but only have {len(valid_records)}. Stopping early.")
                    break

                # Clip the entries
                train_set = valid_records[:size]
                self.logger.info(f"\n--- Triggering training for {size} molecules ---")

                # Transform data for the model
                try:
                    train_inputs = self.scorer.transform_inputs(train_set)
                    train_outputs = self.scorer.transform_outputs(train_set, self.recipe)
                except Exception as e:
                    self.logger.error(f"Failed to transform inputs/outputs for size {size}. Error: {e}", exc_info=True)
                    continue

                # Determine which Parsl method to target
                method_name = 'retrain' if self.strategy == 'same' else f'retrain_{size}'
                topic_name = 'train'

                # 3. Submit model for training
                for model_id, model in enumerate(self.models[0]):
                    try:
                        model_msg = self.scorer.prepare_message(model, training=True)
                        self.queues.send_inputs(
                            model_msg, train_inputs, train_outputs,
                            method=method_name,
                            topic=topic_name,
                            task_info={'model_id': model_id, 'size': size}
                        )
                        self.logger.info(f"Submitted task to method: {method_name}")
                    except Exception as e:
                        self.logger.error(f"Failed to submit task for model {model_id}, size {size}. Error: {e}", exc_info=True)

                # 4. Wait for results and update model
                for _ in range(len(self.models[0])):
                    result = self.queues.get_result(topic=topic_name)
                    
                    if result.success:
                        model_id = result.task_info['model_id']
                        # Resolve proxy if using proxystore, otherwise use value directly
                        model_msg = result.value
                        if hasattr(model_msg, '__proxy__'):
                            from proxystore.proxy import extract
                            model_msg = extract(model_msg)
                            
                        self.models[0][model_id] = self.scorer.update(self.models[0][model_id], model_msg)
                        self.logger.info(f"Successfully updated model {model_id} with {size} molecules. Time running: {result.time_running:.2f}s")
                    else:
                        self.logger.error(f"Training task failed for size {size}! Failure info: {result.failure_info}")

            self.logger.info("=== All offline training steps completed successfully. ===")

        except Exception as e:
            self.logger.error(f"Critical error in training loop: {e}", exc_info=True)
        finally:
            self.done.set()


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=str, help='Path to the existing run directory')
    parser.add_argument('--strategy', choices=['same', 'distinct'], default='same', help='Resource allocation strategy')
    args = parser.parse_args()

    run_path = Path(args.run_dir).absolute()
    db_path = run_path / 'database.json'
    
    if not db_path.exists():
        print(f"ERROR: Database file not found at {db_path}")
        sys.exit(1)

    # Setup core ExaMol components
    recipe = RedoxEnergy(1, energy_config='mopac_pm7', solvent='acn')
    scorer = RDKitScorer(run_dir=os.getcwd())
    
    # We use the ActiveLearning class just to cleanly generate the base retrain function
    solution = SingleFidelityActiveLearning(
        starter=RandomStarter(), minimum_training_size=10, selector=GreedySelector(100, maximize=True),
        scorer=scorer, models=[[make_knn_model()]], num_to_run=100
    )
    base_retrain_func = [f for f in solution.generate_functions() if f.__name__ == 'retrain'][0]

    # Setup Parsl Executors and Methods
    executors = []
    methods = []
    
    if args.strategy == 'same':
        # STATEFUL: One executor pinned to cores 0-2
        label = 'train_same'
        executors.append(HighThroughputExecutor(
            label=label, max_workers_per_node=1,
            provider=LocalProvider(launcher=WrappedLauncher(f"taskset -c 0-2 perf stat -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses -o {run_path}/perf_same.data"))
        ))
        methods.append((base_retrain_func, {'executors': [label]}))
        
    elif args.strategy == 'distinct':
        # STATELESS: 10 executors, each pinned to a distinct 3-core block
        for i, size in enumerate(range(10, 101, 10)):
            core_start = i * 3
            core_end = core_start + 2
            label = f'train_{size}'
            
            executors.append(HighThroughputExecutor(
                label=label, max_workers_per_node=1,
                provider=LocalProvider(launcher=WrappedLauncher(f"taskset -c {core_start}-{core_end} perf stat -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses -o {run_path}/perf_{size}.data"))
            ))
            
            # Simple wrapper to create a unique function name for Colmena routing
            def make_wrapper(base_func, name):
                def wrapper(*args, **kwargs):
                    return base_func(*args, **kwargs)
                wrapper.__name__ = name
                return wrapper
                
            methods.append((make_wrapper(base_retrain_func, f'retrain_{size}'), {'executors': [label]}))

    # Initialize Task Server and Queues
    config = Config(executors=executors, run_dir=str(run_path / f'parsl-logs-{args.strategy}'))
    queues = PipeQueues(topics=['train'])
    doer = ParslTaskServer(queues=queues, methods=methods, config=config)
    store = InMemoryStore(db_path)

    # Run
    doer.start()
    try:
        thinker = OfflineTrainingThinker(
            queues=queues, run_dir=run_path, recipe=recipe, 
            scorer=scorer, models=solution.models, database=store, strategy=args.strategy
        )
        thinker.run()
    finally:
        queues.send_kill_signal()
        doer.join()