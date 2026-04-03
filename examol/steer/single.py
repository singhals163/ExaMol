"""Single-objective and single-fidelity implementation of active learning. As easy as we get"""
import gzip
import json
import pickle as pkl
import shutil
import math
from functools import partial
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from colmena.proxy import get_store
from colmena.queue import ColmenaQueues
from colmena.thinker import event_responder, ResourceCounter, agent
from more_itertools import interleave_longest, batched
from proxystore.proxy import extract, Proxy
from proxystore.store import Store
from proxystore.store.utils import get_key

from .base import MoleculeThinker
from examol.solution import SingleFidelityActiveLearning
from ..score.base import Scorer
from ..store.db.base import MoleculeStore
from ..store.models import MoleculeRecord
from ..store.recipes import PropertyRecipe


def _generate_inputs(record: MoleculeRecord, scorer: Scorer) -> tuple[str, object] | None:
    """Parse a molecule then generate a form ready for inference

    Args:
        record: Molecule record to be parsed
        scorer: Tool used for inference
    Returns:
        - Key for the molecule record
        - Inference-ready format
        Or None if the transformation fails
    """

    try:
        # Compute the features
        readied = scorer.transform_inputs([record])[0]
    except (ValueError, RuntimeError):
        return None
    return record.identifier.smiles, readied


class SingleStepThinker(MoleculeThinker):
    """A thinker which submits all computations needed to evaluate a molecule whenever it is selected

    Args:
        queues: Queues used to communicate with the task server
        run_dir: Directory in which to store logs, etc.
        recipes: Recipes used to compute the target properties
        database: Connection to the store of molecular data
        solution: Settings related to tools used to solve the problem (e.g., active learning strategy)
        search_space: Search space of molecules. Provided as a list of paths to ".smi" files
        num_workers: Number of simulation tasks to run in parallel
        inference_chunk_size: Number of molecules to run inference on per task
    """

    search_space_dir: Path
    """Cache directory for search space"""
    search_space_smiles: list[list[str]]
    """SMILES strings of molecules in the search space"""
    search_space_inputs: list[list[object]]
    """Inputs (or proxies of inputs) to the machine learning models for each molecule in the search space"""

    scorer: Scorer
    """Class used to communicate data and models to distributed workers"""

    solution: SingleFidelityActiveLearning

    def __init__(self,
                 queues: ColmenaQueues,
                 run_dir: Path,
                 recipes: Sequence[PropertyRecipe],
                 solution: SingleFidelityActiveLearning,
                 search_space: list[Path | str],
                 database: MoleculeStore,
                 pool: ProcessPoolExecutor,
                 num_workers: int = 2,
                 inference_chunk_size: int = 10000,
                 run_config: dict = None):
        super().__init__(queues, ResourceCounter(num_workers), run_dir, recipes, solution, search_space, database, pool, run_config=run_config)
        
        self.search_space_dir = self.run_dir / 'search-space'
        self.scorer = solution.scorer
        
        # Load custom run configurations
        self.run_config = run_config or {}
        self.train_freq = self.run_config.get('train_freq', 2)
        self.infer_freq = self.run_config.get('infer_freq', 4)
        self.max_training_loops = self.run_config.get('max_training_loops', 5)
        self.max_loops = self.run_config.get('max_loops', -1)
        self.run_training_cycles = self.run_config.get('run_training_cycles', None)
        
        # Track states to orchestrate independent training and scoring
        self.training_loops_run = 0
        self.first_training_completed = False
        self.inference_loop_counter = 1
        self.simulated_molecules = set()

        # State tracking for the exponential/geometric policy
        self.train_policy = self.run_config.get('train_policy', 'linear')
        self.training_loops_triggered = 0
        self.next_train_target = self.train_freq if self.train_policy == 'exponential' else None

        # Initialize the true energy map for greedy inference
        key_value_file = Path("/users/vthurime/ExaMol/examples/redoxmers/molecules_key_value.txt")
        self.molecules_energies_map = {}
        if key_value_file.exists():
            with open(key_value_file, "r") as f: 
                for line in f:
                    if ":" in line:
                        key, value = line.split(":")
                        self.molecules_energies_map[key.strip()] = float(value.strip())
        self.processed_keys = set(self.molecules_energies_map.keys())

        self._cache_search_space(inference_chunk_size, search_space)

        # Startup-related information
        self.starter = self.solution.starter

        # Model tracking information
        self.models = solution.models.copy()
        if len(set(map(len, self.models))) > 1:  # pragma: no-coverage
            raise ValueError('You must provide the same number of models for each class')
        if len(self.models) != len(recipes):  # pragma: no-coverage
            raise ValueError('You must provide as many model ensembles as recipes')
        self._model_proxies: list[list[Proxy | None]] = [[None] * len(m) for m in self.models]  

        # Partition the search space into smaller chunks
        self.search_space_smiles: list[list[str]]
        self.search_space_inputs: list[list[object]]
        self.search_space_smiles, self.search_space_inputs = zip(*self._cache_search_space(inference_chunk_size, self.search_space))

        # Coordination tools
        self.start_inference: Event = Event()
        self.start_training: Event = Event()

    @property
    def num_models(self) -> int:
        """Number of models being trained by this class"""
        return sum(map(len, self.models))

    def generate_inference_results(self):
        """
        Custom inference logic based on a 10-bucket partition and logarithmic scaling.
        Ensures exploration is wide early on, and narrows to the top buckets as data and training increase.
        """
        if not self.molecules_energies_map:
            self.logger.error("No molecule key-value data found. Cannot run mock inference.")
            return [], []

        all_energies = list(self.molecules_energies_map.values())
        global_min = min(all_energies)
        global_max = max(all_energies)
        energy_range = global_max - global_min
        step = energy_range / 10.0

        # Calculate logical constraints based on hardcoded 200 sims and batches of 10
        # 1. Data ratio (0.0 to 1.0)
        data_ratio = min(1.0, self.completed / 200.0)
        
        # 2. Train ratio (0.0 to 1.0) using logarithmic pattern
        max_expected_trains = 20.0
        train_ratio = math.log(1 + self.training_loops_run) / math.log(1 + max_expected_trains)
        train_ratio = min(1.0, train_ratio)

        # E = number of bottom buckets to drop (0 to 9)
        # Combines data availability and training volume
        E = int(9.0 * data_ratio * train_ratio)
        E = max(0, min(9, E))

        cutoff_energy = global_min + (E * step)
        
        self.logger.info(f"Mock Inference | Sims: {self.completed}/200, Trains: {self.training_loops_run}/20")
        self.logger.info(f"Mock Inference | Data Ratio: {data_ratio:.2f}, Train Log Ratio: {train_ratio:.2f}")
        self.logger.info(f"Mock Inference | Dropping bottom {E} buckets. Allowed: Top {10 - E}. Cutoff: {cutoff_energy:.3f}")

        final_filtered_smiles = []
        final_predictions = []

        all_search_smiles = [smile for chunk in self.search_space_smiles for smile in chunk]

        for smile in all_search_smiles:
            try:
                inchi_key = MoleculeRecord.from_identifier(smile).key
            except Exception:
                continue

            if inchi_key not in self.processed_keys or inchi_key in self.simulated_molecules:
                continue

            true_energy = self.molecules_energies_map.get(inchi_key)
            if true_energy is None:
                continue 

            # Mixing Logic: 
            # If the molecule is inside an allowed bucket, we assign it a random score bounded by the cutoff.
            # This masks which specific bucket it belongs to, forcing the greedy selector to draw randomly 
            # from the ENTIRE allowed pool, preventing instant exhaustion of the absolute best bucket.
            if true_energy >= cutoff_energy:
                prediction = np.random.uniform(cutoff_energy, global_max)
            else:
                # Dropped buckets are assigned a terrible score so they are ignored by the greedy selector
                prediction = true_energy - 100.0 

            final_filtered_smiles.append(smile)
            final_predictions.append(prediction)

        self.logger.info(f"Generated {len(final_filtered_smiles)} predictions.")
        return final_filtered_smiles, final_predictions


    def _cache_search_space(self, inference_chunk_size: int, search_space: list[str | Path]):
        """Cache the search space into a directory within the run"""

        # Check if we must rebuild the cache
        rebuild = True
        config_path = self.search_space_dir / 'settings.json'
        my_config = {
            'inference_chunk_size': inference_chunk_size,
            'scorer': str(self.scorer),
            'paths': [str(Path(p).resolve()) for p in search_space]
        }
        if config_path.exists():
            config = json.loads(config_path.read_text())
            rebuild = config != my_config
            if rebuild:
                self.logger.info('Settings have changed. Rebuilding the cache')
                shutil.rmtree(self.search_space_dir)
        elif self.search_space_dir.exists():
            shutil.rmtree(self.search_space_dir)
        self.search_space_dir.mkdir(exist_ok=True, parents=True)

        # Get the paths to inputs and keys, either by rebuilding or reading from disk
        search_space_keys = {}
        if rebuild:
            # Process the inputs and store them to disk
            search_size = 0
            input_func = partial(_generate_inputs, scorer=self.scorer)

            # Run asynchronously
            mol_iter = self.pool.map(input_func, self.iterate_over_search_space(), chunksize=1000)
            mol_iter_no_failures = filter(lambda x: x is not None, mol_iter)
            for chunk_id, chunk in enumerate(batched(mol_iter_no_failures, inference_chunk_size)):
                keys, objects = zip(*chunk)
                search_size += len(keys)
                chunk_path = self.search_space_dir / f'chunk-{chunk_id}.pkl.gz'
                with gzip.open(chunk_path, 'wb') as fp:
                    pkl.dump(objects, fp)

                search_space_keys[chunk_path.name] = keys
            self.logger.info(f'Saved {search_size} search entries into {len(search_space_keys)} batches')

            # Save the keys and the configuration
            with open(self.search_space_dir / 'keys.json', 'w') as fp:
                json.dump(search_space_keys, fp)
            with config_path.open('w') as fp:
                json.dump(my_config, fp)
        else:
            # Load in keys
            self.logger.info(f'Loading search space from {self.search_space_dir}')
            with open(self.search_space_dir / 'keys.json') as fp:
                search_space_keys = json.load(fp)

        # Load in the molecules, storing them as proxies in the "inference" store if there is a store defined
        self.logger.info(f'Loading in molecules from {len(search_space_keys)} files')
        output = []

        proxy_store = self.inference_store
        if proxy_store is not None:
            self.logger.info(f'Will store inference objects to {proxy_store}')

        for name, keys in search_space_keys.items():
            with gzip.open(self.search_space_dir / name, 'rb') as fp:  # Load from disk
                objects = pkl.load(fp)

            if proxy_store is not None:  # If the store exists, make a proxy
                objects = proxy_store.proxy(objects)
            output.append((keys, objects))
        return output

    @property
    def inference_store(self) -> Store | None:
        """Proxystore used for inference tasks"""
        if (store_name := self.queues.proxystore_name.get('inference')) is not None:
            return get_store(store_name)

    def _get_training_set(self, recipe: PropertyRecipe) -> list[MoleculeRecord]:
        """Gather molecules for which the target property is available

        Args:
            recipe: Recipe to evaluate
        Returns:
            List of molecules for which that property is defined
        """
        return [x for x in self.database.iterate_over_records() if recipe.lookup(x) is not None]

    # TODO (wardlt): Move to a function of the database class?
    def count_training_size(self, recipe: PropertyRecipe) -> int:
        """Count the number of entries available for training each recipe

        Args:
            recipe: Recipe being assessed
        Return:
            Number of records for which this property is defined
        """

        return len([None for r in self.database.iterate_over_records() if recipe.name in r.properties and recipe.level in r.properties[recipe.name]])

    @agent(startup=True)
    def startup(self):
        """Pre-populate the database, if needed."""

        # Determine how many training points are available
        train_size = min(self.count_training_size(r) for r in self.recipes)

        # If enough, start by training
        if train_size > self.solution.minimum_training_size:
            self.logger.info(f'Training set is larger than the threshold size ({train_size}>{self.solution.minimum_training_size}). Starting model training')
            self.start_training.set()
            return

        self.logger.info(f'Training set is smaller than the threshold size ({train_size}<{self.solution.minimum_training_size}). Falling back to greedy initialization.')
        
        # Trigger the greedy inference to generate the first batch instead of random start
        self.start_inference.set()

    def get_additional_training_information(self, train_set: list[MoleculeRecord], recipe: PropertyRecipe) -> dict[str, object]:
        """Determine any additional information to be provided during training

        An example could be to gather low-fidelity data to use to augment the training process

        Args:
            train_set: Training set for the model
            recipe: Recipe being trained
        Returns:
            Additional options
        """
        return {}

    @event_responder(event_name='start_training')
    def retrain(self):
        self.start_training.clear()

        # Check that we have enough data for all recipes
        for recipe in self.recipes:
            train_size = min(self.count_training_size(r) for r in self.recipes)
            if train_size < self.solution.minimum_training_size:
                self.logger.info(f'Too few to entries to train {recipe.name}. Waiting for {self.solution.minimum_training_size}. Have {train_size}')
                # LOCKSTEP FALLBACK: Trigger inference to unpause simulations and prevent deadlock
                self.start_inference.set()
                return

        for recipe_id, recipe in enumerate(self.recipes):
            # Get the training set
            train_set = self._get_training_set(recipe)
            self.logger.info(f'Gathered a total of {len(train_set)} entries for retraining recipe {recipe_id}')

            # ADDED: Log the exact input size to the sequence log
            self._log_sequence(f"Training | input size: {len(train_set)}")

            # Process to form the inputs and outputs
            train_inputs = self.scorer.transform_inputs(train_set)
            train_outputs = self.scorer.transform_outputs(train_set, recipe)
            train_kwargs = self.get_additional_training_information(train_set, recipe)
            self.logger.info('Pre-processed the training entries')

            # Submit all models
            for model_id, model in enumerate(self.models[recipe_id]):
                model_msg = self.scorer.prepare_message(model, training=True)
                self.queues.send_inputs(
                    model_msg, train_inputs, train_outputs,
                    input_kwargs=train_kwargs,
                    method='retrain',
                    topic='train',
                    task_info={'recipe_id': recipe_id, 'model_id': model_id}
                )
            self.logger.info(f'Submitted all models for recipe={recipe_id}')

        # Retrieve the results
        for i in range(self.num_models):
            result = self.queues.get_result(topic='train')
            self._write_result(result, 'train')
            assert result.success, f'Training failed: {result.failure_info}'

            # Update the appropriate model
            model_id = result.task_info['model_id']
            recipe_id = result.task_info['recipe_id']
            model_msg = result.value
            if isinstance(model_msg, Proxy):
                # Forces resolution. Needed to avoid `submit_inference` from making a proxy of `model_msg`, which can happen if it is not resolved
                #  by `scorer.update` and is a problem because the proxy for `model_msg` can be evicted while other processes need it
                model_msg = extract(model_msg)
            self.models[recipe_id][model_id] = self.scorer.update(self.models[recipe_id][model_id], model_msg)
            self.logger.info(f'Updated model {i + 1}/{self.num_models}. Recipe id={recipe_id}. Model id={model_id}')

            self.first_training_completed = True

        self.training_loops_run += 1
        self.logger.info('Finished training all models. Triggering inference in lockstep.')
        
        # LOCKSTEP: Trigger inference sequentially after training is fully completed
        self.start_inference.set()

    def submit_inference(self) -> tuple[list[list[str]], np.ndarray, list[np.ndarray]]:
        """Submit all molecules to be evaluated, return placeholders for their outputs

        Inference tasks are submitted with a few bits of metadata
            - recipe_id: Index of the recipe being evaluated
            - model_id: Index of the model being evaluated
            - chunk_id: Index of the chunk of molecules
            - chunk_size: Number of molecules in chunks being evaluated

        Returns:
            - Smiles strings of the molecules being evaluated
            - Boolean array marking if inference task is done ``n_chunks x recipes x ensemble_size``
            - List of arrays in which to store inference results a total of ``n_chunks`` arrays of size ``recipes x batch_size x models``
        """

        # Get the proxystore for inference, if defined
        store = self.inference_store

        # Directly loop over the updated models for independent execution
        for recipe_id in range(len(self.recipes)):
            for model_id in range(len(self.models[recipe_id])):
                model = self.models[recipe_id][model_id]

                model_msg = self.scorer.prepare_message(model, training=False)
                if store is not None:
                    model_msg = store.proxy(model_msg)
                    self._model_proxies[recipe_id][model_id] = model_msg
                self.logger.info(f'Preparing to submit tasks for recipe {recipe_id}, model {model_id}.')

                for chunk_id, (chunk_inputs, chunk_keys) in enumerate(zip(self.search_space_inputs, self.search_space_smiles)):
                    self.queues.send_inputs(
                        model_msg, chunk_inputs,
                        method='score',
                        topic='inference',
                        task_info={'recipe_id': recipe_id, 'model_id': model_id, 'chunk_id': chunk_id, 'chunk_size': len(chunk_keys)}
                    )
                self.logger.info(f'Submitted all tasks for recipe={recipe_id} model={model_id}')

        # Prepare to store the inference results
        n_chunks = len(self.search_space_inputs)
        ensemble_size = len(self.models[0])
        all_done: np.ndarray = np.zeros((n_chunks, len(self.recipes), ensemble_size), dtype=bool)
        inference_results: list[np.ndarray] = [
            np.zeros((len(self.recipes), len(chunk), ensemble_size)) for chunk in self.search_space_smiles
        ]  # (chunk, recipe, molecule, model)
        return list(self.search_space_smiles), all_done, inference_results

    def _filter_inference_results(self, chunk_id: int, chunk_smiles: list[str], inference_results: np.ndarray) -> tuple[list[str], np.ndarray]:
        """Remove entries from the input array before adding to the selector

        Args:
            chunk_id: Index of the chunk being processed
            chunk_smiles: SMILES strings for molecules in this chunk
            inference_results: Results for the inference
        Returns:
             - SMILES strings of chunk after filtering
             - Inference results after filtering
        """
        return chunk_smiles, inference_results

    @event_responder(event_name='start_inference')
    def run_inference(self):
        """Perform greedy inference simulation, store results, then update the task list"""
        self.start_inference.clear()

        # If max training is 0, select to_select molecules to run randomly each time
        if self.max_loops == 0:
            with open((self.run_dir / 'run_sequence.log'), 'a') as fp:
                fp.write(f"Max Loops 0 Reached. Selecting {self.solution.selector.to_select} random molecules.\n")
            
            search_space_size = sum(map(len, self.search_space_smiles))
            subset = self.starter.select(list(interleave_longest(*self.search_space_smiles)), min(self.num_to_run, search_space_size))
            with self.task_queue_lock:
                for key in subset:
                    self.task_queue.append((key, np.nan))
                    # It is okay to add to simulated_molecules here for the random initialization
                    self.simulated_molecules.add(MoleculeRecord.from_identifier(key).key)
                self.task_queue_lock.notify_all()
            return

        # If max training is reached, note it but still pick from the latest bucketed predictions
        if self.max_loops != -1 and (self.inference_loop_counter - 1) >= self.max_loops:
            with open((self.run_dir / 'run_sequence.log'), 'a') as fp:
                fp.write(f"Max Loops {self.max_loops} Reached. Selecting {self.solution.selector.to_select} previously predicted molecules.\n")

        self.logger.info(f'Starting mock-ML inference blending.')

        # Reset the selector
        selector = self.solution.selector
        selector.update(self.database, self.recipes)
        selector.start_gathering()

        # --- MOCK GREEDY INFERENCE LOGIC ---
        final_filtered_smiles, final_predictions = self.generate_inference_results()

        # Add to selector
        if final_filtered_smiles:
            # Format results for the selector. Shape must be (recipes, molecules, models)
            # We assume 1 recipe and 1 model for this mock score to map cleanly
            final_results_array = np.array(final_predictions).reshape(1, -1, 1)
            selector.add_possibilities(final_filtered_smiles, final_results_array)
            self.logger.info(f"Added {len(final_filtered_smiles)} possibilities to selector.")
        # --- END OF MOCK LOGIC ---

        self.logger.info('Done storing all results')
        
        # LOCKSTEP RESUME: Repopulate the queue and wake up the base thinker
        with self.task_queue_lock:
            # Preserve any in-progress multi-step tasks that got added during inference
            self.task_queue = [x for x in self.task_queue if x[1] == np.inf]
            
            # Create a set of keys currently running to prevent duplicate submissions
            running_keys = {x[0] for x in self.task_queue}
            
            for key_f, score in selector.dispense():
                # Safely extract key for tracking
                mol_record = MoleculeRecord.from_identifier(str(key_f))
                key = str(key_f)
                
                # Only append if it is not already actively running
                if key not in running_keys:
                    self.task_queue.append((key, score))

            self.simulations_paused = False  # UNPAUSE simulations
            # Notify anyone waiting on more tasks
            self.task_queue_lock.notify_all()
            
        self.logger.info('Updated task queue and resumed simulations. All done.')

    def _simulations_complete(self, record: MoleculeRecord):
        # Mark molecule as simulated so greedy inference skips it next time
        self.simulated_molecules.add(record.key)

        # 1. Evaluate Training condition
        trigger_training = False
        trigger_inference_only = False
        
        if self.run_training_cycles is not None:
            if self.completed > 0 and self.completed % self.train_freq == 0:
                cycle_index = (self.completed // self.train_freq) - 1
                if str(cycle_index) in self.run_training_cycles:
                    trigger_training = True
                else:
                    trigger_inference_only = True
        elif self.train_policy == 'exponential':
            if self.completed >= self.next_train_target:
                trigger_training = True
                self.training_loops_triggered += 1
                self.next_train_target += 2 ** self.training_loops_triggered
        else:
            if self.completed > 0 and self.completed % self.train_freq == 0:
                trigger_training = True

        if trigger_training:
            if self.training_loops_run < self.max_training_loops:
                self.logger.info(f'Triggering training. Iterations complete: {self.completed}')
                
                # LOCKSTEP PAUSE: Pause simulations, preserving only in-progress tasks
                with self.task_queue_lock:
                    self.simulations_paused = True
                    self.task_queue = [x for x in self.task_queue if x[1] == np.inf]
                    
                self.start_training.set()
                
        elif trigger_inference_only:
            self.logger.info(f'Skipping training but triggering inference. Iterations complete: {self.completed}')
            
            # Pause simulations and immediately trigger inference directly without updating models
            with self.task_queue_lock:
                self.simulations_paused = True
                self.task_queue = [x for x in self.task_queue if x[1] == np.inf]
                
            self.start_inference.set()