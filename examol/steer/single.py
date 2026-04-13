"""Single-objective and single-fidelity implementation of active learning."""
import gzip
import json
import pickle as pkl
import shutil
import math
import threading
from functools import partial
from pathlib import Path
from threading import Event
from typing import Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from colmena.proxy import get_store
from colmena.queue import ColmenaQueues
from colmena.thinker import event_responder, ResourceCounter, agent
from more_itertools import interleave_longest, batched
from proxystore.proxy import extract, Proxy
from proxystore.store import Store

from .base import MoleculeThinker
from examol.solution import SingleFidelityActiveLearning
from ..score.base import Scorer
from ..store.db.base import MoleculeStore
from ..store.models import MoleculeRecord
from ..store.recipes import PropertyRecipe


def _generate_inputs(record: MoleculeRecord, scorer: Scorer) -> tuple[str, object] | None:
    try:
        readied = scorer.transform_inputs([record])[0]
    except (ValueError, RuntimeError):
        return None
    return record.identifier.smiles, readied


class SingleStepThinker(MoleculeThinker):
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
        self.solution = solution
        self.scorer = solution.scorer
        
        self.train_freq = self.run_config.get('train_freq', 2)
        self.max_training_loops = self.run_config.get('max_training_loops', 5)
        self.max_loops = self.run_config.get('max_loops', -1)
        self.run_training_cycles = self.run_config.get('run_training_cycles', None)
        self.train_policy = self.run_config.get('train_policy', 'linear')
        
        # State and coordination variables
        self.training_loops_run = 0
        self.first_training_completed = False
        self.inference_loop_counter = 1
        self.simulated_molecules = set()
        self.state_lock = threading.Lock()  # Protects state transitions

        self.training_loops_triggered = 0
        self.next_train_target = self.train_freq if self.train_policy == 'exponential' else None

        # Load mock energy map
        key_value_file = Path("/users/vthurime/ExaMol/examples/redoxmers/molecules_key_value.txt")
        self.molecules_energies_map = {}
        if key_value_file.exists():
            with open(key_value_file, "r") as f:
                self.molecules_energies_map = {
                    line.split(":")[0].strip(): float(line.split(":")[1].strip())
                    for line in f if ":" in line
                }
        self.processed_keys = set(self.molecules_energies_map.keys())

        self.starter = self.solution.starter
        self.models = [list(m) for m in solution.models]

        if len(set(map(len, self.models))) > 1:
            raise ValueError('You must provide the same number of models for each class')
        if len(self.models) != len(recipes):
            raise ValueError('You must provide as many model ensembles as recipes')
            
        self._model_proxies: list[list[Proxy | None]] = [[None] * len(m) for m in self.models]  
        self.search_space_smiles, self.search_space_inputs = zip(*self._cache_search_space(inference_chunk_size, self.search_space))

        # Events
        self.start_inference: Event = Event()
        self.start_training: Event = Event()

    @property
    def num_models(self) -> int:
        return sum(map(len, self.models))

    def generate_inference_results(self):
        """
        Advanced Gaussian Noise inference.
        Uses Rational Function learning curves and Polynomial penalties to accurately 
        simulate the slow convergence, overfitting, and catastrophic forgetting of ML models.
        """
        self.inference_loop_counter += 1

        if not self.molecules_energies_map:
            self.logger.error("No molecule key-value data found. Cannot run inference.")
            return [], []

        all_energies = list(self.molecules_energies_map.values())
        global_min, global_max = min(all_energies), max(all_energies)
        energy_range = global_max - global_min

        # 1. State Tracking: Snapshot data ONLY when training occurs
        if not hasattr(self, '_mock_last_train_count'):
            self._mock_last_train_count = 0
            self._mock_last_data_count = 1  # Base start

        # If a training loop completed since the last inference, update the snapshot
        if self.training_loops_run > self._mock_last_train_count:
            self._mock_last_train_count = self.training_loops_run
            self._mock_last_data_count = max(1, self.completed)

        # The model ONLY knows about the data it was explicitly trained on!
        trained_data_count = self._mock_last_data_count
        train_count = self.training_loops_run
        
        # Config Parameters
        coeff = self.run_config.get('mock_ideal_coeff', 1.5)
        plateau_sims = self.run_config.get('mock_plateau_sims', 70)
        overfit_thresh = self.run_config.get('mock_overfit_ratio', 1.25)
        undertrain_thresh = self.run_config.get('mock_undertrain_ratio', 0.5)

        # 2. The Ideal Training Curve with Plateau Enforcement
        effective_data_for_training = min(trained_data_count, plateau_sims)
        optimal_trains = coeff * math.sqrt(effective_data_for_training)
        
        ratio = train_count / max(0.1, optimal_trains)

        # Base Noise: Needs to be large enough to completely scramble the energy range
        base_sigma = energy_range * 1.5

        if train_count == 0:
            # PURE RANDOMNESS: Model has never been trained. 
            # Apply overwhelming noise so the greedy selector picks blindly.
            sigma = base_sigma * 3.0
            self.logger.info("Mock Inference | Untrained model. Applying maximum random noise.")
        else:
            # 3. Determine Training Modifier using Slower Rational Functions
            # Requires ~30 data points to halve the base data error
            data_learning_factor = 30.0 / (trained_data_count + 30.0)
            
            # Requires ~5 training loops to halve the base training error
            effective_trains = min(train_count, optimal_trains)
            train_learning_factor = 5.0 / (effective_trains + 5.0)

            base_train_modifier = data_learning_factor * train_learning_factor

            # 4. Apply Polynomial Penalties
            if ratio > overfit_thresh:
                # OVERFITTING: Cubed penalty for training too much past the ideal curve
                penalty = (ratio / overfit_thresh) ** 3
                train_modifier = base_train_modifier * penalty
                self.logger.warning(f"Mock Inference | Overfitting penalty active! Ratio: {ratio:.2f}")
                
            elif ratio < undertrain_thresh:
                # UNDERTRAINING: Squared penalty for not training enough.
                # Catches the "One Shot" strategy and destroys its accuracy as data grows.
                safe_ratio = max(0.05, ratio) 
                penalty = (undertrain_thresh / safe_ratio) ** 2
                train_modifier = base_train_modifier * penalty
                self.logger.warning(f"Mock Inference | Undertraining penalty active! Ratio: {ratio:.2f}")
                
            else:
                # HEALTHY: Coasting perfectly along the curve
                train_modifier = base_train_modifier

            # 5. Final Noise Calculation
            sigma = base_sigma * train_modifier

        self.logger.info(f"Mock Inference | Trained on Data: {trained_data_count}, Trains: {train_count} (Ideal: {optimal_trains:.1f})")
        self.logger.info(f"Mock Inference | Calculated Noise (Sigma): {sigma:.3f}")

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

            # Prediction = True Value + ML Uncertainty (Noise)
            noise = np.random.normal(0, sigma)
            prediction = true_energy + noise

            final_filtered_smiles.append(smile)
            final_predictions.append(prediction)

        self.logger.info(f"Generated {len(final_filtered_smiles)} predictions.")
        return final_filtered_smiles, final_predictions
    
    def _cache_search_space(self, inference_chunk_size: int, search_space: list[str | Path]):
        rebuild = True
        config_path = self.search_space_dir / 'settings.json'
        my_config = {
            'inference_chunk_size': inference_chunk_size,
            'scorer': str(self.scorer),
            'paths': [str(Path(p).resolve()) for p in search_space]
        }
        
        if config_path.exists():
            rebuild = json.loads(config_path.read_text()) != my_config
            if rebuild:
                self.logger.info('Settings changed. Rebuilding cache.')
                shutil.rmtree(self.search_space_dir)
        elif self.search_space_dir.exists():
            shutil.rmtree(self.search_space_dir)
            
        self.search_space_dir.mkdir(exist_ok=True, parents=True)

        search_space_keys = {}
        if rebuild:
            search_size = 0
            input_func = partial(_generate_inputs, scorer=self.scorer)
            mol_iter = filter(None, self.pool.map(input_func, self.iterate_over_search_space(), chunksize=1000))

            for chunk_id, chunk in enumerate(batched(mol_iter, inference_chunk_size)):
                keys, objects = zip(*chunk)
                search_size += len(keys)
                chunk_path = self.search_space_dir / f'chunk-{chunk_id}.pkl.gz'
                with gzip.open(chunk_path, 'wb') as fp:
                    pkl.dump(objects, fp)
                search_space_keys[chunk_path.name] = keys
                
            with open(self.search_space_dir / 'keys.json', 'w') as fp:
                json.dump(search_space_keys, fp)
            with config_path.open('w') as fp:
                json.dump(my_config, fp)
        else:
            with open(self.search_space_dir / 'keys.json') as fp:
                search_space_keys = json.load(fp)

        output = []
        proxy_store = self.inference_store
        for name, keys in search_space_keys.items():
            with gzip.open(self.search_space_dir / name, 'rb') as fp:
                objects = pkl.load(fp)
            if proxy_store:
                objects = proxy_store.proxy(objects)
            output.append((keys, objects))
            
        return output

    @property
    def inference_store(self) -> Store | None:
        if (store_name := self.queues.proxystore_name.get('inference')) is not None:
            return get_store(store_name)

    def _get_training_set(self, recipe: PropertyRecipe) -> list[MoleculeRecord]:
        return [x for x in self.database.iterate_over_records() if recipe.lookup(x) is not None]

    def count_training_size(self, recipe: PropertyRecipe) -> int:
        return sum(1 for r in self.database.iterate_over_records() if recipe.name in r.properties and recipe.level in r.properties[recipe.name])

    @agent(startup=True)
    def startup(self):
        train_size = min(self.count_training_size(r) for r in self.recipes)
        if train_size > self.solution.minimum_training_size:
            self.logger.info('Training set exceeds threshold. Starting initial training.')
            self.start_training.set()
        else:
            self.logger.info('Training set below threshold. Falling back to greedy init.')
            self.start_inference.set()

    def get_additional_training_information(self, train_set: list[MoleculeRecord], recipe: PropertyRecipe) -> dict[str, object]:
        return {}

    @event_responder(event_name='start_training')
    def retrain(self):
        self.start_training.clear()

        for recipe in self.recipes:
            train_size = min(self.count_training_size(r) for r in self.recipes)
            if train_size < self.solution.minimum_training_size:
                self.logger.info(f'Too few entries to train {recipe.name}. Have {train_size}. Falling back to inference.')
                self.start_inference.set()
                return

        for recipe_id, recipe in enumerate(self.recipes):
            train_set = self._get_training_set(recipe)
            self.logger.info(f'Gathered {len(train_set)} entries for retraining recipe {recipe_id}')
            self._log_sequence(f"Training | input size: {len(train_set)}")

            train_inputs = self.scorer.transform_inputs(train_set)
            train_outputs = self.scorer.transform_outputs(train_set, recipe)
            train_kwargs = self.get_additional_training_information(train_set, recipe)

            for model_id, model in enumerate(self.models[recipe_id]):
                model_msg = self.scorer.prepare_message(model, training=True)
                self.queues.send_inputs(
                    model_msg, train_inputs, train_outputs,
                    input_kwargs=train_kwargs,
                    method='retrain',
                    topic='train',
                    task_info={'recipe_id': recipe_id, 'model_id': model_id}
                )

        for i in range(self.num_models):
            result = self.queues.get_result(topic='train')
            self._write_result(result, 'train')
            assert result.success, f'Training failed: {result.failure_info}'

            model_id, recipe_id = result.task_info['model_id'], result.task_info['recipe_id']
            model_msg = extract(result.value) if isinstance(result.value, Proxy) else result.value
            
            self.models[recipe_id][model_id] = self.scorer.update(self.models[recipe_id][model_id], model_msg)
            self.logger.info(f'Updated model {i + 1}/{self.num_models}. Recipe={recipe_id}, Model={model_id}')

        self.first_training_completed = True
        self.training_loops_run += 1
        
        # Enforce pipeline sequence: Training directly triggers Inference
        self.logger.info('Finished training. Triggering inference sequence.')
        self.start_inference.set()

    def submit_inference(self) -> tuple[list[list[str]], np.ndarray, list[np.ndarray]]:
        store = self.inference_store
        for recipe_id, models in enumerate(self.models):
            for model_id, model in enumerate(models):
                model_msg = self.scorer.prepare_message(model, training=False)
                if store:
                    model_msg = store.proxy(model_msg)
                    self._model_proxies[recipe_id][model_id] = model_msg

                for chunk_id, (chunk_inputs, chunk_keys) in enumerate(zip(self.search_space_inputs, self.search_space_smiles)):
                    self.queues.send_inputs(
                        model_msg, chunk_inputs, method='score', topic='inference',
                        task_info={'recipe_id': recipe_id, 'model_id': model_id, 'chunk_id': chunk_id, 'chunk_size': len(chunk_keys)}
                    )

        n_chunks = len(self.search_space_inputs)
        ensemble_size = len(self.models[0])
        all_done = np.zeros((n_chunks, len(self.recipes), ensemble_size), dtype=bool)
        inference_results = [np.zeros((len(self.recipes), len(c), ensemble_size)) for c in self.search_space_smiles]
        return list(self.search_space_smiles), all_done, inference_results

    def _filter_inference_results(self, chunk_id: int, chunk_smiles: list[str], inference_results: np.ndarray) -> tuple[list[str], np.ndarray]:
        return chunk_smiles, inference_results

    @event_responder(event_name='start_inference')
    def run_inference(self):
        self.start_inference.clear()

        # Enforce Model Readiness Check
        if not self.first_training_completed:
            self.logger.info('Model not ready. Running fallback/greedy inference.')
        else:
            self.logger.info('Model verified. Running active inference.')

        if self.max_loops == 0:
            with open((self.run_dir / 'run_sequence.log'), 'a') as fp:
                fp.write(f"Max Loops 0 Reached. Selecting {self.solution.selector.to_select} random molecules.\n")
            search_space_size = sum(map(len, self.search_space_smiles))
            subset = self.starter.select(list(interleave_longest(*self.search_space_smiles)), min(self.num_to_run, search_space_size))
            
            with self.task_queue_lock:
                for key in subset:
                    self.task_queue.append((key, np.nan))
                    self.simulated_molecules.add(MoleculeRecord.from_identifier(key).key)
                self.simulations_paused = False
                self.task_queue_lock.notify_all()
            return

        selector = self.solution.selector
        selector.update(self.database, self.recipes)
        selector.start_gathering()

        final_filtered_smiles, final_predictions = self.generate_inference_results()

        if final_filtered_smiles:
            final_results_array = np.array(final_predictions).reshape(1, -1, 1)
            selector.add_possibilities(final_filtered_smiles, final_results_array)
            self.logger.info(f"Added {len(final_filtered_smiles)} possibilities to selector.")
        
        dispensed = list(selector.dispense())
        
        with self.task_queue_lock:
            self.task_queue = [x for x in self.task_queue if x[1] == np.inf]
            running_keys = {x[0] for x in self.task_queue}
            
            for key_f, score in dispensed:
                key = str(key_f)
                if key not in running_keys:
                    self.task_queue.append((key, score))

            if not self.task_queue and self.completed < self.num_to_run:
                self.logger.warning("Search space exhausted but target not met. Exiting gracefully.")
                self.done.set()

            self.simulations_paused = False
            self.task_queue_lock.notify_all()
            
        self.logger.info('Updated task queue and resumed simulations.')

    def _pause_and_trigger(self, train: bool):
        """Safely pause queue to prepare for state transition."""
        with self.task_queue_lock:
            self.simulations_paused = True
            self.task_queue = [x for x in self.task_queue if x[1] == np.inf]

        if train:
            self.start_training.set()
        else:
            self.start_inference.set()

    def _simulations_complete(self, record: MoleculeRecord):
        self.simulated_molecules.add(record.key)

        # Thread-safe deterministic evaluation block
        with self.state_lock:
            if self.simulations_paused:
                return

            trigger_training = False
            trigger_inference = False
            
            # 1. Evaluate standard policy constraints
            if self.run_training_cycles is not None:
                if self.completed > 0 and self.completed % self.train_freq == 0:
                    cycle_index = (self.completed // self.train_freq) - 1
                    if str(cycle_index) in self.run_training_cycles:
                        trigger_training = True
                    else:
                        trigger_inference = True
            elif self.train_policy == 'exponential':
                if self.completed >= self.next_train_target:
                    trigger_training = True
                    self.training_loops_triggered += 1
                    self.next_train_target += 2 ** self.training_loops_triggered
            elif self.completed > 0 and self.completed % self.train_freq == 0:
                trigger_training = True

            # 2. Starvation check: Force inference if we run out of valid pending tasks
            with self.task_queue_lock:
                active_tasks = sum(1 for x in self.task_queue if x[1] != np.inf)

            if active_tasks == 0 and self.completed < self.num_to_run:
                self.logger.info("Task queue depleted. Forcing inference to prevent starvation.")
                trigger_inference = True

            # 3. Apply state transitions (Train inherently chains to Infer)
            if trigger_training and self.training_loops_run < self.max_training_loops:
                self.logger.info(f'Transition: Simulate -> Train. Iterations: {self.completed}')
                self._pause_and_trigger(train=True)
            elif trigger_training or trigger_inference:
                self.logger.info(f'Transition: Simulate -> Infer. Iterations: {self.completed}')
                self._pause_and_trigger(train=False)