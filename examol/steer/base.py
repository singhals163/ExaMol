"""Base class that defines core routines used across many steering policies"""
import gzip
import json
import logging
from pathlib import Path
from dataclasses import asdict
from threading import Condition
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Iterator, Sequence, Iterable

import numpy as np
from colmena.models import Result
from colmena.queue import ColmenaQueues
from colmena.thinker import BaseThinker, ResourceCounter, result_processor, task_submitter
from pydantic import ValidationError

from examol.simulate.base import SimResult
from examol.solution import SolutionSpecification
from examol.store.db.base import MoleculeStore
from examol.store.models import MoleculeRecord
from examol.store.recipes import PropertyRecipe, SimulationRequest


class MoleculeThinker(BaseThinker):
    """Base for a thinker which performs molecular design"""

    def __init__(self,
                 queues: ColmenaQueues,
                 rec: ResourceCounter,
                 run_dir: Path,
                 recipes: Sequence[PropertyRecipe],
                 solution: SolutionSpecification,
                 search_space: list[Path | str],
                 database: MoleculeStore,
                 pool: ProcessPoolExecutor,
                 run_config: dict = None):
        super().__init__(queues, resource_counter=rec)
        self.database = database
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.search_space = search_space
        self.run_config = run_config or {}

        # Log mapping
        handler = logging.FileHandler(self.run_dir / 'run.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        for logger_name in [self.logger.name, 'colmena', 'proxystore']:
            logger = logging.getLogger(logger_name)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        self.solution = solution
        self.num_to_run: int = solution.num_to_run
        self.completed: int = 0
        self.molecules_in_progress: dict[str, int] = defaultdict(int)

        self.recipes = tuple(recipes)
        self.task_queue_lock = Condition()
        self.task_queue = []  # List of tuples: (SMILES string, score)
        self.simulations_paused = False
        self.task_iterator = self.task_iterator_generator() 
        self.recipe_types = {r.name: r for r in recipes}

        self.pool: ProcessPoolExecutor = pool

    def _log_sequence(self, message: str):
        with open(self.run_dir / 'run_sequence.log', 'a') as f:
            f.write(f"{message}\n")

    def iterate_over_search_space(self, only_smiles: bool = False) -> Iterator[MoleculeRecord | str]:
        for i, path in enumerate(self.search_space):
            path = Path(path).resolve()
            self.logger.info(f'Reading molecules from file {i + 1}/{len(self.search_space)}: {path}')

            filename_lower = path.name.lower()
            if not any(filename_lower.endswith(ext) or filename_lower.endswith(f'{ext}.gz') for ext in ['.smi', '.json']):
                raise ValueError(f'File type is unrecognized for {path}')

            is_json = '.json' in filename_lower
            open_func = gzip.open if filename_lower.endswith('.gz') else open
            mode = 'rt' if filename_lower.endswith('.gz') else 'r'

            with open_func(path, mode) as fmols:
                for line in fmols:
                    line = line.strip()
                    if not line:
                        continue
                        
                    if only_smiles and is_json:
                        yield json.loads(line)['identifier']['smiles']
                    elif only_smiles and not is_json:
                        yield line
                    elif is_json:
                        yield MoleculeRecord.parse_raw(line)
                    else:
                        try:
                            yield MoleculeRecord.from_identifier(line)
                        except ValidationError:
                            self.logger.warning(f'Parsing failed for molecule: {line}')

    def _write_result(self, result: Result, result_type: str):
        with (self.run_dir / f'{result_type}-results.json').open('a') as fp:
            print(result.json(exclude={'value', 'inputs'}), file=fp)

    def _get_next_tasks(self) -> tuple[MoleculeRecord, float, Iterable[PropertyRecipe]]:
        smiles, score = self.task_queue.pop(0)
        return self.database.get_or_make_record(smiles), score, self.recipes

    def task_iterator_generator(self) -> Iterator[tuple[MoleculeRecord, Iterable[PropertyRecipe], SimulationRequest]]:
        while True:
            with self.task_queue_lock:
                while len(self.task_queue) == 0 or (self.simulations_paused and self.task_queue[0][1] != np.inf):
                    # Pause natively here. Starvation is explicitly handled by `_simulations_complete`.
                    self.logger.info('No tasks available or paused for lockstep. Waiting.')
                    while not self.task_queue_lock.wait(timeout=2):
                        if self.done.is_set():
                            yield None, None, None
                            
                record, score, recipes = self._get_next_tasks()

            recipe_names = [f'{r.name}/{r.level}' for r in recipes]
            self.logger.info(f'Selected {record.key} to run next. Recipes: {", ".join(recipe_names)}. '
                             f'Score={score:.2f}, Queue length={len(self.task_queue)}')

            try:
                suggestions = set()
                for recipe in recipes:
                    suggestions.update(recipe.suggest_computations(record))
            except ValueError as exc:
                self.logger.warning(f'Generating computations for {record.key} failed. Skipping. Reason: {exc}')
                continue
                
            self.logger.info(f'Found {len(suggestions)} more computations to do for {record.key}')
            self.molecules_in_progress[record.key] += len(suggestions)

            for suggestion in suggestions:
                yield record, recipes, suggestion

    def _simulations_complete(self, record: MoleculeRecord):
        pass

    @result_processor(topic='simulation')
    def store_simulation(self, result: Result):
        self.rec.release()

        mol_key = result.task_info["key"]
        record = self.database[mol_key]
        self.logger.info(f'Received result for {mol_key}. Runtime={(result.time_running or np.nan):.1f}s, success={result.success}')

        self.molecules_in_progress[mol_key] -= 1

        if result.success:
            if result.method == 'optimize_structure':
                sim_result, steps, metadata = result.value
                results = [sim_result] + steps
                record.add_energies(sim_result, steps)
            elif result.method == 'compute_energy':
                sim_result, metadata = result.value
                results = [sim_result]
                record.add_energies(sim_result)
            else:
                raise NotImplementedError()

            recipes = [self.recipe_types[r['name']].from_name(**r) for r in result.task_info['recipes']]
            self.logger.info(f'Checking if completed recipes: {", ".join([r.name + "//" + r.level for r in recipes])}')

            not_done = sum(recipe.lookup(record, recompute=True) is None for recipe in recipes)
            if not_done == 0:
                self.completed += 1
                self.logger.info(f'Finished all recipes for {mol_key}. Completed {self.completed}/{self.num_to_run} molecules')
                self.molecules_in_progress.pop(mol_key)
                
                if self.completed == self.num_to_run:
                    self.logger.info('Done!')
                    self.done.set()

                final_results = [recipe.lookup(record) for recipe in self.recipes]
                result.task_info['status'] = 'finished'
                result.task_info['result'] = final_results
                
                for final_energy in final_results:
                    self._log_sequence(f"Simulation result | key: {record.key} value: {final_energy}")

                self._simulations_complete(record)
            else:
                self.logger.info(f'Finished {len(self.recipes) - not_done}/{len(self.recipes)} recipes for {mol_key}')
                result.task_info['status'] = 'in progress'
                if self.molecules_in_progress[mol_key] == 0:
                    self.logger.info('Submitting new computations. Re-adding to front of queue.')
                    with self.task_queue_lock:
                        self.task_queue.insert(0, (record.identifier.smiles, np.inf))
                        self.task_queue_lock.notify_all()

            self.database.update_record(record)

            with open(self.run_dir / 'simulation-records.json', 'a') as fp:
                for res_record in results:
                    print(res_record.json(), file=fp)
        else:
            if self.molecules_in_progress[mol_key] == 0:
                self.molecules_in_progress.pop(mol_key)
                self._simulations_complete(record)

        self._write_result(result, 'simulation')

    @task_submitter()
    def submit_simulation(self):
        record, recipes, suggestion = next(self.task_iterator)
        if record is None:
            return

        task_info = {
            'key': record.key,
            'recipes': [{'name': r.name, 'level': r.level} for r in recipes],
            'computation': asdict(suggestion)
        }
        
        if suggestion.optimize:
            self.logger.info(f'Optimizing structure for {record.key} with charge {suggestion.charge}')
            method = 'optimize_structure'
        else:
            solvent_str = '' if suggestion.solvent is None else f'in {suggestion.solvent}'
            self.logger.info(f'Getting single-point energy for {record.key} with charge {suggestion.charge} {solvent_str}')
            method = 'compute_energy'

        self.queues.send_inputs(
            record.key, suggestion.xyz, suggestion.config_name, suggestion.charge, suggestion.solvent,
            method=method,
            topic='simulation',
            task_info=task_info
        )