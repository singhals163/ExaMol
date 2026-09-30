"""Simulator which replays precomputed results instead of running quantum chemistry codes

Used to run many active learning campaigns against a fixed set of ground-truth values
(see ``experiments/precompute``) so that differences between runs come only from
the learning strategy, not from simulation noise.
"""
import json
from functools import lru_cache
from pathlib import Path
from time import perf_counter, sleep

import numpy as np

from examol.simulate.base import BaseSimulator, SimResult
from examol.store.models import MoleculeRecord, Conformer
from examol.utils.conversions import read_from_string


@lru_cache(maxsize=4)
def _load_results(path: str) -> dict[str, dict]:
    """Load the precomputed results, keyed by InChI key"""
    output = {}
    with open(path) as fp:
        for line in fp:
            if not line.strip():
                continue
            entry = json.loads(line)
            if 'key' not in entry:
                continue
            if entry.get('record') is not None:
                entry['record'] = MoleculeRecord.parse_raw(entry['record'])
            output[entry['key']] = entry
    return output


class ReplaySimulator(BaseSimulator):
    """Return precomputed energies and structures, sleeping for the time the original simulation took

    Args:
        results_path: Path to the ``results.jsonl`` file written by ``experiments/precompute/run_precompute.py``
        time_scale: Multiplier applied to the recorded runtime of each simulation step. Set to 0 to return immediately
        scratch_dir: Unused, kept for compatibility with other simulators
    """

    def __init__(self, results_path: Path | str, time_scale: float = 1.0, scratch_dir: Path | str | None = None):
        super().__init__(scratch_dir)
        self.results_path = str(Path(results_path).absolute())
        self.time_scale = time_scale

    def _get_entry(self, mol_key: str) -> dict:
        results = _load_results(self.results_path)
        if mol_key not in results:
            raise ValueError(f'No precomputed result for {mol_key}')
        return results[mol_key]

    def _wait(self, entry: dict, step_type: str, charge: int, solvent: str | None) -> float:
        """Sleep for the recorded duration of a simulation step and, if the original run failed, raise the same error"""
        runtime = None
        for step in entry['steps']:
            if step['type'] == step_type and step['charge'] == charge and step['solvent'] == solvent:
                runtime = step['runtime']
                break

        if entry['status'] != 'ok':
            # Failed in the original run: charge the time it spent before failing
            if runtime is None:
                runtime = entry['total_runtime'] - sum(s['runtime'] for s in entry['steps'])
            sleep(max(runtime, 0) * self.time_scale)
            raise ValueError(f'Precomputed simulation {entry["status"]}: {entry.get("error", "")}')

        if runtime is None:
            raise ValueError(f'No precomputed {step_type} step for {entry["key"]} with charge={charge} solvent={solvent}')
        sleep(runtime * self.time_scale)
        return runtime

    @staticmethod
    def _check_config(entry: dict, config_name: str):
        record: MoleculeRecord = entry['record']
        if not any(c.config_name == config_name for c in record.conformers):
            raise ValueError(f'No precomputed results for config={config_name}')

    def optimize_structure(self, mol_key: str, xyz: str, config_name: str, charge: int = 0, solvent: str | None = None, **kwargs) \
            -> tuple[SimResult, list[SimResult], str | None]:
        start_time = perf_counter()
        entry = self._get_entry(mol_key)
        runtime = self._wait(entry, 'optimize_structure', charge, solvent)
        self._check_config(entry, config_name)

        # Return the relaxed structure from the original run
        record: MoleculeRecord = entry['record']
        conf, energy = record.find_lowest_conformer(config_name, charge, solvent, optimized_only=True)
        result = SimResult(config_name=config_name, charge=charge, solvent=solvent, xyz=conf.xyz, energy=energy)
        return result, [], json.dumps({'runtime': perf_counter() - start_time, 'recorded_runtime': runtime, 'replay': True})

    def compute_energy(self, mol_key: str, xyz: str, config_name: str, charge: int = 0, solvent: str | None = None, forces: bool = True,
                       **kwargs) -> tuple[SimResult, str | None]:
        start_time = perf_counter()
        entry = self._get_entry(mol_key)
        runtime = self._wait(entry, 'compute_energy', charge, solvent)
        self._check_config(entry, config_name)

        # Find the conformer from the original run at this geometry with the requested energy
        record: MoleculeRecord = entry['record']
        positions = read_from_string(xyz, 'xyz').positions
        match: Conformer | None = None
        for conf in record.conformers:
            if conf.get_energy_index(config_name, charge, solvent) is None:
                continue
            conf_pos = conf.atoms.positions
            if conf_pos.shape == positions.shape and np.abs(conf_pos - positions).max() < 1e-3:
                match = conf
                break
        if match is None:
            raise ValueError(f'No precomputed energy for {mol_key} at this geometry with config={config_name} charge={charge} solvent={solvent}')

        result = SimResult(config_name=config_name, charge=charge, solvent=solvent, xyz=match.xyz,
                           energy=match.get_energy(config_name, charge, solvent))
        return result, json.dumps({'runtime': perf_counter() - start_time, 'recorded_runtime': runtime, 'replay': True})
