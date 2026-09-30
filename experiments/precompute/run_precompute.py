"""Precompute MOPAC/PM7 redox potentials (ACN) for a sample of the search space

Runs the same sequence of simulations ExaMol's thinker would run for
``RedoxEnergy(1, energy_config='mopac_pm7', solvent='acn')`` and records the
wall-clock time of every simulation step, so the replay simulator can reproduce
per-molecule cost.

Output is appended to ``results.jsonl`` (one line per molecule), so the run can be
restarted and will skip molecules already done.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')  # One core per molecule

import gzip
import json
import random
import signal
import socket
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter, time

from rdkit import RDLogger

from examol.simulate.ase import ASESimulator
from examol.store.models import MoleculeRecord
from examol.store.recipes import RedoxEnergy

RDLogger.DisableLog('rdApp.*')

_recipe = RedoxEnergy(1, energy_config='mopac_pm7', solvent='acn')
_simulator: ASESimulator | None = None
_timeout: float = 3600


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _init_worker(scratch: str, timeout: float):
    global _simulator, _timeout
    _simulator = ASESimulator(scratch_dir=Path(scratch) / f'w{os.getpid()}', clean_after_run=True, retain_failed=False)
    _timeout = timeout
    signal.signal(signal.SIGALRM, _alarm)


def run_molecule(smiles: str) -> dict:
    """Run every simulation needed to compute the redox potential of one molecule"""
    out = {'smiles': smiles, 'host': socket.gethostname(), 'start': time(), 'steps': []}
    start = perf_counter()
    signal.alarm(int(_timeout))
    try:
        record = MoleculeRecord.from_identifier(smiles)
        out['key'] = record.key
        out['n_atoms'] = None
        for _ in range(10):  # Recipe needs 2 relaxations + 2 solvent energies; cap loop defensively
            requests = _recipe.suggest_computations(record)
            if len(requests) == 0:
                break
            for req in requests:
                step_start = perf_counter()
                if req.optimize:
                    sim_result, steps, _ = _simulator.optimize_structure(record.key, req.xyz, req.config_name, req.charge, req.solvent)
                    record.add_energies(sim_result, steps)
                    n_steps = len(steps)
                else:
                    sim_result, _ = _simulator.compute_energy(record.key, req.xyz, req.config_name, req.charge, req.solvent)
                    record.add_energies(sim_result)
                    n_steps = 0
                out['n_atoms'] = len(sim_result.atoms)
                out['steps'].append({
                    'type': 'optimize_structure' if req.optimize else 'compute_energy',
                    'charge': req.charge, 'solvent': req.solvent,
                    'runtime': perf_counter() - step_start, 'opt_steps': n_steps,
                })
        out['value'] = _recipe.update_record(record)
        out['status'] = 'ok'
        out['record'] = record.json()
    except _Timeout:
        out['status'] = 'timeout'
    except Exception as e:
        out['status'] = 'failed'
        out['error'] = f'{type(e).__name__}: {e}'[:500]
    finally:
        signal.alarm(0)
    out['total_runtime'] = perf_counter() - start
    return out


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--search-space', default=str(Path(__file__).parents[2] / 'examples' / 'redoxmers-bebop' / 'search-space.smi'))
    parser.add_argument('--num', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--timeout', type=float, default=3600, help='Max seconds per molecule')
    parser.add_argument('--out-dir', default=str(Path(__file__).parent / 'data'))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sample the molecules deterministically and save the list
    # Only neutral molecules: RedoxEnergy assumes the starting molecule has charge 0
    from rdkit import Chem
    all_smiles = []
    for s in Path(args.search_space).read_text().splitlines():
        mol = Chem.MolFromSmiles(s.strip()) if s.strip() else None
        if mol is not None and Chem.GetFormalCharge(mol) == 0:
            all_smiles.append(s.strip())
    sample = random.Random(args.seed).sample(all_smiles, args.num)
    (out_dir / 'sample.smi').write_text('\n'.join(sample) + '\n')

    # Skip those already done
    results_path = out_dir / 'results.jsonl'
    done = set()
    if results_path.exists():
        with results_path.open() as fp:
            done = {json.loads(line)['smiles'] for line in fp if line.strip()}
    todo = [s for s in sample if s not in done]
    print(f'Sampled {len(sample)} molecules; {len(done)} already done; running {len(todo)} on {args.workers} workers', flush=True)

    wall_start = perf_counter()
    counts = {}
    with Pool(args.workers, initializer=_init_worker, initargs=(str(out_dir / 'scratch'), args.timeout), maxtasksperchild=25) as pool, \
            results_path.open('a') as fp:
        for i, result in enumerate(pool.imap_unordered(run_molecule, todo, chunksize=1)):
            fp.write(json.dumps(result) + '\n')
            fp.flush()
            counts[result['status']] = counts.get(result['status'], 0) + 1
            if (i + 1) % 20 == 0 or i + 1 == len(todo):
                elapsed = perf_counter() - wall_start
                rate = (i + 1) / elapsed
                eta_h = (len(todo) - i - 1) / rate / 3600
                print(f'{i + 1}/{len(todo)} done in {elapsed / 60:.1f} min ({rate * 60:.1f}/min) status={counts} ETA={eta_h:.2f} h', flush=True)
