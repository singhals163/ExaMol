"""Check that replaying precomputed results reproduces the precomputed redox potentials"""
import json
import sys
from pathlib import Path

from examol.simulate.replay import ReplaySimulator
from examol.store.models import MoleculeRecord
from examol.store.recipes import RedoxEnergy

path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / 'data' / 'results.jsonl')
recipe = RedoxEnergy(1, energy_config='mopac_pm7', solvent='acn')
sim = ReplaySimulator(path, time_scale=0)

n_ok = n_bad = n_fail_ok = 0
for line in path.open():
    entry = json.loads(line)
    if 'key' not in entry:
        continue
    record = MoleculeRecord.from_identifier(entry['smiles'])
    try:
        for _ in range(10):  # Same loop the thinker follows
            requests = recipe.suggest_computations(record)
            if not requests:
                break
            for req in requests:
                if req.optimize:
                    res, steps, _ = sim.optimize_structure(record.key, req.xyz, req.config_name, req.charge, req.solvent)
                    record.add_energies(res, steps)
                else:
                    res, _ = sim.compute_energy(record.key, req.xyz, req.config_name, req.charge, req.solvent)
                    record.add_energies(res)
        value = recipe.update_record(record)
    except ValueError as e:
        if entry['status'] != 'ok':
            n_fail_ok += 1  # Failure replayed as failure
        else:
            n_bad += 1
            print('replay failed', entry['key'], e)
        continue
    if entry['status'] == 'ok' and abs(value - entry['value']) < 1e-8:
        n_ok += 1
    else:
        n_bad += 1
        print('mismatch', entry['key'], entry['status'], value, entry.get('value'))
print(f'matched={n_ok} failures_reproduced={n_fail_ok} bad={n_bad}')
