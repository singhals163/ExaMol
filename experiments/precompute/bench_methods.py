"""Time alternative recipes (MOPAC vs xTB, adiabatic vs vertical) on the same molecules"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import json, sys
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter
import run_precompute as rp
from examol.store.recipes import RedoxEnergy

CONFIGS = {
    'mopac_adiabatic': RedoxEnergy(1, 'mopac_pm7', solvent='acn'),
    'mopac_vertical': RedoxEnergy(1, 'mopac_pm7', vertical=True, solvent='acn'),
    'xtb_adiabatic': RedoxEnergy(1, 'xtb', solvent='acn'),
    'xtb_vertical': RedoxEnergy(1, 'xtb', vertical=True, solvent='acn'),
}

def job(args):
    name, smiles = args
    rp._recipe = CONFIGS[name]
    out = rp.run_molecule(smiles)
    out.pop('record', None)
    out['method'] = name
    return out

if __name__ == '__main__':
    smiles = [json.loads(l)['smiles'] for l in open('data/results.jsonl')][:int(sys.argv[1])]
    tasks = [(n, s) for s in smiles for n in CONFIGS]
    with Pool(4, initializer=rp._init_worker, initargs=('/tmp/claude-20001/-users-vthurime/3cd0ed85-5dca-4dcc-8740-ec1f80773f14/scratchpad/bench', 1800)) as p, open('data/bench.jsonl', 'w') as fp:
        for r in p.imap_unordered(job, tasks):
            fp.write(json.dumps(r) + '\n'); fp.flush()
