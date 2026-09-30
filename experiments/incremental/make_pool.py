"""Freeze a snapshot of the precomputed results as the search space and ground truth for campaigns

Writes to ``pools/<name>/``:
    - ``results.jsonl``: precomputed results used by the replay simulator (successful molecules only)
    - ``search-space.smi``: SMILES of those molecules
    - ``ground-truth.csv``: key, SMILES, redox potential, and total simulation time for each molecule
"""
import json
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument('name')
parser.add_argument('--source', default=str(Path(__file__).parents[1] / 'precompute' / 'data' / 'results.jsonl'))
args = parser.parse_args()

out_dir = Path(__file__).parent / 'pools' / args.name
out_dir.mkdir(parents=True, exist_ok=False)

seen = set()
n = 0
with open(args.source) as fi, (out_dir / 'results.jsonl').open('w') as fr, \
        (out_dir / 'search-space.smi').open('w') as fs, (out_dir / 'ground-truth.csv').open('w') as fg:
    print('key,smiles,value,sim_time', file=fg)
    for line in fi:
        entry = json.loads(line)
        if entry['status'] != 'ok' or entry['key'] in seen:
            continue
        seen.add(entry['key'])
        fr.write(line if line.endswith('\n') else line + '\n')
        print(entry['smiles'], file=fs)
        print(f'{entry["key"]},{entry["smiles"]},{entry["value"]},{entry["total_runtime"]}', file=fg)
        n += 1
print(f'Wrote {n} molecules to {out_dir}')
