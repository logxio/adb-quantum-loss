"""Reproduce S4 suites and official evaluation runs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/s4'


def setup():
    DATA.mkdir(parents=True, exist_ok=True)
    for seed in (20261122, 20261123, 20261124):
        dest = DATA/f'holdout-{seed}'
        if not dest.exists():
            subprocess.run([sys.executable,'-m','qjlb.cli','holdout','--seed',str(seed),'--data',str(ROOT/'benchmark/data'),'--output',str(dest)],check=True)
    dest = DATA/'suite-13'
    if not dest.exists():
        shutil.copytree(DATA/'holdout-20261122',dest)
        manifest = json.loads((dest/'suite.json').read_text())
        for src in sorted((ROOT/'benchmark/scenarios').glob('*.json')):
            shutil.copyfile(src,dest/'cases'/src.name)
            manifest['cases'].append('cases/'+src.name)
            manifest['files']['cases/'+src.name] = hashlib.sha256(src.read_bytes()).hexdigest()
        (dest/'suite.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return dest


def run(suite, mode, repeats=10, samples=8192):
    output = DATA/f'evaluation-{suite.name}-{mode}'
    subprocess.run([sys.executable,'-m','qjlb.cli','evaluate',str(suite),'--output',str(output),
        '--execution','hybrid' if mode=='hybrid' else 'classical_cpu','--trusted-code',
        '--repeats',str(repeats),'--samples',str(samples),'--',sys.executable,str(ROOT/'team_solver.py'),'--mode',mode],check=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', nargs='?', default='suite-13')
    parser.add_argument('mode', nargs='?', choices=('exact', 'sample', 'hybrid'))
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--samples', type=int, default=8192)
    args = parser.parse_args()
    setup()
    if args.mode:
        run(DATA/args.suite, args.mode, args.repeats, args.samples)
