"""Rebuild the dual-track evaluation from a separately supplied official package."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('all', 'exact', 'sample', 'hybrid'), default='all')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--samples', type=int, default=8192)
    args = parser.parse_args()
    if args.repeats != 10 or args.samples != 8192:
        parser.error('evaluation-v1 reproduction requires --repeats 10 --samples 8192')
    required = ['pyproject.toml', 'src/qjlb/cli.py', 'data/adb_terms.json',
                'data/seismic.npz', 'data/flood.npz', 'scenarios/adb-base.json']
    missing = [name for name in required if not (ROOT/'benchmark'/name).is_file()]
    if missing:
        parser.error('Place the complete official v0.5.0 package in benchmark/. Missing: '+', '.join(missing))
    from run_s4_qjlb import DATA, setup, run
    modes = ('exact', 'sample', 'hybrid') if args.mode == 'all' else (args.mode,)
    # Refuse a partial restart before invoking any new timed runs.
    for mode in modes:
        if (DATA/f'evaluation-suite-13-{mode}').exists():
            parser.error('Move existing evaluation directories aside before reproducing: '+mode)
    suite = setup()
    for mode in modes:
        run(suite, mode, args.repeats, args.samples)
    complete = all((DATA/f'evaluation-suite-13-{mode}/report.json').is_file()
                   for mode in ('exact', 'sample', 'hybrid'))
    if not complete:
        print('Mode complete. Run the remaining modes to perform full verification.')
        return 0
    from verify_s4_qjlb import additional_processes, verify
    additional_processes()
    subprocess.run([sys.executable, str(ROOT/'src/s4_scientific_findings.py')], check=True)
    subprocess.run([sys.executable, str(ROOT/'src/s4_resource_probe.py')], check=True)
    # Compare the reproduced summary to its own receipts. Historical timings
    # remain unchanged in the checked-in evidence and vary with the host.
    summary = {'protocol': 'evaluation-v1', 'modes': {}}
    for mode in ('exact', 'sample', 'hybrid'):
        report = json.loads((DATA/f'evaluation-suite-13-{mode}/report.json').read_text())
        summary['modes'][mode] = {'tracks': report['tracks']}
    (DATA/'reproduced_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    status = verify()
    result = json.loads((DATA/'verification.json').read_text())
    summary['verification'] = result
    summary['historical_assertions'] = 3684
    summary['integrity_scope'] = 'Public code manifest; private archive preservation checks are excluded.'
    (DATA/'reproduction.json').write_text(json.dumps(summary, indent=2)+'\n')
    return status


if __name__ == '__main__':
    raise SystemExit(main())
