"""Measure actual child-process RSS and startup imports on representative cases."""
import json
import platform
from pathlib import Path
import re
import subprocess
import sys
from qjlb.cli import load_model

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/s4'


def main():
    rows=[]
    for case in ('case-005','case-006'):
        config=DATA/'suite-13/cases'/f'{case}.json'
        model,_=load_model(config)
        request={'config_file':str(config),'samples':8192,'seed':8761,'run_id':'resource-'+case,
            'problem_id':model.fingerprint,'targets':list(model.metric_names)}
        path=DATA/'resource-request.json'; path.write_text(json.dumps(request)+'\n')
        for mode in ('exact','sample','hybrid'):
            mac = platform.system() == 'Darwin'
            command=['/usr/bin/time','-l' if mac else '-v',sys.executable,str(ROOT/'team_solver.py'),'--request',str(path),'--output',str(DATA/'resource-result.json'),'--mode',mode]
            r=subprocess.run(command,capture_output=True,text=True,check=True)
            pattern = r'(\d+)\s+maximum resident set size' if mac else r'Maximum resident set size \(kbytes\):\s*(\d+)'
            rss=int(re.search(pattern,r.stderr)[1]) * (1 if mac else 1024)
            rows.append({'case':case,'track':model.config['track'],'mode':mode,'peak_resident_bytes':rss,'under_4_gib':rss<4*1024**3})
    (DATA/'resource_probe.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
