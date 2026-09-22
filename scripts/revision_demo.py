#!/usr/bin/env python3
"""Run the authored CPU selection-validity cases, without a model or network."""
import argparse
from pathlib import Path
from wccu_eval.revisions import KINDS,build_case,cpu
from wccu_eval.common import save_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--cases-per-kind',type=int,default=8)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.cases_per_kind<1 or a.output.exists():p.error('Use positive case count and a new output path')
    cases=[build_case(k,i,'test') for k in KINDS for i in range(a.cases_per_kind)]
    initial,rows=cpu(cases)
    save_json(a.output,{'base_cases':len(cases),'revision_cases':len(rows),'initial':initial,'rows':rows,
        'scope':'Authored CPU mechanism cases; no model completion or action-safety estimate.'})
    print(f'{len(cases)} base cases; {len(rows)} revision cases; no model calls')
if __name__=='__main__':main()
