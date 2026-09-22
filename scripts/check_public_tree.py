#!/usr/bin/env python3
"""Conservative public-tree hygiene check, not a complete secret-scanning product."""
from pathlib import Path
import re,sys

ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','.venv','.pytest_cache','__pycache__','build','dist'}
BLOCKED={'.parquet','.zip','.pdf','.png','.jpg','.sqlite','.db','.bin','.safetensors','.pt','.pem','.key','.ttf','.otf','.woff','.woff2'}
PATTERNS=[re.compile(r'\bsk-[A-Za-z0-9_-]{24,}\b'),re.compile(r'\bgh[pousr]_[A-Za-z0-9]{25,}\b'),
          re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]

def main():
    errors=[];count=0
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT)
        if any(x in SKIP or x.endswith('.egg-info') for x in rel.parts):continue
        if p.is_symlink():errors.append(f'Symlink: {rel}');continue
        if not p.is_file():continue
        count+=1
        if p.suffix.lower() in BLOCKED or p.name=='.env':errors.append(f'Non-code/private artifact: {rel}')
        if p.stat().st_size>2_000_000:errors.append(f'Unexpectedly large file: {rel}')
        try:text=p.read_text(encoding='utf-8')
        except UnicodeError:errors.append(f'Binary file: {rel}');continue
        if any(r.search(text) for r in PATTERNS):errors.append(f'Potential credential: {rel}')
        # Windows is also considered; examples may use relative private-data paths.
        if p.name != 'check_public_tree.py' and re.search(r'/(?:mnt/data|home/oai|Users/[^/\s]+)',text):errors.append(f'Environment-specific path: {rel}')
    for e in errors:print(e,file=sys.stderr)
    print(f'Inspected {count} text files; {len(errors)} hygiene findings')
    return int(bool(errors))
if __name__=='__main__':raise SystemExit(main())
