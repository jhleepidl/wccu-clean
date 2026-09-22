#!/usr/bin/env python3
"""Prepare local benchmark data; no downloading, inference, or implicit licensing."""
import argparse
from pathlib import Path
from wccu_eval.common import load_json, save_json
from wccu_eval.data import read_records, prepare_candidates


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['2wiki', 'musique'], required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--aliases', type=Path, help='Official 2Wiki id_aliases.json JSONL file')
    p.add_argument('--tokenizer', type=Path, required=True, help='Local pinned tokenizer.json, not a model identifier')
    p.add_argument('--split', choices=['test', 'dev'], default='test')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): p.error('Use a new output directory')
    try:
        from tokenizers import Tokenizer
    except ImportError:
        p.error('Install the tokenizer extra first: pip install -e ".[tokenizer]"')
    tokenizer = Tokenizer.from_file(str(a.tokenizer))
    jobs, labels = prepare_candidates(read_records(a.input), load_json(a.manifest),
        lambda s: tokenizer.encode(s, add_special_tokens=False).ids, dataset=a.dataset,
        aliases=read_records(a.aliases) if a.aliases else None, split=a.split)
    a.output.mkdir(parents=True)
    save_json(a.output / 'jobs.json', jobs)
    save_json(a.output / 'labels.json', labels)
    print(f'Prepared {len(jobs)} jobs; evaluation labels stored separately')

if __name__ == '__main__': main()
