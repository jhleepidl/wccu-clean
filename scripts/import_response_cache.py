#!/usr/bin/env python3
"""Import privately supplied exact request/response pairs into the clean cache.

Supports a request file with {"request": ...}, or a raw request object. Raw provider
responses must have the same filename. This script never sends a network request.
"""
import argparse
from pathlib import Path
from wccu_eval.common import load_json, save_json, canonical, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--requests', type=Path, required=True)
    p.add_argument('--responses', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not a.requests.is_dir() or not a.responses.is_dir():p.error('Existing input directories required')
    count = 0
    for f in sorted(a.requests.glob('*.json')):
        wrapper = load_json(f); request = wrapper.get('request', wrapper)
        h = digest(canonical(request))
        if f.stem != h:raise ValueError(f'Payload hash mismatch: {f.name}')
        response = load_json(a.responses/f.name)
        if response.get('model') != request.get('model'):raise ValueError('Model mismatch')
        dst = a.output/(h+'.json');record = {'request':request,'response':response}
        if dst.exists() and load_json(dst) != record:raise ValueError('Existing cache disagrees; nothing overwritten')
        save_json(dst,record);count+=1
    if not count:raise ValueError('No request pairs found')
    print(f'Imported {count} exact payloads; no network calls')

if __name__ == '__main__':main()
