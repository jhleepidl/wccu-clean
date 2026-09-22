#!/usr/bin/env python3
"""Reproduce the one support-disjoint transfer experiment from local public data.

No download, model call, answer generation, or new choice of sample/parameters.
Supply the original HotpotQA data and Qwen tokenizer JSON yourself. Hash-only
sampling exclusions are distributed; source text and vocabulary are not.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, sys, unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from wccu_eval.common import digest, frame, validate_job, save_json
from wccu_eval.tokenizer_json import LocalQwenBPE
from wccu_eval.budget_transfer import run


def normalize_source(s: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())


def select_sample(records: list[dict], exclusions: dict) -> tuple[list[dict],dict]:
    """Select from source/support identities alone, before any selector scoring."""
    used=set(exclusions['archived_query_ids']);bt=set(exclusions['banned_title_hashes']);bb=set(exclusions['banned_body_hashes'])
    valid={}
    for r in records:
        q=r['_id'];cx=r['context'];titles=[t for t,_ in cx];texts=[''.join(ss) for _,ss in cx]
        if len(cx)!=10 or len(set(titles))!=10 or not all(texts) or len(set(map(digest,texts)))!=len(texts):continue
        gold=set();invalid=False
        for t,i in r['supporting_facts']:
            if t not in titles or not isinstance(i,int) or i<0 or i>=len(cx[titles.index(t)][1]):invalid=True;break
            gold.add(titles.index(t))
        if invalid or not gold:continue
        keys=set()
        for i in gold:keys.update([('t',normalize_source(titles[i])),('b',digest(normalize_source(texts[i])))])
        valid[q]={'record':r,'gold':sorted(gold),'keys':keys}
    parent={q:q for q in valid}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    owner={}
    for q,v in valid.items():
        for k in v['keys']:
            if k in owner:
                a,b=find(q),find(owner[k])
                if a!=b:parent[max(a,b)]=min(a,b)
            else:owner[k]=q
    groups=collections.defaultdict(list)
    for q in valid:groups[find(q)].append(q)
    eligible=[]
    for members in groups.values():
        members=sorted(members);cid=digest('|'.join(members))
        keys=set().union(*(valid[q]['keys'] for q in members))
        if used.intersection(members):continue
        if any((digest(k) in bt) if tp=='t' else (k in bb) for tp,k in keys):continue
        q=min(members,key=lambda x:digest('wccu-r44-hotpot-v1|representative|'+x))
        eligible.append({'id':q,'component':cid,'type':valid[q]['record']['type']})
    eligible.sort(key=lambda x:digest('wccu-r44-hotpot-v1|component|'+x['component']))
    return eligible[:512],valid


def build_jobs(records, sample, valid, tokenizer):
    jobs=[];labels={}
    for m in sample:
        r=valid[m['id']]['record'];docs=[]
        for i,(title,sentences) in enumerate(r['context']):
            text=''.join(sentences)
            d={'id':str(i),'title':title,'text':text,'sha256':digest(text)}
            d['tokens']=len(tokenizer.encode(frame(d)));docs.append(d)
        j={'id':m['id'],'question':r['question'],'docs':docs};validate_job(j);jobs.append(j)
        labels[m['id']]={'support_indices':valid[m['id']]['gold'],'answers':[r['answer']],'component':m['component'],'type':m['type']}
    return jobs,labels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True,type=Path,help='Local hotpot_dev_distractor JSON')
    p.add_argument('--tokenizer-json',required=True,type=Path,help='Pinned Qwen3-4B tokenizer.json')
    p.add_argument('--output',required=True,type=Path,help='New output directory; never an input directory')
    a=p.parse_args();cfg=ROOT/'configs/hotpot_budget_transfer'
    if a.output.exists() and any(a.output.iterdir()):p.error('Use an empty output directory to avoid overwriting a previous result')
    audit=json.loads((cfg/'SAMPLING_AUDIT.json').read_text())
    for source,key in [(a.data,'raw_sha256'),(a.tokenizer_json,'tokenizer_sha256')]:
        if digest(source.read_bytes())!=audit[key]:p.error(f'Unexpected SHA-256 for {source.name}; use the recorded source revision')
    records=json.loads(a.data.read_text(encoding='utf-8'))
    exclusions={k:json.loads((cfg/name).read_text()) for k,name in {'archived_query_ids': 'EXCLUDED_QUERY_IDS.json', 'banned_title_hashes': 'EXCLUDED_SOURCE_TITLES.json', 'banned_body_hashes': 'EXCLUDED_SOURCE_BODIES.json'}.items()}
    sample,valid=select_sample(records,exclusions)
    expected=json.loads((cfg/'SAMPLE.json').read_text())
    if sample!=expected:raise RuntimeError('Regenerated sample differs; no outcome has been computed')
    tokenizer=LocalQwenBPE(a.tokenizer_json)
    jobs,labels=build_jobs(records,sample,valid,tokenizer)
    a.output.mkdir(parents=True,exist_ok=True)
    save_json(a.output/'JOBS.json',jobs);save_json(a.output/'LABELS.json',labels)
    run(jobs,labels,a.output)

if __name__=='__main__':main()
