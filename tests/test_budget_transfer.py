import importlib.util
from pathlib import Path
import numpy as np
from wccu_eval.budget_transfer import bootstrap
from wccu_eval.common import digest

spec=importlib.util.spec_from_file_location('transfer_script',Path(__file__).parents[1]/'scripts/reproduce_hotpot_budget_transfer.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def record(q, prefix, first=None):
    return {'_id':q,'question':'Which sources?', 'type':'bridge','answer':'not passed to selection',
            'context':[(first if i==0 and first else f'{prefix} source {i}',[f'{prefix} text {i}']) for i in range(10)],
            'supporting_facts':[(first if first else f'{prefix} source 0',0),(f'{prefix} source 1',0)]}

EMPTY={'archived_query_ids':[],'banned_title_hashes':[],'banned_body_hashes':[]}

def test_shared_support_creates_one_component():
    a=record('a','A');b=record('b','B',first='A source 0')
    selected,_=mod.select_sample([a,b],EMPTY)
    assert len(selected)==1

def test_prior_candidate_support_exclusion():
    a=record('a','A');ex=dict(EMPTY,banned_title_hashes=[digest(mod.normalize_source('A source 0'))])
    assert mod.select_sample([a],ex)[0]==[]

def test_invalid_support_not_evaluated():
    a=record('a','A');a['supporting_facts'][0]=('missing',0)
    assert mod.select_sample([a],EMPTY)[0]==[]

def test_primary_bootstrap_is_paired_and_fixed():
    x=np.array([1,1,0,-1,0,1],float);st=np.array(['b']*4+['c']*2)
    a=bootstrap(x,st,repetitions=2000);b=bootstrap(x,st,repetitions=2000)
    assert a==b and a['mean']==x.mean() and a['ci95'][0]<=a['mean']<=a['ci95'][1]
