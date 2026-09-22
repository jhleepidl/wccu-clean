import copy,json
from pathlib import Path
import pytest
from wccu_eval.common import digest,load_json,normalize,score_answer,validate_job
from wccu_eval.selection import select_records,POLICIES
from wccu_eval.reader import initial_answer,dynamic_reference,compose_selective,terminal_parser
from wccu_eval.statistics import paired_bootstrap
from wccu_eval.answer_audit import quality
from wccu_eval.experiment import evaluate,aggregate

ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture
def job():return load_json(ROOT/'examples/synthetic_jobs.json')[0]

@pytest.mark.parametrize('policy',POLICIES)
def test_budget_and_canonical_order(job,policy):
    r=select_records(job,policy,80)
    assert r['selected_indices']==sorted(set(r['selected_indices']))
    if policy!='full':assert r['spent_tokens']<=80
    else:assert len(r['selected_indices'])==len(job['docs'])

@pytest.mark.parametrize('policy',('rank','singleton','bundle','iterative','mmr'))
def test_exclusion_pool_and_zero_budget(job,policy):
    assert select_records(job,policy,0)['selected_indices']==[]
    assert set(select_records(job,policy,1000,pool=[1,3])['selected_indices']) <= {1,3}

@pytest.mark.parametrize('policy',('rank','singleton','bundle','iterative','mmr'))
def test_body_deduplication(job,policy):
    d=copy.deepcopy(job['docs'][0]);d['id']='duplicate';job['docs'].append(d)
    out=select_records(job,policy,1000)['selected_indices']
    assert len({job['docs'][i]['sha256'] for i in out})==len(out)

def test_no_label_in_reader_job(job):
    job['answers']=['not permitted']
    with pytest.raises(ValueError):validate_job(job)

def test_corrupted_body(job):
    job['docs'][0]['text']+=' corruption'
    with pytest.raises(ValueError):select_records(job)

@pytest.mark.parametrize('raw,expected,ok',[
    ('Reasoning. So the answer is: Cedar Bay.','Cedar Bay',True),
    ('So the answer is: UNKNOWN','UNKNOWN',True),
    ('Answer: Cedar Bay','Answer: Cedar Bay',False),
    ('So the answer is: A\nSo the answer is: B','So the answer is: A\nSo the answer is: B',False),
    ('So the answer is: Cedar Bay\nextra','So the answer is: Cedar Bay\nextra',False)])
def test_terminal_parser(raw,expected,ok):assert terminal_parser(raw)==(expected,ok)

def test_metric_preserves_official_normalization():
    assert normalize('The A-Team!')=='ateam'
    assert score_answer('Munich, Bavaria',['Munich'])['em']==0
    assert quality('April 3, 2013',['3 April 2013'])['date_normalized_em']==1
    assert quality('2013',['3 April 2013'])['date_normalized_em']==0
    assert quality('April 3, 2013',['3 April 2013'])['em']==0

def test_literal_trigger():
    f={'answer':'UNKNOWN.','parser_ok':True,'selected':[1],'request_ids':['f']}
    r={'answer':'Cedar Bay','parser_ok':True,'request_ids':['r','s']}
    out=compose_selective(f,r);assert out['trigger'] and out['request_ids']==['f','r','s']
    f['answer']='No answer known';f['parser_ok']=False
    assert not compose_selective(f,None)['trigger']
    f['answer']='UNKNOWN'
    with pytest.raises(ValueError):compose_selective(f,None)

def test_bootstrap_pairing_and_ratio():
    a={'b':{'stratum':'x','em':1,'cost':2},'a':{'stratum':'y','em':0,'cost':2}}
    b={'a':{'stratum':'y','em':0,'cost':4},'b':{'stratum':'x','em':1,'cost':4}}
    assert paired_bootstrap(a,b,'em',replicates=50)['ci95']==[0,0]
    assert paired_bootstrap(a,b,'cost',ratio=True,replicates=50)['ci95']==[.5,.5]
    b['a']['stratum']='z'
    with pytest.raises(ValueError):paired_bootstrap(a,b,'em')

def test_full_dynamic_reader_and_pipeline(job):
    class Mock:
        def __init__(self):self.calls=[]
        def __call__(self,msg,limit):
            self.calls.append((msg,limit));return str(len(self.calls)), 'So the answer is: Cedar Bay'
        def usage(self,ids):return {'calls':len(ids),'prompt_tokens':100*len(ids),'output_tokens':8*len(ids)}
    call=Mock();ref=dynamic_reference(job,call,'Synthetic example')
    assert len(ref['request_ids'])==2 and [c[1] for c in call.calls]==[128,1280]
    assert ref['answer']=='Cedar Bay'
    labels={job['id']:{'answers':['Cedar Bay'],'support_indices':[0,1]}}
    rows=evaluate([job],labels,Mock(),'Synthetic example',policies=('full',),budgets=(80,))
    assert len(rows)==3 and all(r['em']==1 for r in rows)
    assert aggregate(rows)[0]['n']==1


def test_invalid_labels_do_not_spend_calls(job):
    def call(*args):pytest.fail("Must validate labels before a model call")
    with pytest.raises(ValueError):evaluate([job],{job['id']:{'answers':[],'support_indices':[0]}},call,'')
