import json,urllib.error
from pathlib import Path
import pytest
from wccu_eval.common import canonical,digest,save_json,load_json
from wccu_eval.transport import ResponseCache
from wccu_eval.cli import main

MSG=[{'role':'user','content':'Synthetic test only'}]

def record(c):
    req=c.payload(MSG,20)
    return {'request':req,'response':{'model':'test-model','choices':[{'message':{'content':'OK'}}],
            'usage':{'prompt_tokens':10,'completion_tokens':2}}}

def test_no_network_default(tmp_path,monkeypatch):
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**k:pytest.fail('Network must not be reached'))
    c=ResponseCache(tmp_path,'test-model',1)
    with pytest.raises(FileNotFoundError):c(MSG,20)

def test_replay_full_payload_and_logical_costs(tmp_path):
    c=ResponseCache(tmp_path,'test-model',1);r=record(c);h=digest(canonical(r['request']))
    save_json(tmp_path/(h+'.json'),r)
    assert c(MSG,20)==(h,'OK');assert c(MSG,20)==(h,'OK')
    assert c.usage([h,h])=={'calls':2,'prompt_tokens':20,'output_tokens':4}
    r['request']['seed']=2;save_json(tmp_path/(h+'.json'),r)
    with pytest.raises(ValueError):ResponseCache(tmp_path,'test-model',1)(MSG,20)

def test_network_requires_explicit_safe_endpoint_and_ceiling(tmp_path):
    with pytest.raises(ValueError):ResponseCache(tmp_path,'m',1,execute=True,endpoint='http://example.com',max_new_calls=2)
    with pytest.raises(ValueError):ResponseCache(tmp_path,'m',1,execute=True,endpoint='https://example.com',max_new_calls=0)
    with pytest.raises(ValueError):ResponseCache(tmp_path,'m',1,request_options={'model':'other'})

def test_failure_redaction_and_no_automatic_retry(tmp_path,monkeypatch):
    count=[]
    def fail(*args,**kwargs):count.append(1);raise RuntimeError('DO_NOT_PERSIST_THIS_SECRET')
    monkeypatch.setattr('urllib.request.urlopen',fail)
    c=ResponseCache(tmp_path,'test-model',1,execute=True,endpoint='http://127.0.0.1:1',max_new_calls=2)
    with pytest.raises(RuntimeError):c(MSG,20)
    with pytest.raises(RuntimeError):c(MSG,20)
    assert len(count)==1
    assert all('DO_NOT_PERSIST_THIS_SECRET' not in p.read_text() for p in tmp_path.iterdir())

def test_cli_smoke_and_output_guard(tmp_path):
    jobs=Path(__file__).resolve().parents[1]/'examples/synthetic_jobs.json'
    args=['select','--jobs',str(jobs),'--policy','bundle','--budget','80','--output',str(tmp_path/'out.json')]
    assert main(args)==0
    assert load_json(tmp_path/'out.json')[0]['spent_tokens']<=80
    assert main(args)==2
