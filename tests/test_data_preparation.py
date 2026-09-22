import pytest
from wccu_eval.data import prepare_candidates
from wccu_eval.common import digest,frame

def test_2wiki_labels_separate_and_hash_check():
    d={'id':'0','title':'Aster Observatory','text':'Built in Cedar Bay.','sha256':digest('Built in Cedar Bay.')}
    encode=lambda s:list(s.encode('utf-8'))
    entry={'id':'synthetic','split':'test','stratum':'chain','documents':[{k:d[k] for k in ('id','sha256')} | {'tokens':len(encode(frame(d)))}]}
    r={'_id':'synthetic','question':'Where?','context':[['Aster Observatory',['Built in Cedar Bay.']]],'supporting_facts':[['Aster Observatory',0]],'answer':'Cedar Bay'}
    with pytest.raises(ValueError):prepare_candidates([r],[entry],encode,dataset='2wiki')
    jobs,labels=prepare_candidates([r],[entry],encode,dataset='2wiki',aliases=[])
    assert 'answers' not in jobs[0] and labels['synthetic']['support_indices']==[0]
    r['context'][0][1]=['Changed.']
    with pytest.raises(ValueError):prepare_candidates([r],[entry],encode,dataset='2wiki',aliases=[])
