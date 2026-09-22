"""Run the fixed transfer experiment. No network, model calls, or parameter search."""
from __future__ import annotations
import argparse, json, time, hashlib
from pathlib import Path
import numpy as np
from wccu_eval.selection import get_features, select, scale
from wccu_eval.common import validate_job


def bootstrap(x, strata, seed=2026092144, repetitions=20000):
    x=np.asarray(x,dtype=float);rng=np.random.default_rng(seed)
    groups=[np.flatnonzero(strata==s) for s in sorted(set(strata))]
    bs=[]
    for start in range(0,repetitions,1000):
        m=min(1000,repetitions-start);b=np.zeros(m)
        for ids in groups:b+=x[ids[rng.integers(0,len(ids),size=(m,len(ids)))]] .sum(axis=1)/len(x)
        bs.extend(b)
    return {'mean':float(x.mean()),'ci95':[float(v) for v in np.quantile(bs,[.025,.975])],'n':len(x)}


def run(jobs, labels, output):
    start=time.perf_counter();output.mkdir(parents=True,exist_ok=True)
    rows=[];raw=[];policies={'rank':'rank','singleton':'bridge_atomic','bundle':'bridge_bundle'}
    for j in jobs:
        validate_job(j);lab=labels[j['id']];gold=set(lab['support_indices'])
        f=get_features(j);rel=scale(f['bm25'])
        cgold=sum(j['docs'][i]['tokens'] for i in gold);total=sum(d['tokens'] for d in j['docs'])
        for cap in (512,2048):
            for name, policy in policies.items():
                r=select(j,f,rel,policy,cap);inds=set(r['selected_indices'])
                assert r['spent_tokens']==sum(j['docs'][i]['tokens'] for i in inds)<=cap
                rows.append({'id':j['id'],'component':lab['component'],'type':lab['type'],'cap':cap,'method':name,
                    'selected_indices':r['selected_indices'],'spent_tokens':r['spent_tokens'],
                    'complete_support':int(gold<=inds),'support_recall':len(inds&gold)/len(gold),
                    'support_cost':cgold,'full_context_cost':total,'support_feasible':cgold<=cap,
                    'full_context_feasible':total<=cap,'steps':r['steps']})
        raw.append({'id':j['id'],'component':lab['component'],'type':lab['type'],
                    'support_cost':cgold,'full_context_cost':total,'gold_support_count':len(gold)})
    strata=np.array([labels[j['id']]['type'] for j in jobs]);lookup={(r['id'],r['cap'],r['method']):r for r in rows}
    def vec(cap,m,k='complete_support'):return np.array([lookup[(j['id'],cap,m)][k] for j in jobs],float)
    contrast={};summary=[]
    for cap in (512,2048):
        for m in policies:
            relevant=[r for r in rows if r['cap']==cap and r['method']==m]
            v=vec(cap,m)
            summary.append({'cap':cap,'method':m,'n':len(v),'complete_support_count':int(v.sum()),'complete_support_rate':float(v.mean()),
                'mean_support_recall':float(vec(cap,m,'support_recall').mean()),'mean_source_tokens':float(vec(cap,m,'spent_tokens').mean()),
                'support_feasible_n':sum(r['support_feasible'] for r in relevant),'full_context_feasible_n':sum(r['full_context_feasible'] for r in relevant)})
        d=vec(cap,'bundle')-vec(cap,'singleton');s=bootstrap(d,strata)
        s.update(gains=int((d>0).sum()),losses=int((d<0).sum()),ties=int((d==0).sum()))
        contrast[str(cap)]=s
    delta=(vec(512,'bundle')-vec(512,'singleton'))-(vec(2048,'bundle')-vec(2048,'singleton'))
    primary=bootstrap(delta,strata)
    primary['positive_budget_dependence_supported']=bool(primary['ci95'][0]>0 and contrast['512']['ci95'][0]>0)
    # Prespecified descriptive splits, with all rows retained in the primary.
    subgroups={}
    for s in sorted(set(strata)):
        mask=strata==s
        subgroups[s]={'n':int(mask.sum()),'primary':bootstrap(delta[mask],strata[mask]),'budgets':{str(c):bootstrap((vec(c,'bundle')-vec(c,'singleton'))[mask],strata[mask]) for c in (512,2048)}}
    feasible={}
    for cap in (512,2048):
        mask=vec(cap,'bundle','support_cost')<=cap
        feasible[str(cap)]={'n':int(mask.sum()),'contrast':bootstrap((vec(cap,'bundle')-vec(cap,'singleton'))[mask],strata[mask])}
    result={'name':'support_disjoint_hotpot_budget_transfer','sample_n':len(jobs),'bootstrap':{'seed':2026092144,'repetitions':20000,'strata_weights':'observed component proportions','method':'paired percentile'},
        'summary':summary,'bundle_minus_singleton':contrast,'primary_interaction':primary,'by_type_descriptive':subgroups,'feasible_only_descriptive':feasible,
        'full_context':{'mean_tokens':float(np.mean([r['full_context_cost'] for r in raw])),'max_tokens':max(r['full_context_cost'] for r in raw),'coverage_upper_bound':1.0,'reader_evaluated':False},
        'runtime_seconds':time.perf_counter()-start,'new_model_calls':0,'answer_em_measured':False}
    for name,obj in [('PER_QUERY.json',rows),('SUMMARY.json',result)]:
        (output/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['sample_n','summary','bundle_minus_singleton','primary_interaction','runtime_seconds']},indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run(json.loads(a.jobs.read_text()),json.loads(a.labels.read_text()),a.output)
