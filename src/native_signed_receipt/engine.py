"""CPU research operator matching reviewed tau3 single-query BM25 semantics.

All snapshot ingestion, global vocabulary IDF recalculation, retained index bytes,
full ranking work, fallback work and receipt bytes must be accounted separately.
This is not a new BM25 algorithm, a production search engine, transaction isolation,
an adversarial certificate, or proof of model task accuracy. Strict positive
margin checks use outward floating-point slack, not machine-verified intervals.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Iterable
import numpy as np
from .rank_bm25_reviewed import BM25Okapi

OPERATOR = 'tau3-single-query-reviewed-bm25-v022-k1=1.5-b=.75-eps=.25-order=input-v1'
BOUND_VERSION = 'signed-additive-drift-v1'
FP_SLACK = 1e-11

def encode(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

@dataclass(frozen=True)
class Plan:
    query: str
    k: int = 10
    operator: str = OPERATOR
    exposure: str = 'ordered-id-score-text-without-native-timing-v1'
    def __post_init__(self):
        if not isinstance(self.query, str) or type(self.k) is not int or self.k < 0:
            raise ValueError('String query and nonnegative integer k required')
        if self.operator != OPERATOR or self.exposure != 'ordered-id-score-text-without-native-timing-v1':
            raise ValueError('Unsupported operator/exposure; never reuse this bound for another plan')
    @property
    def terms(self): return tuple(self.query.lower().split())
    @property
    def identity(self): return digest(encode(asdict(self)))

@dataclass(frozen=True)
class Stats:
    avgdl: float
    effective_idf: tuple[float, ...]  # One entry PER token, including repeats.

class Snapshot:
    """Trusted immutable input text/order; complete index rebuilt once per epoch.

    The scorer and postings are private runtime state. Mutation by arbitrary
    Python code is out of scope. No pickle, unsafe loading, or source downloading.
    """
    def __init__(self, documents: Iterable[dict], scope='tau3-kb'):
        documents = tuple(documents)
        self.ids = tuple(d['id'] for d in documents)
        if not documents or len(set(self.ids)) != len(self.ids) or not all(isinstance(i,str) for i in self.ids):
            raise ValueError('Nonempty unique string document IDs required')
        self.texts = tuple(d.get('text') or d.get('content') or '' for d in documents)
        if not all(isinstance(t,str) for t in self.texts): raise ValueError('String texts required')
        self.scope = scope
        self.position = MappingProxyType({key:i for i,key in enumerate(self.ids)})
        self.hashes = tuple(digest(t.encode()) for t in self.texts)
        self.identity = digest(encode({'scope':scope,'operator':OPERATOR,'ids':self.ids,'hashes':self.hashes}))
        tokens = [t.lower().split() for t in self.texts]
        if not any(tokens): raise ValueError('No vocabulary: native BM25 is undefined')
        self._scorer = BM25Okapi(tokens)
        if self._scorer.avgdl <= 0 or not math.isfinite(self._scorer.average_idf):
            raise ValueError('Nonfinite/empty statistics')
        self.postings = {}
        for i, freq in enumerate(self._scorer.doc_freqs):
            for term in freq: self.postings.setdefault(term,set()).add(i)
        self.postings = MappingProxyType({q:frozenset(v) for q,v in self.postings.items()})
        self.input_bytes = sum(len(t.encode()) for t in self.texts)
        self.index_serialized_bytes = len(encode({'ids':self.ids,'hashes':self.hashes,
            'tf':self._scorer.doc_freqs,'lengths':self._scorer.doc_len,'idf':self._scorer.idf,
            'postings':{q:sorted(v) for q,v in self.postings.items()}}))
    def stats(self, plan):
        return Stats(self._scorer.avgdl, tuple(self._scorer.idf.get(q,0.) for q in plan.terms))
    def scores(self, plan, indices=None):
        values = self._scorer.get_scores(plan.terms) if indices is None else self._scorer.get_batch_scores(plan.terms,list(indices))
        if not all(math.isfinite(float(v)) for v in values): raise ValueError('Nonfinite score; no certificate')
        return [float(v) for v in values]
    def hits(self, plan):
        hits=set()
        for q in plan.terms: hits.update(self.postings.get(q,()))
        return sorted(hits)
    def payload(self, selected):
        return encode([{'id':key,'score':score,'text':self.texts[self.position[key]]} for key,score in selected])

@dataclass(frozen=True)
class Transition:
    before: str
    after: str
    scope: str
    changed_ids: tuple[str, ...]
    complete: bool = True
    @classmethod
    def from_snapshots(cls, old: Snapshot, new: Snapshot, complete=True):
        # Full snapshot comparison is SHARED ingestion work, NOT a free oracle.
        if old.scope != new.scope: raise ValueError('Scope mismatch')
        changes = tuple(sorted(key for key in set(old.ids)|set(new.ids)
                    if key not in old.position or key not in new.position or
                    old.hashes[old.position[key]] != new.hashes[new.position[key]]))
        return cls(old.identity, new.identity, new.scope, changes, complete)

@dataclass(frozen=True)
class Receipt:
    plan_hash: str
    scope: str
    snapshot_hash: str
    selected: tuple[tuple[str,float], ...]
    stats: Stats
    tail_upper: float | None
    payload_sha256: str
    bound_version: str = BOUND_VERSION
    @property
    def metadata_bytes(self): return len(encode(asdict(self)))

@dataclass(frozen=True)
class Decision:
    receipt: Receipt
    doc_scores: int
    ranked_entries: int
    bound_computes: int
    reason: str
    payload_unchanged: bool | None


def _rank(snapshot, values):
    # Stable original input order breaks ties, including all-zero/negative scores.
    return sorted(values, key=lambda x:(-x[1],snapshot.position[x[0]]))


def issue(snapshot: Snapshot, plan: Plan, reason='initial_exact', prior=None, extra_scores=0) -> Decision:
    if not plan.terms or not plan.k:
        selected=();tail=None;scored=0;ranked=[]
    else:
        ranked=_rank(snapshot,list(zip(snapshot.ids,snapshot.scores(plan))))
        selected=tuple(ranked[:plan.k]);tail=max((v for _,v in ranked[plan.k:]),default=None)
        scored=len(snapshot.ids)
    payload=digest(snapshot.payload(selected))
    receipt=Receipt(plan.identity,snapshot.scope,snapshot.identity,selected,snapshot.stats(plan),tail,payload)
    return Decision(receipt,scored+extra_scores,len(ranked),0,reason,
                    None if prior is None else prior.payload_sha256==payload)


def signed_drift_upper(old: Stats, new: Stats, k1:float=1.5) -> float:
    """Uniform absolute score drift bound for unchanged tf and document length.

    h_A=f/(f+c+z/A), c=k1(1-b)>=0. For A,A'>0,
    |h_A'-h_A| <= |sqrt(A')-sqrt(A)|/(sqrt(A')+sqrt(A)).
    Contribution difference <= (k1+1)(|idf'-idf| + |idf| * ratio).
    Sum repeated query tokens. The bound covers negative, zero and sign flips.
    Effective IDFs are recomputed from the WHOLE vocabulary by the shared index.
    """
    if old.avgdl<=0 or new.avgdl<=0 or len(old.effective_idf)!=len(new.effective_idf): return math.inf
    values=(old.avgdl,new.avgdl,*old.effective_idf,*new.effective_idf)
    if not all(math.isfinite(x) for x in values):return math.inf
    a,b=math.sqrt(old.avgdl),math.sqrt(new.avgdl)
    ratio=abs(b-a)/(b+a)
    value=(k1+1)*math.fsum(abs(v-u)+abs(u)*ratio for u,v in zip(old.effective_idf,new.effective_idf))
    # Equality is a real zero bound; unchanged-stat fast path is exact arithmetic.
    if value==0:return 0.
    magnitude=(k1+1)*math.fsum(abs(x) for x in (*old.effective_idf,*new.effective_idf))
    if len(old.effective_idf)>4096:return math.inf
    return math.nextafter(value+FP_SLACK*max(1.,value,magnitude)*max(1,len(old.effective_idf)),math.inf)


def advance(snapshot: Snapshot, plan: Plan, old: Receipt, transition: Transition) -> Decision:
    if (old.plan_hash!=plan.identity or old.scope!=snapshot.scope or transition.scope!=snapshot.scope
        or transition.before!=old.snapshot_hash or transition.after!=snapshot.identity
        or not transition.complete or old.bound_version!=BOUND_VERSION):
        return issue(snapshot,plan,'provenance_or_plan_fallback',old)
    if not plan.terms or not plan.k:return issue(snapshot,plan,'blank_or_zero_k',old)
    if transition.before==transition.after:
        return Decision(old,0,0,0,'identical_snapshot',True)
    now=snapshot.stats(plan)
    # Exclude deleted old winners; include EVERY new/modified document.
    ids={key for key,_ in old.selected if key in snapshot.position}
    ids.update(key for key in transition.changed_ids if key in snapshot.position)
    positions=sorted(snapshot.position[key] for key in ids)
    ranked=_rank(snapshot,[(snapshot.ids[i],v) for i,v in zip(positions,snapshot.scores(plan,positions))])
    selected=tuple(ranked[:plan.k])
    delta=signed_drift_upper(old.stats,now)
    tail=None if old.tail_upper is None else old.tail_upper+delta
    threshold=selected[-1][1] if selected else -math.inf
    slack=FP_SLACK*max(1.,abs(threshold),abs(tail or 0.))
    covers_all=len(ranked)==len(snapshot.ids)
    enough=len(selected)==min(plan.k,len(snapshot.ids))
    certified=covers_all or enough and (tail is None or threshold>tail+slack)
    if not certified:
        # Count unsuccessful partial scoring too; never hide fallback overhead.
        exact=issue(snapshot,plan,'ambiguous_margin_exact_fallback',old,len(positions))
        return Decision(exact.receipt,exact.doc_scores,exact.ranked_entries+len(ranked),1,
                        exact.reason,exact.payload_unchanged)
    tails=[v for _,v in ranked[plan.k:]]
    if tail is not None and not covers_all:tails.append(tail)
    new_tail=max(tails,default=None)
    payload=digest(snapshot.payload(selected))
    receipt=Receipt(plan.identity,snapshot.scope,snapshot.identity,selected,now,new_tail,payload)
    return Decision(receipt,len(positions),len(ranked),1,
                    'all_documents_scored' if covers_all else 'signed_margin_certificate',
                    payload==old.payload_sha256)


def exact_postings(snapshot: Snapshot, plan: Plan):
    if not plan.terms or not plan.k:return (),0
    positions=snapshot.hits(plan)
    values=dict(zip(positions,snapshot.scores(plan,positions)))
    # Nonmatching zero scores MUST stay: they outrank negative-score matches.
    ranked=_rank(snapshot,[(key,values.get(i,0.)) for i,key in enumerate(snapshot.ids)])
    return tuple(ranked[:plan.k]),len(positions)

class ScoreMap:
    """O(N) per-query state baseline with effective-IDF+avgdl invalidation."""
    def __init__(self,snapshot,plan):
        self.plan=plan;self.snapshot_hash=snapshot.identity;self.stats=snapshot.stats(plan)
        self.values=dict(zip(snapshot.ids,snapshot.scores(plan))) if plan.terms and plan.k else {}
        self.initial_scores=len(self.values)
    def update(self,snapshot,transition):
        now=snapshot.stats(self.plan)
        if not self.plan.terms or not self.plan.k:return (),0
        if (not transition.complete or transition.before!=self.snapshot_hash or
            transition.after!=snapshot.identity or now!=self.stats):
            self.values=dict(zip(snapshot.ids,snapshot.scores(self.plan)));scored=len(snapshot.ids)
        else:
            positions=sorted(snapshot.position[k] for k in transition.changed_ids if k in snapshot.position)
            for key in transition.changed_ids:self.values.pop(key,None)
            self.values.update((snapshot.ids[i],v) for i,v in zip(positions,snapshot.scores(self.plan,positions)))
            scored=len(positions)
        self.stats=now;self.snapshot_hash=snapshot.identity
        return tuple(_rank(snapshot,list(self.values.items()))[:self.plan.k]),scored
    @property
    def metadata_bytes(self):return len(encode({'plan':asdict(self.plan),'snapshot':self.snapshot_hash,
                                               'stats':asdict(self.stats),'scores':self.values}))

@dataclass(frozen=True)
class Block:
    positions: tuple[int,...]
    max_tf: dict[str,int]
    min_tf: dict[str,int]
    min_length: int
    max_length: int

class BlockMax:
    """Educational signed block-max baseline; not production WAND.

    One shared block index per corpus snapshot. Negative term contributions use
    minimum term frequency and maximum document length; absent terms imply zero.
    Block construction/state are charged separately from query scores.
    """
    def __init__(self,snapshot:Snapshot,size=32):
        if type(size) is not int or size<=0:raise ValueError('Positive block size required')
        self.snapshot=snapshot;self.blocks=[]
        for start in range(0,len(snapshot.ids),size):
            positions=tuple(range(start,min(start+size,len(snapshot.ids))))
            maximum={};minimum={};occurrence={}
            for i in positions:
                for q,f in snapshot._scorer.doc_freqs[i].items():
                    maximum[q]=max(f,maximum.get(q,0));minimum[q]=min(f,minimum.get(q,f))
                    occurrence[q]=occurrence.get(q,0)+1
            minimum={q:f for q,f in minimum.items() if occurrence[q]==len(positions)}
            lengths=[snapshot._scorer.doc_len[i] for i in positions]
            self.blocks.append(Block(positions,maximum,minimum,min(lengths),max(lengths)))
    @property
    def metadata_bytes(self):return len(encode([asdict(b) for b in self.blocks]))
    def query(self,plan:Plan):
        if not plan.terms or not plan.k:return (),0,0
        s=self.snapshot;bm=s._scorer;pending=[]
        for block in self.blocks:
            terms=[]
            for q in plan.terms:
                idf=bm.idf.get(q,0.)
                f=block.max_tf.get(q,0) if idf>=0 else block.min_tf.get(q,0)
                length=block.min_length if idf>=0 else block.max_length
                terms.append(idf*(f*2.5/(f+1.5*(.25+.75*length/bm.avgdl))))
            bound=math.fsum(terms)
            scale=math.fsum(abs(x) for x in terms)
            bound=math.nextafter(bound+FP_SLACK*max(1.,scale),math.inf)
            pending.append((bound,block))
        pending.sort(key=lambda x:(-x[0],x[1].positions[0]))
        best=[];scored=0;skipped=0
        for bound,block in pending:
            if len(best)>=min(plan.k,len(s.ids)) and best[-1][1]>bound:
                skipped+=1;continue
            values=s.scores(plan,block.positions);scored+=len(block.positions)
            best=_rank(s,best+[(s.ids[i],v) for i,v in zip(block.positions,values)])[:plan.k]
        return tuple(best),scored,skipped
