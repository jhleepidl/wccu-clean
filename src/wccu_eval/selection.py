"""Paper selectors. Bundles and graph-aware singletons share the same scoring terms.

Extracted from the measured implementation, with a typed, validated public entry point.
Canonical MMR deliberately uses lexical cosine and no cost normalization.
"""
from __future__ import annotations
import itertools
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from .common import tokens, validate_job

def scale(x):
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)

def bm25(query, texts):
    words = [tokens(t) for t in texts]
    q = set(tokens(query)) - ENGLISH_STOP_WORDS
    n = len(words)
    avg = sum(map(len, words)) / max(n, 1)
    out = np.zeros(n)
    for w in q:
        df = sum((w in x for x in words))
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for i, x in enumerate(words):
            tf = x.count(w)
            out[i] += idf * tf * 1.9 / (tf + 0.9 * (0.6 + 0.4 * len(x) / max(avg, 1))) if tf else 0
    return out

def mention(title, text):
    a = tokens(title)
    b = tokens(text)
    return len(a) > 0 and len(' '.join(a)) >= 4 and any((b[i:i + len(a)] == a for i in range(len(b) - len(a) + 1)))

def features(job, embeddings):
    docs = job['docs']
    texts = [d['title'] + '\n' + d['text'] for d in docs]
    n = len(docs)
    tf = TfidfVectorizer(stop_words='english').fit_transform(texts)
    sim = (tf @ tf.T).toarray()
    np.fill_diagonal(sim, 0)
    graph = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        if mention(docs[i]['title'], docs[j]['text']) or mention(docs[j]['title'], docs[i]['text']):
            graph[i, j] = graph[j, i] = 1
    qterms = set(tokens(job['question'])) - ENGLISH_STOP_WORDS
    covers = [set(tokens(t)) & qterms for t in texts]
    expansions = []
    vec = TfidfVectorizer(stop_words='english')
    arr = vec.fit_transform(texts).toarray()
    vocab = vec.get_feature_names_out()
    for row in arr:
        terms = [vocab[k] for k in sorted(range(len(row)), key=lambda k: (-row[k], vocab[k]))[:8] if row[k] > 0]
        expansions.append(scale(bm25(' '.join(terms), texts)))
    return {'texts': texts, 'bm25': bm25(job['question'], texts), 'sim': np.maximum(embeddings @ embeddings.T, 0), 'lexical_sim': sim, 'graph': graph, 'covers': covers, 'qterms': qterms, 'expansions': np.array(expansions)}

def select(job, f, rel, policy, budget):
    n = len(rel)
    selected = []
    steps = []
    spent = 0
    seen = set()
    used_terms = set()
    units = [(i,) for i in range(n)]
    if policy in ('bridge_bundle', 'similarity_bundle'):
        adj = f['graph'] if policy == 'bridge_bundle' else f['lexical_sim'] >= 0.2
        units += [p for p in itertools.combinations(range(n), 2) if adj[p[0], p[1]]]
        units += [p for p in itertools.combinations(range(n), 3) if sum((bool(adj[i, j]) for i, j in itertools.combinations(p, 2))) >= 2]
    while True:
        choices = []
        for u in units:
            new = tuple((i for i in u if i not in selected))
            if not new:
                continue
            hashes = [job['docs'][i]['sha256'] for i in new]
            if len(set(hashes)) != len(hashes) or set(hashes) & seen:
                continue
            cost = sum((job['docs'][i]['tokens'] for i in new))
            if spent + cost > budget:
                continue
            coverage = len(set().union(*(f['covers'][i] for i in new)) - used_terms) / max(1, len(f['qterms']))
            if policy == 'rank':
                value = rel[new[0]]
            elif policy == 'iterative':
                i = new[0]
                exp = max((f['expansions'][j, i] for j in selected), default=0)
                value = 0.6 * rel[i] + 0.4 * exp
            elif policy == '_legacy_cost_normalized_mmr':
                i = new[0]
                red = max((f['sim'][i, j] for j in selected), default=0)
                value = (0.7 * rel[i] - 0.3 * red) / math.sqrt(cost)
            elif policy == 'bridge_atomic':
                i = new[0]
                link = max((f['graph'][i, j] for j in selected), default=0)
                value = (rel[i] + 0.35 * coverage + 0.3 * link) / cost ** 0.25
            else:
                adj = f['graph'] if policy == 'bridge_bundle' else f['lexical_sim'] >= 0.2
                links = sum((bool(adj[i, j]) for i, j in itertools.combinations(new, 2))) + sum((any((adj[i, j] for j in selected)) for i in new))
                rr = sorted((rel[i] for i in new), reverse=True)
                value = (rr[0] + 0.25 * sum(rr[1:]) / max(1, len(rr) - 1) + 0.35 * coverage + 0.3 * min(1, links / max(1, len(new) - 1))) / cost ** 0.25
            choices.append((float(value), -cost, tuple((-i for i in new)), new, u))
        if not choices:
            break
        _, _, _, new, u = max(choices)
        cost = sum((job['docs'][i]['tokens'] for i in new))
        selected.extend(new)
        seen.update((job['docs'][i]['sha256'] for i in new))
        spent += cost
        used_terms.update(set().union(*(f['covers'][i] for i in new)))
        steps.append({'inventory_unit': list(u), 'emitted': list(new), 'tokens': cost})
    return {'selected_indices': sorted(selected), 'spent_tokens': spent, 'steps': steps}

def get_features(job, query=None):
    j = job if query is None else dict(job, question=query)
    return features(j, np.zeros((len(job['docs']), 1)))

def canonical_mmr(job, f, budget):
    rel = scale(f['bm25'])
    chosen = []
    seen = set()
    spent = 0
    while True:
        options = []
        for i, d in enumerate(job['docs']):
            if i in chosen or d['sha256'] in seen or spent + d['tokens'] > budget:
                continue
            redundancy = max((f['lexical_sim'][i, j] for j in chosen), default=0.0)
            value = 0.7 * rel[i] - 0.3 * redundancy
            options.append((float(value), -i, i))
        if not options:
            break
        i = max(options)[2]
        chosen.append(i)
        seen.add(job['docs'][i]['sha256'])
        spent += job['docs'][i]['tokens']
    return {'selected_indices': sorted(chosen), 'spent_tokens': spent}


POLICIES = ("rank", "iterative", "mmr", "singleton", "bundle", "full")


def select_records(job: dict, policy: str = "bundle", budget: int = 1024,
                   query: str | None = None, pool: list[int] | None = None) -> dict:
    """Select whole records; `full` intentionally ignores the source-token cap.

    Ranking features are computed over the original candidate set even when an
    iterative retrieval pool restricts eligibility. Costs must have been measured
    with the chosen source tokenizer before calling this function.
    """
    import copy
    validate_job(job)
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    if type(budget) is not int or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    if pool is not None and any(type(i) is not int or i < 0 or i >= len(job["docs"]) for i in pool):
        raise ValueError("pool contains an invalid source index")
    if policy == "full":
        selected = sorted(set(range(len(job["docs"]))) if pool is None else set(pool))
        return {"selected_indices": selected, "spent_tokens": sum(job["docs"][i]["tokens"] for i in selected), "steps": []}
    f = get_features(job, query)
    masked = copy.deepcopy(job)
    if pool is not None:
        allowed = set(pool)
        for i, doc in enumerate(masked["docs"]):
            if i not in allowed:
                doc["tokens"] = budget + 1
    if policy == "mmr":
        return canonical_mmr(masked, f, budget)
    internal = {"singleton": "bridge_atomic", "bundle": "bridge_bundle"}.get(policy, policy)
    return select(masked, f, scale(f["bm25"]), internal, budget)
