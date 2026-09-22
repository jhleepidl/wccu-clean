"""Original authored policy-revision workload and strict decision parser.

Expected decisions score completed proposals. They are never reader inputs.
This module models evidence selection/re-evaluation, not remote transactions.
"""
from __future__ import annotations
import copy,random,time,json
from .common import canonical,digest
from native_signed_receipt.engine import Snapshot,Plan,Transition,issue,advance,exact_postings
KINDS = ('invoice','allocation','refund','routing')
CHANGES = ('unchanged','irrelevant_edit','exposed_edit','unexposed_promotion','insertion','query_config','hidden_policy')
def render_policy(scope, kind, revision, params, repetitions, padding=0):
    record = {'type': 'policy', 'scope': scope, 'kind': kind, 'revision': revision, **params}
    return (' '.join([scope] * repetitions) + '\n' + canonical(record) + '\n' + ' '.join(['archive'] * padding)).strip()

def oracle(case, documents):
    policies = [d['policy'] for d in documents if d.get('policy', {}).get('scope') == case['request']['scope'] and d['policy']['kind'] == case['kind']]
    revision = max((p['revision'] for p in policies))
    latest = [p for p in policies if p['revision'] == revision]
    assert len(latest) == 1
    p = latest[0]
    q = case['request']
    k = case['kind']
    if k == 'invoice':
        value = q['quantity'] * p['unit_cents'] + p['fee_cents']
        decision = 'commit'
    elif k == 'allocation':
        value = q['used'] + q['requested']
        decision = 'commit' if value <= p['limit'] else 'deny'
        value = value if decision == 'commit' else 0
    elif k == 'refund':
        decision = 'commit' if q['age_days'] <= p['window_days'] else 'deny'
        value = min(q['paid_cents'], p['cap_cents']) if decision == 'commit' else 0
    else:
        value = p['urgent_queue'] if q['severity'] >= p['threshold'] else p['normal_queue']
        decision = 'commit'
    return {'decision': decision, 'value': value}

def build_case(kind, index, split):
    rng = random.Random(int(digest(f'{kind}:{index}:{split}')[:12], 16))
    scope = 'scope' + digest(f'r26:{kind}:{index}:{split}')[:12]
    if kind == 'invoice':
        q = {'quantity': rng.randint(2, 9)}
        params = {'unit_cents': rng.randint(110, 790), 'fee_cents': rng.randint(10, 180)}
        new = {**params, 'unit_cents': params['unit_cents'] + rng.randint(30, 200)}
    elif kind == 'allocation':
        q = {'used': rng.randint(20, 50), 'requested': rng.randint(5, 18)}
        params = {'limit': q['used'] + q['requested'] + 3}
        new = {'limit': q['used'] + q['requested'] - 1}
    elif kind == 'refund':
        q = {'age_days': rng.randint(8, 20), 'paid_cents': rng.randint(500, 2400)}
        params = {'window_days': q['age_days'] + 2, 'cap_cents': q['paid_cents'] - 50}
        new = {**params, 'window_days': q['age_days'] - 1}
    else:
        q = {'severity': rng.randint(4, 9)}
        params = {'threshold': q['severity'] + 1, 'normal_queue': rng.randint(10, 40), 'urgent_queue': rng.randint(60, 90)}
        new = {**params, 'threshold': q['severity']}
    q.update(scope=scope, kind=kind)

    def policy(identity, revision, values, repeat, padding=0):
        return {'id': identity, 'text': render_policy(scope, kind, revision, values, repeat, padding), 'policy': {'scope': scope, 'kind': kind, 'revision': revision, **values}}
    docs = [policy('policy-0', 1, params, 12)]
    for i in range(3):
        docs.append({'id': f'reference-{i}', 'text': ' '.join([scope] * (9 - i)) + '\n' + canonical({'type': 'reference', 'note': 'Administrative reference without decision rules.'})})
    docs.extend(({'id': f'noise-{i}', 'text': f'Unrelated archive catalog {i} ' + 'historical records ' * 12} for i in range(60)))
    case = {'id': scope, 'kind': kind, 'split': split, 'request': q, 'documents': docs, 'revisions': {}}
    for change in CHANGES:
        ds = copy.deepcopy(docs)
        k = 3
        if change == 'irrelevant_edit':
            ds[-1]['text'] += ' revised inventory memo'
        elif change == 'exposed_edit':
            ds[0] = policy('policy-0', 2, new, 12)
        elif change == 'unexposed_promotion':
            ds[-1] = policy(ds[-1]['id'], 2, new, 36)
        elif change == 'insertion':
            ds.append(policy('policy-new', 2, new, 36))
        elif change == 'hidden_policy':
            ds.append(policy('policy-hidden', 2, new, 1, 100))
        elif change == 'query_config':
            k = 4
        case['revisions'][change] = {'documents': ds, 'k': k}
    return case

def visible(snapshot, selected):
    return [{'id': key, 'text': snapshot.texts[snapshot.position[key]]} for key, _ in selected]

def independent_rank(snapshot, plan):
    assert len(plan.terms) == 1
    scorer = snapshot._scorer
    word = plan.terms[0]
    scores = []
    for i, tf in enumerate(scorer.doc_freqs):
        f = tf.get(word, 0)
        denom = f + 1.5 * (1 - 0.75 + 0.75 * scorer.doc_len[i] / scorer.avgdl)
        scores.append((snapshot.ids[i], scorer.idf.get(word, 0.0) * f * 2.5 / denom))
    return tuple(sorted(scores, key=lambda x: (-x[1], snapshot.position[x[0]]))[:plan.k])

def cpu(cases):
    rows = []
    init = []
    timings = []
    for case in cases:
        t = time.perf_counter()
        old = Snapshot(case['documents'], scope=case['id'])
        init_time = time.perf_counter() - t
        plan = Plan(case['id'], 3)
        initial = issue(old, plan)
        old_visible = visible(old, initial.receipt.selected)
        assert 'policy-0' in {r['id'] for r in old_visible}
        init.append({'id': case['id'], 'kind': case['kind'], 'split': case['split'], 'visible': old_visible, 'initial_oracle': oracle(case, case['documents']), 'index_seconds': init_time, 'index_bytes': old.index_serialized_bytes, 'initial_scored_docs': initial.doc_scores})
        for change in CHANGES:
            cfg = case['revisions'][change]
            t = time.perf_counter()
            current = Snapshot(cfg['documents'], scope=case['id'])
            index_time = time.perf_counter() - t
            t = time.perf_counter()
            transition = Transition.from_snapshots(old, current)
            transition_time = time.perf_counter() - t
            nowplan = Plan(case['id'], cfg['k'])
            choices = ['query_exact', 'query_postings', 'query_receipt']
            offset = int(digest(case['id'] + change)[:2], 16) % 3
            choices = choices[offset:] + choices[:offset]
            results = {}
            for arm in choices:
                t = time.perf_counter()
                if arm == 'query_exact':
                    d = issue(current, nowplan)
                    selected = d.receipt.selected
                    work = d.doc_scores
                    reason = 'exact'
                elif arm == 'query_postings':
                    selected, work = exact_postings(current, nowplan)
                    reason = 'ordinary_exact_postings'
                else:
                    d = advance(current, nowplan, initial.receipt, transition)
                    selected = d.receipt.selected
                    work = d.doc_scores
                    reason = d.reason
                results[arm] = {'visible': visible(current, selected), 'selected': selected, 'seconds': time.perf_counter() - t, 'scored_docs': work, 'reason': reason}
            gold_rank = independent_rank(current, nowplan)
            for arm, result in results.items():
                assert [k for k, v in gold_rank] == [k for k, v in result['selected']], (case['id'], change, arm)
                assert result['visible'] == results['query_exact']['visible']
            current_visible = results['query_exact']['visible']
            changed = current_visible != old_visible
            content_stale = any((r['id'] not in current.position or r['text'] != current.texts[current.position[r['id']]] for r in old_visible))
            expected = oracle(case, cfg['documents'])
            oracle_changed = expected != oracle(case, case['documents'])
            if change in ['unexposed_promotion', 'insertion']:
                assert changed and (not content_stale) and oracle_changed
            if change == 'hidden_policy':
                assert not changed and (not content_stale) and oracle_changed
            if change == 'irrelevant_edit':
                assert not changed and (not oracle_changed)
            if change == 'query_config':
                assert changed and (not oracle_changed)
            rows.append({'id': case['id'], 'kind': case['kind'], 'split': case['split'], 'change': change, 'visible': current_visible, 'old_visible': old_visible, 'selection_changed': changed, 'exposed_content_stale': content_stale, 'oracle_changed': oracle_changed, 'expected': expected, 'refresh': {'content_only': content_stale, 'epoch': old.identity != current.identity or plan.identity != nowplan.identity, **{a: changed for a in ['query_exact', 'query_postings', 'query_receipt']}}, 'retrieval': {a: {k: v for k, v in x.items() if k not in ['selected', 'visible']} for a, x in results.items()}, 'index_seconds': index_time, 'transition_seconds': transition_time, 'index_bytes': current.index_serialized_bytes, 'receipt_bytes': initial.receipt.metadata_bytes, 'current_documents': len(current.ids)})
    return (init, rows)

def decode(raw):

    def pairs(xs):
        d = {}
        for k, v in xs:
            if k in d:
                raise ValueError('Duplicate key')
            d[k] = v
        return d
    try:
        d = json.loads(raw, object_pairs_hook=pairs)
    except (ValueError, TypeError):
        return None
    if not isinstance(d, dict) or set(d) != {'decision', 'value'} or d['decision'] not in ['commit', 'deny', 'need_context'] or (type(d['value']) is not int) or (d['value'] < 0):
        return None
    if d['decision'] != 'commit' and d['value'] != 0:
        return None
    return d
