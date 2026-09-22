"""Canonical source records and the manuscript's original EM/F1 normalization."""
from __future__ import annotations
import collections
import hashlib
import json
import re
import string
from pathlib import Path

def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def digest(x):
    return hashlib.sha256(x.encode() if isinstance(x, str) else x).hexdigest()

def tokens(s):
    return re.findall('\\b\\w\\w+\\b', s.lower())

def frame(d):
    return canonical({k: d[k] for k in ('id', 'title', 'text', 'sha256')}) + '\n'

def normalize(s):
    return ' '.join(re.sub('\\b(a|an|the)\\b', ' ', ''.join((c for c in s.lower() if c not in string.punctuation))).split())

def score_answer(raw, answers):
    a = normalize(raw).split()
    em = f1 = 0.0
    for ans in answers:
        b = normalize(ans).split()
        em = max(em, float(a == b))
        hit = sum((collections.Counter(a) & collections.Counter(b)).values())
        f1 = max(f1, 2 * hit / (len(a) + len(b)) if a and b else float(a == b))
    return {'em': em, 'f1': f1}


def load_json(path: str | Path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_job(job: dict) -> None:
    """Reject malformed source inputs; labels are deliberately kept in a separate file."""
    if not isinstance(job, dict) or not isinstance(job.get("id"), str) or not isinstance(job.get("question"), str):
        raise ValueError("Each job needs string id and question fields")
    forbidden = {"answers", "answer", "support_indices", "supporting_facts", "gold"}
    if forbidden.intersection(job):
        raise ValueError("Keep evaluation labels outside the reader job")
    if not isinstance(job.get("docs"), list) or not job["docs"]:
        raise ValueError("A nonempty docs list is required")
    seen = set()
    for doc in job["docs"]:
        if not all(isinstance(doc.get(k), str) for k in ("id", "title", "text", "sha256")):
            raise ValueError("Each source record requires id, title, text and sha256 strings")
        if doc["id"] in seen:
            raise ValueError("Source IDs must be unique within a job")
        seen.add(doc["id"])
        if digest(doc["text"]) != doc["sha256"]:
            raise ValueError("Source-body digest mismatch")
        if type(doc.get("tokens")) is not int or doc["tokens"] <= 0:
            raise ValueError("Record token counts must be positive integers")
        if forbidden.intersection(doc):
            raise ValueError("Source records must not carry answer/support labels")
