"""Convert locally supplied benchmark records using an explicit hash-only sample index.

This module does not fetch data, accept source licenses, or select a new test set.
"""
from __future__ import annotations
import json
from pathlib import Path
from .common import digest, frame, validate_job


def read_records(path: str | Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    records = json.loads(text) if text.lstrip().startswith("[") else [json.loads(x) for x in text.splitlines() if x.strip()]
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise ValueError("Expected a JSON array or JSONL records")
    return records


def prepare_candidates(records: list[dict], manifest: list[dict], encode, *,
                       dataset: str, aliases: list[dict] | None = None,
                       split: str = "test") -> tuple[list[dict], dict]:
    """The encoder returns token IDs without adding model special tokens.

    Candidate order and full serialized-record costs are checked against the index.
    Gold labels are written separately and never become candidate fields.
    """
    if dataset not in ("2wiki", "musique"):
        raise ValueError("dataset must be 2wiki or musique")
    key = "_id" if dataset == "2wiki" else "id"
    indexed = {}
    for row in records:
        q = row[key]
        if q in indexed:
            raise ValueError("Duplicate benchmark ID")
        indexed[q] = row
    if dataset == "2wiki" and aliases is None:
        raise ValueError("Supply the official 2Wiki id_aliases file; missing aliases silently change EM")
    amap = {}
    for r in aliases or []:
        av = r["aliases"]
        if isinstance(av, dict):
            av = av.get("aliases", [])
        amap[r["Q_id"]] = list(av) + list(r.get("demonyms", []))
    selected = [r for r in manifest if r["split"] == split]
    if not selected or len({r['id'] for r in selected}) != len(selected):
        raise ValueError("Manifest must contain a nonempty set of unique selected IDs")
    jobs, labels = [], {}
    for entry in selected:
        row = indexed[entry["id"]]
        if dataset == "2wiki":
            paragraphs = [(str(i), title, " ".join(sentences)) for i, (title, sentences) in enumerate(row["context"])]
            titles = {title for title, _ in row["supporting_facts"]}
            support = [i for i, (_, title, _) in enumerate(paragraphs) if title in titles]
            if len({title for _, title, _ in paragraphs}) != len(paragraphs):
                raise ValueError("Duplicate candidate titles are outside the 2Wiki study population")
            if not titles <= {title for _, title, _ in paragraphs}:
                raise ValueError("Support title absent from supplied candidates")
            answers = list(dict.fromkeys([row["answer"]] + amap.get(row.get("answer_id", ""), [])))
        else:
            pp = row["paragraphs"]
            paragraphs = [(str(p["idx"]), p["title"], p["paragraph_text"]) for p in pp]
            support = [i for i, p in enumerate(pp) if p["is_supporting"]]
            answers = [row["answer"]] + row.get("answer_aliases", [])
        docs = []
        for identity, title, body in paragraphs:
            d = {"id": identity, "title": title, "text": body, "sha256": digest(body)}
            d["tokens"] = len(encode(frame(d)))
            docs.append(d)
        observed = [{k: d[k] for k in ("id", "sha256", "tokens")} for d in docs]
        if observed != entry["documents"]:
            raise ValueError(f"Candidate body/order/tokenization does not match manifest for {entry['id']}")
        job = {"id": entry["id"], "question": row["question"], "stratum": entry["stratum"], "split": split, "docs": docs}
        validate_job(job)
        jobs.append(job)
        labels[entry["id"]] = {"answers": answers, "support_indices": support}
    return jobs, labels
