from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .phase4f import TileRerankPolicy

PEERQA_HF_COMMIT = "e9e953c298a1784a3c5433eeed27521bb3ebb07a"
PEERQA_QA_SHA256 = "055880a01d82d3ace5d16e0992480903114b14fbbf0e9018a04262c231abbac8"
PEERQA_PAPERS_SHA256 = "92d90c1f70ba6f9ad4cace4a2a8faf6d4c3335d9a57a5f5ef6e61bd432270199"
PEERQA_QRELS_SHA256 = "36225dc2ceb1a74c673a662e1fc565cb6f02ad35afee0484e6121e91883634d6"
PEERQA_FILES = {
    "qa": ("qa/test-00000-of-00001.parquet", PEERQA_QA_SHA256),
    "papers": ("papers/test-00000-of-00001.parquet", PEERQA_PAPERS_SHA256),
    "qrels": ("qrels-paragraphs/test-00000-of-00001.parquet", PEERQA_QRELS_SHA256),
}

CONSUMED_HOTPOT_SHA256 = "e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8"
QWEN_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_EMBED_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QWEN_QUERY_INSTRUCTION = "Given a query, retrieve semantic context tiles that contain the evidence needed to answer it."
BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
BGE_RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
LOCAL_QWEN_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LOCAL_QWEN_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
TERRA_MODEL = "gpt-5.6-terra"
SOL_MODEL = "gpt-5.6-sol"

PRIMARY_RRF_K = 60
TRACE_DEPTH = 8
OUTPUT_K = 2
BOOTSTRAP_SEED = 20260904


@dataclass(frozen=True)
class Phase5APolicy:
    inventory_variant: str = "semantic_locality"
    rrf_k: int = PRIMARY_RRF_K
    trace_depth: int = TRACE_DEPTH
    output_k: int = OUTPUT_K
    local_llm_model: str = LOCAL_QWEN_MODEL
    local_llm_revision: str = LOCAL_QWEN_REVISION
    terra_model: str = TERRA_MODEL
    sol_model: str = SOL_MODEL
    reasoning_effort: str = "none"
    temperature: float = 0.0
    external_max_output_tokens: int = 128
    batch_completion_window: str = "24h"

    def tile_policy(self) -> TileRerankPolicy:
        return TileRerankPolicy(
            trace_depth=self.trace_depth,
            output_k=self.output_k,
            rrf_k=self.rrf_k,
            fusion_k=self.rrf_k,
        )


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def peerqa_expected_hashes() -> dict[str, str]:
    return {key: sha for key, (_, sha) in PEERQA_FILES.items()}


def verify_peerqa_files(qa: Path, papers: Path, qrels: Path) -> dict[str, Any]:
    supplied = {"qa": qa, "papers": papers, "qrels": qrels}
    got = {}
    for key, path in supplied.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_path(path)
        expected = peerqa_expected_hashes()[key]
        if digest != expected:
            raise RuntimeError(f"PeerQA {key} SHA mismatch expected={expected} got={digest}")
        got[key] = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
    return got


def reciprocal_rank_fusion_from_traces(
    qwen_order: Sequence[str],
    bge_order: Sequence[str],
    *,
    k: int = PRIMARY_RRF_K,
) -> tuple[tuple[str, float], ...]:
    ranks: dict[str, float] = {}
    for order in (qwen_order, bge_order):
        for rank, tile_id in enumerate(order, 1):
            ranks[str(tile_id)] = ranks.get(str(tile_id), 0.0) + 1.0 / (k + rank)
    return tuple(sorted(ranks.items(), key=lambda kv: (-kv[1], kv[0])))


def external_ranking_schema(labels: Sequence[str]) -> dict[str, Any]:
    labels = tuple(str(x) for x in labels)
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("labels must be unique and non-empty")
    props = {f"rank_{i}": {"type": "string", "enum": list(labels)} for i in range(1, len(labels) + 1)}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def render_external_prompt(case: Mapping[str, Any]) -> str:
    candidates = list(case.get("candidates") or [])
    blocks = []
    for c in candidates:
        blocks.append(f"[{c['label']}]\n" + "\n\n".join(str(x) for x in c["paragraphs"]))
    labels = " ".join(str(c["label"]) for c in candidates)
    return (
        "Query:\n" + str(case["query"]) + "\n\n"
        "Candidate evidence tiles (shown in retrieval order):\n" + "\n\n".join(blocks)
        + "\n\nRank ALL candidate tiles from most useful to least useful for answering the query. "
        "Do not answer the query and do not rewrite the evidence. "
        "The shown order is an upstream retrieval prior that may be useful. "
        f"The ranking must contain each of these labels exactly once: {labels}."
    )


def external_request_body(case: Mapping[str, Any], *, model: str) -> dict[str, Any]:
    policy = Phase5APolicy()
    if model not in {policy.terra_model, policy.sol_model}:
        raise ValueError(f"unexpected Phase5A external model: {model}")
    labels = tuple(str(c["label"]) for c in case["candidates"])
    return {
        "model": model,
        "reasoning": {"effort": policy.reasoning_effort},
        "temperature": policy.temperature,
        "store": False,
        "instructions": (
            "You rerank already-retrieved evidence tiles. Return only the structured ranking. "
            "Do not answer the query, do not construct new tiles, and do not use external tools."
        ),
        "input": render_external_prompt(case),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "phase5a_peerqa_tile_ranking",
                "strict": True,
                "schema": external_ranking_schema(labels),
            },
            "verbosity": "low",
        },
        "max_output_tokens": policy.external_max_output_tokens,
    }


def parse_external_output(body: Mapping[str, Any], labels: Sequence[str]) -> tuple[str, ...]:
    text = None
    for item in body.get("output") or []:
        if isinstance(item, Mapping) and item.get("type") == "message":
            for content in item.get("content") or []:
                if isinstance(content, Mapping) and content.get("type") == "output_text":
                    text = content.get("text")
                    break
    if not isinstance(text, str):
        raise ValueError("response has no output_text")
    payload = json.loads(text)
    labels = tuple(str(x) for x in labels)
    ranking = tuple(str(payload[f"rank_{i}"]) for i in range(1, len(labels) + 1))
    if len(ranking) != len(set(ranking)) or set(ranking) != set(labels):
        raise ValueError("external ranking is not a complete label permutation")
    return ranking


def fresh_claim_contract() -> dict[str, Any]:
    """Pre-specified claim hierarchy. Do not edit after Phase5A freeze."""
    return {
        "primary_method": "local_qwen3_4b_rrf_llm_fusion",
        "primary_baseline": "qwen_bge_rrf_k60",
        "primary_metric": "union_f1_at_2",
        "primary_success_rule": "cluster-bootstrap 95% CI for primary_method minus primary_baseline is strictly above zero",
        "cluster_unit": "PeerQA paper_id",
        "secondary_metrics": ["complete_coverage_at_2", "best_tile_f1_at_2", "ndcg_at_2"],
        "secondary_cells": [
            "qwen_embedding_provider_inventory",
            "bge_reranker_provider_inventory",
            "local_qwen3_4b_llm_only",
            "gpt_5_6_terra_llm_only",
            "gpt_5_6_terra_rrf_llm_fusion",
            "gpt_5_6_sol_llm_only",
            "gpt_5_6_sol_rrf_llm_fusion",
        ],
        "interpretation_rule": "secondary cells cannot rescue a failed primary fresh superiority claim",
        "no_retuning_after_outcome_inspection": True,
    }


def load_peerqa_fresh_docs(qa: Path, papers: Path, qrels: Path):
    from .benchmark_adapters import load_peerqa_parquet_with_audit
    files = verify_peerqa_files(qa, papers, qrels)
    docs, audit = load_peerqa_parquet_with_audit(qa, papers, qrels)
    if not docs:
        raise RuntimeError("PeerQA fresh eligible set is empty")
    return docs, audit, files


def fit_frozen_provider_lsa(hotpot_path: Path):
    """Rebuild the exact consumed-Hotpot LSA constructor used by Phase 4B.1/4C."""
    from .benchmark_adapters import load_hotpotqa_with_audit
    from .phase3i import LSAHashModel, deterministic_split, deterministic_subsample
    got = sha256_path(hotpot_path)
    if got != CONSUMED_HOTPOT_SHA256:
        raise RuntimeError(f"consumed Hotpot SHA mismatch expected={CONSUMED_HOTPOT_SHA256} got={got}")
    hotpot, audit = load_hotpotqa_with_audit(hotpot_path, invalid_support_policy="skip_record")
    train, _, split = deterministic_split(hotpot, train_fraction=.70)
    fit, _, inner = deterministic_split(train, train_fraction=.80)
    fit_docs = deterministic_subsample(fit, 1600)
    lsa = LSAHashModel().fit(fit_docs)
    return lsa, {"hotpot_sha256": got, "fit_docs": len(fit_docs), "split": split, "inner_split": inner, "audit": audit}
