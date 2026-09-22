# Reproduction scope

## Local tests

`python -m pytest -q` runs synthetic and fixture-independent tests. The selection
and revision examples also run locally. These checks exercise the implementation
without benchmark data, model weights or API credentials.

## Selection-only reproduction

The [HotpotQA budget-transfer script](HOTPOT_BUDGET_TRANSFER.md) reconstructs the
512-component sample from the local benchmark file, tokenizer and supplied
exclusion hashes. It recomputes selections, support, costs and paired intervals.
It is independent of reader-response caches.

## Reader evaluation

Obtain the source dataset, answer aliases, pinned tokenizer and the exact two
IRCoT demonstrations. The sample manifests reconstruct the selected static,
selective and pilot samples; they do not rebuild every earlier exclusion decision.
Candidate body hashes, order and serialized token counts must match the manifests.

The source tokenizer is `Qwen/Qwen3-4B-Instruct-2507`, revision
`cdbee75f17c01a7cc42f958dc650907174af0554`. Token costs cover complete serialized
records, not passage bodies alone. Synthetic demonstration text is only for tests
and must not replace the study's demonstrations in reported comparisons.

A saved request/response cache reproduces the reader outputs that it contains.
The original response archive is not distributed here. New inference uses the
same methods but can return different answers, even with a fixed model identifier
and seed. The HTTP interface is covered by offline tests, not a live-provider
compatibility guarantee.

## Scoring and statistical units

The parser retains malformed responses as raw text for normalized scoring.
Only a returned answer that normalizes to `UNKNOWN` triggers fallback. Fallback
starts from the question and does not inherit first-stage reasoning. Cached
requests still count toward each policy's standalone logical cost.

Pair outcomes by question/component ID and resample within the specified strata.
Repeated layouts, revision branches and shared requests do not add independent
units. Input ratios are ratios of means. Date-format sensitivity remains separate
from official EM/F1; descriptive categories are not human-adjudicated labels.

## Implementation coverage

The public package provides selection/reader evaluation and the footprint/revision
modules listed in [PAPER_MAP.md](PAPER_MAP.md). The full neural ranking pipeline,
public-trajectory extraction, native banking/airline adapters and every original
launcher are not included. Source corpora, model/tokenizer files and saved model
responses must be obtained or maintained separately.

The code contains no remote transaction, crash-recovery or automatic dependency
extraction guarantee. Local validation assumes the dependencies and rules supplied
by each experiment's adapter.
