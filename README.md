# Semantic Footprints

Implementation and evaluation code for **Semantic Footprints for LLM Context:
A Controlled Study of Coverage, Selective Retrieval, and Validation**, by
Joohyun Lee and Wen-Syan Li.

The experiments compare evidence selection under token budgets, answer generation
with selective retrieval, and context reuse after source changes.

## Included methods

- BM25 rank, graph-aware singleton, linked pair/triple bundles, lexical expansion,
  maximal marginal relevance, and an all-record control.
- A shared one-call reader, `UNKNOWN`-triggered retrieval and a final union reader.
- Answer scoring, source/provider token accounting, and paired bootstrap analysis.
- Footprint, dependency-validation and policy-revision research modules.

[Paper-to-code map](docs/PAPER_MAP.md) describes which experiments each module covers.

## Quick start

Python 3.10 or newer is required. These examples run locally without a model or API key.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/check_public_tree.py
wccu-eval select --jobs examples/synthetic_jobs.json \
  --policy bundle --budget 80 --output outputs/synthetic_selection.json
python scripts/revision_demo.py --output outputs/revision_cases.json
```

The selection example uses synthetic text and illustrative token costs. The
revision example generates the study's CPU scenarios without reader calls.
Output paths must be new; the commands do not overwrite existing results.

## Run the experiments

### HotpotQA budget transfer

With local HotpotQA distractor-development data and the specified Qwen tokenizer,
run the selection-only comparison on 512 components:

```bash
python -m pip install -e '.[tokenizer]'
python scripts/reproduce_hotpot_budget_transfer.py \
  --data private-data/hotpot_dev_distractor.json \
  --tokenizer-json private-data/qwen-tokenizer/tokenizer.json \
  --output outputs/hotpot-budget
```

This checks file hashes, selects the fixed sample and evaluates rank, singleton
and bundle at 512 and 2,048 source tokens. It reports complete support and the
paired budget interaction. See [HotpotQA instructions](docs/HOTPOT_BUDGET_TRANSFER.md)
for input sources, hashes and statistical units. No reader inference is involved.

### Reader and selective-retrieval comparisons

Prepare a sample from local benchmark data. The official 2Wiki answer-alias file
is required for the paper's scoring.

```bash
python scripts/prepare_candidates.py --dataset 2wiki \
  --input private-data/2wiki/train.json \
  --aliases private-data/2wiki/id_aliases.json \
  --manifest configs/splits/selective_rescue.json \
  --tokenizer private-data/qwen-tokenizer/tokenizer.json \
  --output private-data/prepared-selective
```

Use `configs/splits/static_transfer.json` for the static sample. For the MuSiQue
pilot, use `--dataset musique`, the original JSONL file and
`configs/splits/musique_pilot.json`, without `--aliases`. Sample IDs, source hashes,
order and token costs are checked during conversion.

Provide the study's two IRCoT demonstration texts as a plain UTF-8 file. With a
local request/response cache, the following command runs without network access:

```bash
wccu-eval run --jobs private-data/prepared-selective/jobs.json \
  --labels private-data/prepared-selective/labels.json \
  --demos private-data/demos.txt --cache private-cache/reader \
  --model gpt-4.1-2025-04-14 --seed 9153101 \
  --policies rank singleton bundle full --caps 512 1024 2048 \
  --output outputs/reader_rows.json
wccu-eval summarize --rows outputs/reader_rows.json \
  --output outputs/summary.json
```

A missing cache entry raises an error. New API calls require `--execute`, the
complete `--endpoint` URL and a positive `--max-new-calls`. Set `WCCU_API_KEY` in
the environment when the endpoint requires authentication. The limit counts new
requests, not spending, and requests are not retried automatically. Transport
tests use mocked responses; live endpoints should be checked before collection.

`full` sends every candidate and ignores the initial source cap. Selective
fallback starts from the question with its own 1,024-source-token cap; its final
reader receives the exposed union. Source-token caps and provider input/output
tokens are reported separately.

See [formats and analysis commands](docs/FORMATS.md) for cache imports and paired
comparisons, and [reproduction scope](docs/REPRODUCIBILITY.md) for required inputs.

## Data and scope

The repository includes sample IDs, source hashes, configurations and synthetic
examples. Obtain benchmark texts, answer files, tokenizer/model assets and the
IRCoT examples from their original sources under their respective terms. Collected
model responses are not included; exact replay requires those responses, whereas
new inference may return different answers.

The public code covers the listed selection/reader methods and bounded validation
modules. It does not include the complete native banking/airline adapters or
public-trajectory annotation workflow. The validation modules assume supplied
dependencies and local state; they are not a production transaction service.

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite the accompanying research manuscript.

No project-wide license has been selected for the author-developed code; the
current status is recorded in [LICENSE](LICENSE). Third-party components retain
the licenses listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
