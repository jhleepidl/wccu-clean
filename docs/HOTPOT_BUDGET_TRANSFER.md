# HotpotQA budget transfer

This experiment measures complete supporting-paragraph coverage for rank,
graph-aware singleton and bundle selectors on 512 HotpotQA components. Source
budgets are 512 and 2,048 tokens. The main contrast is the paired change in the
bundle-minus-singleton effect between those budgets. Reader answers are not generated.

## Inputs

Obtain HotpotQA distractor-development data and the Qwen3-4B-Instruct-2507 tokenizer
JSON at revision `cdbee75f17c01a7cc42f958dc650907174af0554`:

- HotpotQA: https://github.com/hotpotqa/hotpot
- Tokenizer: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507

These external files follow their maintainers' terms. Expected hashes are in
`configs/hotpot_budget_transfer/SAMPLING_AUDIT.json`.

## Run

```bash
python -m pip install -e '.[test,tokenizer]'
python scripts/reproduce_hotpot_budget_transfer.py \
  --data private-data/hotpot_dev_distractor.json \
  --tokenizer-json private-data/qwen-tokenizer/tokenizer.json \
  --output outputs/hotpot-budget
```

The script checks hashes, reconstructs the sample, serializes records with numeric
candidate IDs and evaluates 3,072 method/budget conditions. It then runs the paired
bootstrap. No network or model calls are made. Local outputs may contain source
text and labels; keep the output directory outside version control.

The Python encoder supports the specified Qwen NFC/regex/byte-BPE format. Its
counts were checked against 4,000 stored complete-record costs; other tokenizer
formats require separate validation.

## Interpretation

Supporting sources are disjoint from the recorded prior evaluation candidates
and demonstrations after the supplied exclusions. HotpotQA remains a previously
used public benchmark; this separation does not rule out pretraining exposure.
The 426 bridge and 86 comparison components retain their observed proportions
in bootstrap. Budget-infeasible cases remain in the primary population.

See [the protocol](../configs/hotpot_budget_transfer/PROTOCOL.md) for sampling,
fixed parameters and contrasts. Internal hash prefixes are part of sample identity
and must be retained to reconstruct the same sample.
