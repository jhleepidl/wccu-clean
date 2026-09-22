# Input and output formats

## Jobs and evaluation labels

A jobs file is a JSON array. Each job requires string `id` and `question`, optional
`stratum`, and nonempty `docs`. Each document requires `id`, `title`, `text`,
SHA-256 of the UTF-8 body as `sha256`, and a positive integer `tokens`. A token cost
covers the entire canonical JSON record (`id/title/text/sha256`) plus its newline.
Do not tokenize the body alone. See `wccu_eval.common.frame`.

Gold answers/supports belong in a separate dictionary keyed by job ID:

```json
{"synthetic-001": {"answers": ["Cedar Bay"], "support_indices": [0, 1]}}
```

Support indices refer to candidate **positions**, not logical IDs. Dataset labels
are only used after reading to score EM/F1 and complete support. The fabricated
files in `examples/` demonstrate the schema without distributing benchmark text.

## Cache records

Each file is `<sha256(canonical(request))>.json`, containing `request` and raw
provider `response`. The entire payload matters, including prompts, model, seed,
output cap and optional fields. The collector preserves the raw response before
parsing. Keys are read from `WCCU_API_KEY`, never from a committed file.

## Paired statistics

Create dictionaries such as:

```json
{
  "question-1": {"stratum": "chain", "em": 1.0, "prompt_tokens": 2000},
  "question-2": {"stratum": "comparison", "em": 0.0, "prompt_tokens": 1000}
}
```

Both methods must contain exactly the same IDs and strata. Run:

```bash
wccu-eval compare --a outputs/method_a.json --b outputs/method_b.json \
  --metric em --replicates 20000 --seed 20260921 \
  --output outputs/em_difference.json
wccu-eval compare --a outputs/method_a.json --b outputs/method_b.json \
  --metric prompt_tokens --ratio --output outputs/input_ratio.json
```

Use actual independent units. Repeated layouts, policy arms, shared requests and
revision branches are not additional independent questions. Percentages in the
paper are fractions multiplied by 100; stored EM/F1 values remain in [0,1].
