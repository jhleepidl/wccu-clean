# Protocol: selection-unit effect under two source budgets

## Question and fixed design
Does the extra benefit of linked pair/triple additions over an information-matched
singleton increase under a tight source budget on a different benchmark population?
This CPU-only experiment measures selection; reader responses are not collected.

- Data: the locally supplied HotpotQA distractor development JSON (7,405 source rows).
- Exclude malformed/missing supporting-fact references, duplicate titles or body hashes,
  empty sources, and records not containing ten candidates. No exclusion by selector
  performance, predicted answer, source cost, or number/strength of title links.
- Form components by shared supporting source title or normalized supporting body.
  Use one hash-selected representative per component. Exclude components with an
  archived evaluated query ID, or any support title/body matching a candidate from
  archived Hotpot evaluations and supplied prior lexical jobs/demonstrations.
- Sample at most 512 eligible components in ascending SHA256 order with the fixed
  prefix `wccu-r44-hotpot-v1|component|`; independently choose each representative
  using `wccu-r44-hotpot-v1|representative|`. Do not adjust the sample after scoring.
- Methods: fixed BM25 rank, graph-aware singleton, and title-linked bundle. Keep the
  original coefficients fixed. All methods use identical questions, candidates,
  source order and token costs; singleton and bundle additionally share the graph
  and scoring terms.
- Budgets: 512 and 2,048 Qwen3-4B source-record tokens, exactly as in the existing
  lexical study. Use the pinned tokenizer JSON; verify the local encoder against
  previously stored source-record costs before testing.
- Source record IDs are candidate-local numeric strings (`0` through `9`), exactly
  as in the static-transfer/selective-rescue record serializer; the question ID lives outside the
  serialized record.
- Full-context is only the unbudgeted support/cost reference. It has no reader
  outcome in this test and is not an equal-cap comparison.
- Outcome: complete coverage of all annotated supporting paragraphs; also support
  recall, actual source tokens, budget feasibility and method gain/loss counts.
- Primary contrast per component:
  I = (bundle_512 - singleton_512) - (bundle_2048 - singleton_2048).
  Pair all methods and budgets within the same component. Use 20,000 bootstrap
  draws, seed 2026092144, percentile two-sided 95% intervals, stratified by the
  representative's bridge/comparison type with the observed type weights fixed.
- A positive budget-dependence interpretation requires BOTH a positive lower
  bound for I and positive lower bound for the tight-budget bundle-singleton gain.
  Report the estimate and interval regardless of outcome. A failure is not equality.
  Secondary intervals are descriptive; no selection among them changes the primary.
- Keep infeasible-support cases in the primary. A feasible-only breakdown is a
  descriptive sensitivity, not a replacement population.

## Interpretation
The evaluation extends the fixed selector to supporting sources disjoint from
recorded prior evaluations within a public, previously used benchmark. It measures
selection on that sample, not reader accuracy or agent outcomes. It is not externally
preregistered, and pretraining exposure is unknown. Results include both question
types and budget-infeasible cases under the original coefficients and budgets.
