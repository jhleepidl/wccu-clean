# Third-party notices

## IRCoT helpers

`src/wccu_eval/upstream_helpers.py` contains the three helper functions extracted
in the study from `StonyBrookNLP/ircot`, commit
`3c1820f698eea5eeddb4fba3c56b64c961e063e4`.

Upstream: https://github.com/StonyBrookNLP/ircot
License: Apache-2.0; the supplied license text is `third_party/IRCoT_LICENSE`.
The local change makes the regex literal raw to remove a Python escape warning;
its regex value and matching behavior are unchanged. Only these helpers are included; obtain published worked examples separately.

## Reviewed BM25 helper

`src/native_signed_receipt/rank_bm25_reviewed.py` is the same reviewed transcription
used by the original query-validity study, derived from `dorianbrown/rank_bm25`
0.2.2 under Apache-2.0. It retains BM25/BM25Okapi and omits unrelated variants.

Upstream: https://github.com/dorianbrown/rank_bm25
License text: `third_party/RANK_BM25_LICENSE`.
The query-validity kernel imports this modified helper to retain the study's computation.
The lexical bundle selector uses its own separate BM25 formulation and parameters.

## Dependencies and datasets

NumPy, SciPy, scikit-learn, optional tokenizer/inference libraries, and all dataset
sources retain their own licenses. Their distributions, model weights and data
are not copied into this repository. Evaluation code and sample identifiers do not change those redistribution terms.
