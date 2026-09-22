# Validation

## Local release checks

The test environment uses Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0,
scikit-learn 1.8.0, pytest 9.0.2 and regex 2026.5.9. Dependency versions
are listed in `requirements-tested.txt`.

| Check | Result |
|---|---|
| Synthetic and fixture-independent tests | 60 passed, no failures or skips |
| Source-tree check | No findings in the checked credential, path and asset patterns |
| Wheel and console command | Package built and installed; synthetic selection executed |
| CPU revision example | 32 base cases and 224 revision conditions |
| Algorithm preservation | Selection, reader, parser, statistics and validation implementations unchanged |

Tests and examples above make no model calls. Installation was checked in a
separate virtual environment using the already-installed numerical dependencies.
The source-tree scanner checks specified patterns, not every possible secret or
third-party-rights issue.

## Data-dependent equivalence checks

The release retains the following previously recorded equivalence results. They
require benchmark inputs or saved responses that are not included in this source
repository and are not rerun by the public test suite.

| Comparison | Recorded result |
|---|---|
| Selectors against the measured implementation | 5,760 matched selections/costs across 384 questions, three caps and five selectors |
| Initial requests and answer parsing | 2,304 matching logical conditions |
| Dynamic path and union reader | 48 MuSiQue cases with matching ordered requests, exposed unions and final answers |
| Paired statistics | 90 comparisons of 20,000 resamples; maximum absolute difference 2.22e-16 |
| HotpotQA sample, selections and statistics | 3,072 method/budget rows and paired intervals reproduced, excluding runtime |
| Local-Qwen tokenizer costs | 4,000 stored complete-record counts matched |

[SOURCE_MAP.json](SOURCE_MAP.json) records implementation origins. Dataset
preparation and exact-response replay requirements are in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).
