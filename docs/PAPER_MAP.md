# Paper-to-code map

The map uses section titles and experiment names rather than manuscript revision
numbers. The methods share building blocks, but their evaluation units and outputs
differ.

| Experiment or method | Code and inputs | What is included |
|---|---|---|
| Fixed lexical bundle selection; Algorithm 1 | `wccu_eval.selection` | Rank, matched singleton, linked bundles, lexical expansion and MMR |
| Static selection transfer | `selection`, `reader`; `configs/splits/static_transfer.json` | Selectors, static prompt and sample metadata; reader outputs supplied separately |
| Selective rescue; Algorithm 3 | `reader`, `experiment`, `transport`; `configs/splits/selective_rescue.json` | First answer, fixed abstention rule, dynamic restart, union reader and logical token/call totals |
| Matched-selection and wider-context controls | `statistics`, `data`, the selective sample | Paired analysis of supplied evaluation rows; all-record and budgeted policies |
| Support-disjoint HotpotQA budget transfer | `scripts/reproduce_hotpot_budget_transfer.py`; `configs/hotpot_budget_transfer/` | Local-data sample reconstruction, all method/budget conditions and paired intervals |
| Data/reader transfer pilot | `reader`, `experiment`; `configs/splits/musique_pilot.json` | Fixed policy evaluation with the supplied examples and model configuration |
| Footprints and representation capacity | `mosaic_wccu.models`, `semantic_tiles`, `external_partition` | Range/identity and capacity-evaluation functions; external data required |
| Read-witness capture and byte-cap controls | `mosaic_wccu.phase5c`, `phase6b` | Physical-range capture and expansion controls from supplied ranking traces |
| Validation with supplied read sets and rules | `mosaic_wccu.conflict`, `phase6a` | Bounded synthetic validation mechanisms |
| Selection validity after policy revisions | `wccu_eval.revisions`, `native_signed_receipt.engine` | Four scenario types, seven changes, local ranking and proposal parsing |
| Descriptive date sensitivity | `wccu_eval.answer_audit` | Complete-date normalization, separate from official EM/F1 |

The neural ranking matrix, all original experiment launchers, full native effect
adapters and public-trajectory annotation workflow are outside this release.
The revision example generates cases and checks the local retrieval kernel; it
cannot reproduce model-completion tables without the original reader responses
and effect-evaluation workflow.

Internal module names and sampling hash prefixes are retained where they identify
recorded implementations. [SOURCE_MAP.json](SOURCE_MAP.json) lists their origins;
those identifiers are not publication or benchmark version claims.
