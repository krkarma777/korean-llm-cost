# Data

Source corpora are **downloaded by scripts**, not committed.

## Planned sources (Phase 1)

| Category | Dataset | License | Loader |
|---|---|---|---|
| Conversational | AI Hub 일상대화 corpus | AI Hub academic | TBD |
| News | KLUE-YNAT, Korean Wikipedia | Public | TBD |
| Medical | KorMedMCQA, AI Hub medical text | Public / AI Hub | TBD |
| Parallel ko↔en | KorNLI ↔ MNLI overlap, OpenSubtitles ko-en | Public | TBD |

## Important

- Do not re-host downloaded corpora here. `data/corpora/*` and `data/parallel/*` are git-ignored.
- Do not generate the English parallel set via GPT translation — reviewers reject results with translation-induced bias. Use existing parallel datasets only, or multi-model translation with averaging.
