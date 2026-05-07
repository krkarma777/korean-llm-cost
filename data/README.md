# Data

Source corpora are **downloaded by scripts or manually placed here**, never committed to git.

## Planned sources (Phase 1)

| Category | Dataset | License | Status |
|---|---|---|---|
| Conversational | AI Hub Topic-based Korean Daily Conversation | AI Hub academic-research license (no redistribution) | downloaded → `data/conversation/korean/`; loader TBD |
| News | KLUE-YNAT | CC-BY-SA 4.0 | wired in `corpus_loader.load_category("news")`; n=1000 measured |
| Medical | KorMedMCQA / AI Hub medical | Public / AI Hub | not yet downloaded |
| Parallel ko↔en | KorNLI ↔ MNLI overlap, OpenSubtitles ko-en | Public | not yet downloaded |

---

## AI Hub — Topic-based Korean Daily Conversation

- **Source**: AI Hub (https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=543)
- **License**: AI Hub academic-research license — redistribution prohibited
- **Year**: 2021
- **Size**: 256.73 MB (raw)
- **Status**: NOT included in this repo. See download instructions below.

### Download

1. Sign up at AI Hub (academic-email verification required).
2. On the dataset page, fill in the usage-purpose form and download.
3. Unpack the archive into `data/conversation/korean/`
   (current layout: `1.Training/`, `2.Validation/`).
4. Run `python data/scripts/prepare_konv.py`.

### Citation

[Insert the dataset's citation form here.]

---

## Important

- Do not re-host downloaded corpora in this repo. `data/corpora/*`, `data/parallel/*`, `data/conversation/`, and `data/medical/` are git-ignored.
- Do not generate the English parallel set via GPT translation — reviewers reject results biased by machine translation. Use existing parallel datasets, or multi-model translation with averaging if no alternative exists.
- AI Hub data redistribution is **prohibited by license**. Each user must download their own copy after AI Hub registration.
