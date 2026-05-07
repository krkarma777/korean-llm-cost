# korean-llm-cost

Tokenizer efficiency and cost comparison for Korean text across major LLMs (frontier, Korean-specialized, and open multilingual).

**Status**: Phase 1 scaffolding (private repo). Target venue: HCLT 2026.

## What this measures

- **TPC** (Tokens Per Character) — how many tokens each tokenizer assigns per Korean character
- **KPR** (Korean Penalty Ratio) — `TPC_korean / TPC_english_equivalent`, Korean's relative cost
- **ECPC** (Effective Cost per 1K chars) — TPC weighted by published per-token pricing

## Phased plan

| Phase | Scope | Gate |
|---|---|---|
| **1** | 6 models × 3 categories (conversation / news / medical) + ko↔en parallel | run on small sample, check whether any of H1/H2/H3 surfaces a surprising finding |
| **2** *(conditional)* | 12 models × 5 categories, add subword consistency metric | proceed only if Phase 1 finding is paper-worthy |

## Layout

```
src/korean_llm_cost/
  tokenizers/         # provider-specific Tokenizer wrappers (base.py = interface)
  metrics.py          # TPC / KPR / ECPC (later phase)
  pricing.py          # per-model $/token, dated (later phase)
  corpus_loader.py    # (later phase)

experiments/          # one numbered script per RQ (later phase)
notebooks/            # exploratory work; 01_sanity_check.ipynb is the first
data/                 # corpora are downloaded, never re-hosted (.gitignore)
results/              # raw/ ignored; figures/ tables/ committed
```

## Getting started

```bash
uv sync                  # installs tiktoken (Phase 1 minimum)
uv sync --extra dev      # adds jupyter for the sanity notebook
```

Open `notebooks/01_sanity_check.ipynb` and run all cells. Expected: TPC for Korean text in the **0.5–0.8** range, English in the **0.20–0.30** range — if you see numbers outside that, something is wrong (encoding, sample, model name).

## Reproducibility

Every tokenizer wrapper records `name + version + provider` (see `tokenizers/base.py::TokenizerInfo`). Pricing snapshots will be dated. Tokenizers can change with model updates — always check `info.version` in any reported result.

## License

MIT (see `LICENSE` — to be added).
