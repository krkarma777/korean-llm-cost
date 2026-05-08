"""P8 (per-media balance) + Gemini -15.3% origin analysis.

Reads ``results/raw/03_tpc_conversation_n1000_per_text.csv`` and produces
two complementary sub-analyses on the n=1000 conversation corpus.

P8 — per-media TPC balance
--------------------------
Pre-registered claim: for every (model, media) cell, aggregate TPC is
within ±5% of that model's overall conversation TPC. Refutation: any
cell deviates > ±5%.

For each (model, media) we report:
- aggregate TPC + 95% bootstrap CI
- deviation from the model-overall conversation TPC, in percent
- a per-cell verdict (within / outside the ±5% band)

Gemini -15.3% origin decomposition
----------------------------------
Gemini drops 15.3% from news (0.6961) to conversation (0.5894) — the
largest delta among all 7 models. We ask: is this drop concentrated in a
particular subject, tier, or media, or is it broadly distributed?

We attack the question two ways:

1. **Absolute Gemini conversation TPC by stratum** (per-subject and
   per-media), with the per-stratum delta vs Gemini's news TPC of 0.6961.
   Strata where Gemini drops most identify "Gemini-friendly conversation".

2. **Gemini's edge vs GPT-4o (per stratum)**: ratio
   ``TPC_gemini_stratum / TPC_gpt_stratum``. News-overall = 0.96×.
   Conversation-overall = 0.89×. Per-stratum ratios show where Gemini's
   tokenizer outperforms GPT *most* — that's the source of the -15.3%.
   Lower ratio = bigger Gemini-over-GPT gap = where the Gemini advantage
   is concentrated.

Outputs
-------
- ``results/raw/05_p8_per_media.csv``       per-(model, media) TPC + CI + delta
- ``results/raw/05_p8_summary.csv``         single-row P8 verdict
- ``results/raw/05_gemini_per_subject.csv`` Gemini absolute + ratio vs GPT per subject
- ``results/raw/05_gemini_per_tier.csv``    same, per pre-registered tier
- ``results/raw/05_gemini_per_media.csv``   same, per media
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from korean_llm_cost.subject_groups import SUBJECT_TO_TIER, TIER_ORDER  # noqa: E402

PER_TEXT_CSV = ROOT / "results" / "raw" / "03_tpc_conversation_n1000_per_text.csv"
OUT_DIR = ROOT / "results" / "raw"

# News n=1000 baselines (from 02_tpc_news_n1000_aggregate.csv, copied to
# 03_*.py earlier). Used for delta-vs-news computations.
NEWS_TPC = {
    "GPT-4o": 0.7226,
    "Claude Sonnet 4.5": 1.0831,
    "Gemini 2.5 Flash": 0.6961,
    "Solar 10.7B": 1.2055,
    "EXAONE 3.5 7.8B": 0.5468,
    "Qwen 2.5 7B": 0.8226,
    "Llama 3.1 8B": 0.7325,
}

MODEL_ORDER = (
    "EXAONE 3.5 7.8B",
    "Gemini 2.5 Flash",
    "GPT-4o",
    "Llama 3.1 8B",
    "Qwen 2.5 7B",
    "Claude Sonnet 4.5",
    "Solar 10.7B",
)


def aggregate_tpc(tok: np.ndarray, chr_: np.ndarray) -> float:
    return float(tok.sum() / chr_.sum())


def bootstrap_tpc_ci(
    tok: np.ndarray, chr_: np.ndarray,
    *, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(tok)
    if n == 0:
        return float("nan"), float("nan")
    ratios = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        ratios[b] = tok[idx].sum() / chr_[idx].sum()
    alpha = (1 - ci) / 2
    return float(np.quantile(ratios, alpha)), float(np.quantile(ratios, 1 - alpha))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 76)
    print("P8 (per-media balance) + Gemini -15.3% origin decomposition")
    print(f"  source: {PER_TEXT_CSV.relative_to(ROOT)}")
    print("=" * 76)

    # Read all rows once.
    with PER_TEXT_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Index per (model, *) -> (tokens, chars)
    by_model: dict[str, list[tuple[int, int]]] = defaultdict(list)
    by_model_media: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    by_model_subject: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    by_model_tier: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)

    for r in rows:
        m = r["model_label"]
        t = int(r["n_tokens"])
        c = int(r["n_chars"])
        media = r["media"]
        subj = r["subject"]
        tier = SUBJECT_TO_TIER.get(subj)
        by_model[m].append((t, c))
        by_model_media[(m, media)].append((t, c))
        by_model_subject[(m, subj)].append((t, c))
        if tier is not None:
            by_model_tier[(m, tier)].append((t, c))

    # Model-overall TPC for delta-from-overall computations.
    overall_tpc: dict[str, float] = {}
    for m, pairs in by_model.items():
        a = np.asarray([p[0] for p in pairs])
        b = np.asarray([p[1] for p in pairs])
        overall_tpc[m] = aggregate_tpc(a, b)

    medias = sorted({r["media"] for r in rows if r["media"]})
    models = [m for m in MODEL_ORDER if m in by_model]

    # ----- P8: per-media balance -----
    print("\n[1/3] P8 — per-(model, media) TPC balance")
    print("-" * 76)
    print(f"  Refutation criterion: any |Δ from model-overall conv TPC| > 5%")
    print(f"  Bootstrap n=1000, seed=42")
    print()

    p8_csv = OUT_DIR / "05_p8_per_media.csv"
    n_outside_5pct = 0
    n_outside_3pct = 0
    p8_rows: list[tuple[str, str, int, float, float, float, float]] = []
    with p8_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_label", "media", "n",
                    "tpc", "tpc_ci_low", "tpc_ci_high",
                    "model_overall_tpc", "delta_pct", "within_5pct"])
        for m in models:
            ovr = overall_tpc[m]
            for media in medias:
                pairs = by_model_media.get((m, media), [])
                if not pairs:
                    continue
                tok = np.asarray([p[0] for p in pairs], dtype=np.int64)
                chr_ = np.asarray([p[1] for p in pairs], dtype=np.int64)
                tpc = aggregate_tpc(tok, chr_)
                lo, hi = bootstrap_tpc_ci(tok, chr_, seed=42)
                delta = (tpc - ovr) / ovr * 100
                within = abs(delta) <= 5.0
                if not within:
                    n_outside_5pct += 1
                if abs(delta) > 3.0:
                    n_outside_3pct += 1
                p8_rows.append((m, media, len(tok), tpc, lo, hi, delta))
                w.writerow([m, media, len(tok),
                            round(tpc, 4), round(lo, 4), round(hi, 4),
                            round(ovr, 4), round(delta, 2), within])

    # Pretty matrix print
    header = f"{'Model':<22} " + " ".join(f"{m:>10}" for m in medias) + f"   {'overall':>8}"
    print(header)
    print("-" * len(header))
    for m in models:
        cells = []
        for media in medias:
            pairs = by_model_media.get((m, media), [])
            if not pairs:
                cells.append(f"{'-':>10}")
                continue
            tok = np.asarray([p[0] for p in pairs], dtype=np.int64)
            chr_ = np.asarray([p[1] for p in pairs], dtype=np.int64)
            tpc = aggregate_tpc(tok, chr_)
            delta = (tpc - overall_tpc[m]) / overall_tpc[m] * 100
            cells.append(f"{tpc:.4f}{delta:+.1f}%"[:10].rjust(10))
        print(f"{m:<22} " + " ".join(cells) + f"   {overall_tpc[m]:>8.4f}")

    n_cells = len(models) * len(medias)
    print(f"\n  cells outside ±5%: {n_outside_5pct} / {n_cells}")
    print(f"  cells outside ±3%: {n_outside_3pct} / {n_cells}")
    p8_held = n_outside_5pct == 0
    print(f"  P8 (per-media within ±5%): {'HELD' if p8_held else 'FALSIFIED'}")

    sum_csv = OUT_DIR / "05_p8_summary.csv"
    with sum_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_models", "n_medias", "n_cells",
                    "n_outside_5pct", "n_outside_3pct",
                    "p8_verdict"])
        w.writerow([len(models), len(medias), n_cells,
                    n_outside_5pct, n_outside_3pct,
                    "HELD" if p8_held else "FALSIFIED"])
    print(f"  → {p8_csv.relative_to(ROOT)}")
    print(f"  → {sum_csv.relative_to(ROOT)}")

    # ----- Gemini origin: per-subject -----
    print("\n[2/3] Gemini -15.3% origin — per-subject decomposition")
    print("-" * 76)

    gem = "Gemini 2.5 Flash"
    gpt = "GPT-4o"
    if gem not in by_model or gpt not in by_model:
        print("[skip] Gemini or GPT-4o missing — cannot run origin analysis.")
        return

    gem_news = NEWS_TPC[gem]
    gem_overall_conv = overall_tpc[gem]
    gpt_overall_conv = overall_tpc[gpt]
    overall_ratio_news = NEWS_TPC[gem] / NEWS_TPC[gpt]
    overall_ratio_conv = gem_overall_conv / gpt_overall_conv
    print(f"  News overall:        Gemini {gem_news:.4f} / GPT {NEWS_TPC[gpt]:.4f} = {overall_ratio_news:.3f}×")
    print(f"  Conversation overall: Gemini {gem_overall_conv:.4f} / GPT {gpt_overall_conv:.4f} = {overall_ratio_conv:.3f}×")
    print(f"  Conversation Δ from news (Gemini): {(gem_overall_conv - gem_news) / gem_news * 100:+.1f}%")
    print()

    subj_rows: list[tuple[str, int, float, float, float, float]] = []
    subjects = sorted({r["subject"] for r in rows if r["subject"]})
    for subj in subjects:
        gem_pairs = by_model_subject.get((gem, subj), [])
        gpt_pairs = by_model_subject.get((gpt, subj), [])
        if not gem_pairs or not gpt_pairs:
            continue
        gt = np.asarray([p[0] for p in gem_pairs], dtype=np.int64)
        gc = np.asarray([p[1] for p in gem_pairs], dtype=np.int64)
        pt = np.asarray([p[0] for p in gpt_pairs], dtype=np.int64)
        pc = np.asarray([p[1] for p in gpt_pairs], dtype=np.int64)
        gem_tpc = aggregate_tpc(gt, gc)
        gpt_tpc = aggregate_tpc(pt, pc)
        ratio = gem_tpc / gpt_tpc
        delta_vs_news = (gem_tpc - gem_news) / gem_news * 100
        subj_rows.append((subj, len(gt), gem_tpc, gpt_tpc, ratio, delta_vs_news))

    subj_csv = OUT_DIR / "05_gemini_per_subject.csv"
    with subj_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "tier", "n",
                    "gemini_tpc", "gpt_tpc", "ratio_gem_over_gpt",
                    "delta_gemini_vs_news_pct"])
        for subj, n, g, p, r, d in subj_rows:
            w.writerow([subj, SUBJECT_TO_TIER.get(subj, ""), n,
                        round(g, 4), round(p, 4), round(r, 4), round(d, 2)])
    print(f"{'Subject':<14} {'tier':<14} {'n':>4} {'Gemini':>7} {'GPT':>7} {'ratio':>7} {'Δ vs news':>11}")
    for subj, n, g, p, r, d in sorted(subj_rows, key=lambda x: x[4]):  # sort by ratio asc
        tier = SUBJECT_TO_TIER.get(subj, "?")
        print(f"  {subj:<14} {tier:<14} {n:>4} {g:>7.4f} {p:>7.4f} {r:>7.3f} {d:>+10.1f}%")
    print(f"\n  → {subj_csv.relative_to(ROOT)}")

    # ----- Gemini origin: per-tier -----
    print("\n  Per pre-registered tier:")
    tier_csv = OUT_DIR / "05_gemini_per_tier.csv"
    with tier_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tier", "n",
                    "gemini_tpc", "gpt_tpc", "ratio_gem_over_gpt",
                    "delta_gemini_vs_news_pct"])
        for tier in TIER_ORDER:
            gem_pairs = by_model_tier.get((gem, tier), [])
            gpt_pairs = by_model_tier.get((gpt, tier), [])
            gt = np.asarray([p[0] for p in gem_pairs], dtype=np.int64)
            gc = np.asarray([p[1] for p in gem_pairs], dtype=np.int64)
            pt = np.asarray([p[0] for p in gpt_pairs], dtype=np.int64)
            pc = np.asarray([p[1] for p in gpt_pairs], dtype=np.int64)
            g, p = aggregate_tpc(gt, gc), aggregate_tpc(pt, pc)
            r = g / p
            d = (g - gem_news) / gem_news * 100
            w.writerow([tier, len(gt),
                        round(g, 4), round(p, 4), round(r, 4), round(d, 2)])
            print(f"    {tier:<14} n={len(gt):>4}  Gemini={g:.4f}  GPT={p:.4f}  ratio={r:.3f}  Δ={d:+.1f}%")
    print(f"  → {tier_csv.relative_to(ROOT)}")

    # ----- Gemini origin: per-media -----
    print("\n[3/3] Gemini -15.3% origin — per-media decomposition")
    print("-" * 76)
    media_csv = OUT_DIR / "05_gemini_per_media.csv"
    print(f"{'Media':<12} {'n':>4} {'Gemini':>7} {'GPT':>7} {'ratio':>7} {'Δ vs news':>11}")
    with media_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["media", "n",
                    "gemini_tpc", "gpt_tpc", "ratio_gem_over_gpt",
                    "delta_gemini_vs_news_pct"])
        for media in medias:
            gem_pairs = by_model_media.get((gem, media), [])
            gpt_pairs = by_model_media.get((gpt, media), [])
            gt = np.asarray([p[0] for p in gem_pairs], dtype=np.int64)
            gc = np.asarray([p[1] for p in gem_pairs], dtype=np.int64)
            pt = np.asarray([p[0] for p in gpt_pairs], dtype=np.int64)
            pc = np.asarray([p[1] for p in gpt_pairs], dtype=np.int64)
            g, p = aggregate_tpc(gt, gc), aggregate_tpc(pt, pc)
            r = g / p
            d = (g - gem_news) / gem_news * 100
            w.writerow([media, len(gt),
                        round(g, 4), round(p, 4), round(r, 4), round(d, 2)])
            print(f"  {media:<12} {len(gt):>4} {g:>7.4f} {p:>7.4f} {r:>7.3f} {d:>+10.1f}%")
    print(f"  → {media_csv.relative_to(ROOT)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
