"""P7 sub-analysis — 3-tier entity-density gradient on conversation n=1000.

Reads ``results/raw/03_tpc_conversation_n1000_per_text.csv`` and computes
per-(model, tier) aggregate TPC with bootstrap CIs, plus the statistical
tests pre-registered in ``notes/05_next_steps.md``:

1. **Within-model tier comparison** — Mann-Whitney U (rich vs light) and
   Kruskal-Wallis (rich vs neutral vs light) on per-text TPCs. These are
   independent-sample tests because tiers contain disjoint texts.
2. **Cross-model gradient consistency** — Friedman test on the 3
   tier-mean TPCs across the 7 models, treating each model as a block.
   This tests whether the rich > neutral > light ordering is consistent
   *across* models, which is the central P7 claim.
3. **EXAONE mechanism check** — relative rich-to-light gap
   ``(TPC_rich - TPC_light) / TPC_light`` per model, with bootstrap CI.
   Mechanism predicts EXAONE has the smallest gap (vocabulary extension
   buys robustness against entity-dense surface forms).

Outputs
-------
- ``results/raw/04_p7_subject_tiers.csv`` — per-(model, tier) TPC + CI
- ``results/raw/04_p7_within_model_tests.csv`` — MWU + KW p-values
- ``results/raw/04_p7_friedman.csv`` — single-row global test
- ``results/raw/04_p7_relative_gap.csv`` — per-model rich-vs-light gap
- ``results/figures/04_p7_subject_tiers.png`` — 7 × 3 tier-line plot
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from korean_llm_cost.subject_groups import (  # noqa: E402
    ENTITY_LIGHT,
    ENTITY_RICH,
    NEUTRAL,
    SUBJECT_TO_TIER,
    TIER_ORDER,
)


PER_TEXT_CSV = ROOT / "results" / "raw" / "03_tpc_conversation_n1000_per_text.csv"

OUT_DIR = ROOT / "results" / "raw"
FIG_DIR = ROOT / "results" / "figures"

# Sort models in a stable order matching the headline result table.
MODEL_ORDER = (
    "EXAONE 3.5 7.8B",
    "Gemini 2.5 Flash",
    "GPT-4o",
    "Llama 3.1 8B",
    "Qwen 2.5 7B",
    "Claude Sonnet 4.5",
    "Solar 10.7B",
)


def load_rows() -> list[dict]:
    with PER_TEXT_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate_tpc(tokens: np.ndarray, chars: np.ndarray) -> float:
    return float(tokens.sum() / chars.sum())


def bootstrap_tpc_ci(
    tokens: np.ndarray,
    chars: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Aggregate-style bootstrap (resample paired (token,char) entries)."""
    rng = np.random.default_rng(seed)
    n = len(tokens)
    ratios = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        ratios[b] = tokens[idx].sum() / chars[idx].sum()
    alpha = (1 - ci) / 2
    return float(np.quantile(ratios, alpha)), float(np.quantile(ratios, 1 - alpha))


def bootstrap_relative_gap_ci(
    rich_tok: np.ndarray, rich_chr: np.ndarray,
    light_tok: np.ndarray, light_chr: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap CI for ((TPC_rich - TPC_light) / TPC_light)."""
    rng = np.random.default_rng(seed)
    n_r, n_l = len(rich_tok), len(light_tok)
    gaps = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        ir = rng.integers(0, n_r, n_r)
        il = rng.integers(0, n_l, n_l)
        tpc_r = rich_tok[ir].sum() / rich_chr[ir].sum()
        tpc_l = light_tok[il].sum() / light_chr[il].sum()
        gaps[b] = (tpc_r - tpc_l) / tpc_l
    alpha = (1 - ci) / 2
    return float(np.quantile(gaps, alpha)), float(np.quantile(gaps, 1 - alpha))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("P7 sub-analysis — 3-tier entity-density gradient")
    print(f"  source: {PER_TEXT_CSV.relative_to(ROOT)}")
    print(f"  pre-registration: src/korean_llm_cost/subject_groups.py")
    print("=" * 76)

    rows = load_rows()

    # Group per (model_label, tier) -> arrays
    bucket_tok: dict[tuple[str, str], list[int]] = defaultdict(list)
    bucket_chr: dict[tuple[str, str], list[int]] = defaultdict(list)
    bucket_tpc_per_text: dict[tuple[str, str], list[float]] = defaultdict(list)

    skipped = 0
    for r in rows:
        subj = r["subject"]
        if subj not in SUBJECT_TO_TIER:
            skipped += 1
            continue
        tier = SUBJECT_TO_TIER[subj]
        key = (r["model_label"], tier)
        t = int(r["n_tokens"])
        c = int(r["n_chars"])
        bucket_tok[key].append(t)
        bucket_chr[key].append(c)
        bucket_tpc_per_text[key].append(t / c)
    if skipped:
        print(f"[warn] {skipped} rows had unknown subjects (skipped)")

    models = [m for m in MODEL_ORDER if (m, "entity-rich") in bucket_tok]
    if len(models) != len(MODEL_ORDER):
        missing = [m for m in MODEL_ORDER if m not in models]
        print(f"[warn] models missing in CSV: {missing}")

    print(f"\n[1/4] Per-(model, tier) aggregate TPC + 95% bootstrap CI")
    print("-" * 76)
    header = f"{'Model':<22} {'Tier':<14} {'n':>5} {'TPC':>7} {'CI95':>17}"
    print(header)
    print("-" * len(header))

    tier_csv = OUT_DIR / "04_p7_subject_tiers.csv"
    per_model_tier_tpc: dict[tuple[str, str], float] = {}
    per_model_tier_ci: dict[tuple[str, str], tuple[float, float]] = {}
    with tier_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_label", "tier", "n", "tokens_total", "chars_total",
                    "tpc", "tpc_ci_low", "tpc_ci_high"])
        for m in models:
            for tier in TIER_ORDER:
                tok = np.asarray(bucket_tok[(m, tier)], dtype=np.int64)
                chr_ = np.asarray(bucket_chr[(m, tier)], dtype=np.int64)
                tpc = aggregate_tpc(tok, chr_)
                lo, hi = bootstrap_tpc_ci(tok, chr_, seed=42)
                per_model_tier_tpc[(m, tier)] = tpc
                per_model_tier_ci[(m, tier)] = (lo, hi)
                w.writerow([m, tier, len(tok), int(tok.sum()), int(chr_.sum()),
                            round(tpc, 4), round(lo, 4), round(hi, 4)])
                ci_str = f"[{lo:.3f},{hi:.3f}]"
                print(f"{m:<22} {tier:<14} {len(tok):>5} {tpc:>7.4f} {ci_str:>17}")
            print()
    print(f"  → {tier_csv.relative_to(ROOT)}")

    # ----- 2) Monotonic gradient check -----
    print("\n[2/4] Monotonic ordering check (rich > neutral > light per model)")
    print("-" * 76)
    monotonic_count = 0
    for m in models:
        r_tpc = per_model_tier_tpc[(m, "entity-rich")]
        n_tpc = per_model_tier_tpc[(m, "neutral")]
        l_tpc = per_model_tier_tpc[(m, "entity-light")]
        ok = r_tpc > n_tpc > l_tpc
        if ok:
            monotonic_count += 1
        verdict = "OK" if ok else "VIOLATED"
        print(f"  {m:<22}  rich={r_tpc:.4f}  neutral={n_tpc:.4f}  light={l_tpc:.4f}  {verdict}")
    print(f"\n  Monotonic in {monotonic_count}/{len(models)} models.")

    # ----- 3) Within-model nonparametric tests -----
    print("\n[3/4] Within-model nonparametric tier tests (per-text TPCs)")
    print("-" * 76)
    from scipy.stats import mannwhitneyu, kruskal, friedmanchisquare
    test_csv = OUT_DIR / "04_p7_within_model_tests.csv"
    print(f"{'Model':<22} {'MWU rich-vs-light':>22} {'Kruskal-Wallis':>20}")
    with test_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_label", "mwu_statistic", "mwu_pvalue",
                    "kw_statistic", "kw_pvalue"])
        for m in models:
            rich = np.asarray(bucket_tpc_per_text[(m, "entity-rich")])
            light = np.asarray(bucket_tpc_per_text[(m, "entity-light")])
            neutral = np.asarray(bucket_tpc_per_text[(m, "neutral")])
            mwu = mannwhitneyu(rich, light, alternative="greater")
            kw = kruskal(rich, neutral, light)
            w.writerow([m,
                        round(float(mwu.statistic), 2), float(mwu.pvalue),
                        round(float(kw.statistic), 2), float(kw.pvalue)])
            print(f"  {m:<22}  U={float(mwu.statistic):>10.0f}  p={float(mwu.pvalue):.2e}    "
                  f"H={float(kw.statistic):>5.1f}  p={float(kw.pvalue):.2e}")
    print(f"  → {test_csv.relative_to(ROOT)}")

    # ----- Friedman global test -----
    # Treat each model as a block; ranks across tiers within model.
    rich_means = [per_model_tier_tpc[(m, "entity-rich")] for m in models]
    neutral_means = [per_model_tier_tpc[(m, "neutral")] for m in models]
    light_means = [per_model_tier_tpc[(m, "entity-light")] for m in models]
    fr = friedmanchisquare(rich_means, neutral_means, light_means)
    print(f"\n  Friedman across {len(models)} models on (rich, neutral, light) tier-means:")
    print(f"    chi2={float(fr.statistic):.3f}, df=2, p={float(fr.pvalue):.2e}")

    fr_csv = OUT_DIR / "04_p7_friedman.csv"
    with fr_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_models", "tiers", "chi2", "df", "pvalue", "interpretation"])
        w.writerow([len(models), "rich,neutral,light",
                    round(float(fr.statistic), 4), 2, float(fr.pvalue),
                    "Tests whether tier-mean TPC differs across the 3 tiers, treating each model as a block."])
    print(f"  → {fr_csv.relative_to(ROOT)}")

    # ----- 4) Relative gap (mechanism check) -----
    print("\n[4/4] Relative rich-vs-light gap (mechanism: smallest for EXAONE?)")
    print("-" * 76)
    gap_csv = OUT_DIR / "04_p7_relative_gap.csv"
    rows_for_print: list[tuple[str, float, float, float]] = []
    with gap_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_label", "tpc_rich", "tpc_light",
                    "rel_gap", "rel_gap_ci_low", "rel_gap_ci_high"])
        for m in models:
            r_tok = np.asarray(bucket_tok[(m, "entity-rich")], dtype=np.int64)
            r_chr = np.asarray(bucket_chr[(m, "entity-rich")], dtype=np.int64)
            l_tok = np.asarray(bucket_tok[(m, "entity-light")], dtype=np.int64)
            l_chr = np.asarray(bucket_chr[(m, "entity-light")], dtype=np.int64)
            tpc_r = aggregate_tpc(r_tok, r_chr)
            tpc_l = aggregate_tpc(l_tok, l_chr)
            gap = (tpc_r - tpc_l) / tpc_l
            lo, hi = bootstrap_relative_gap_ci(r_tok, r_chr, l_tok, l_chr, seed=42)
            w.writerow([m, round(tpc_r, 4), round(tpc_l, 4),
                        round(gap, 4), round(lo, 4), round(hi, 4)])
            rows_for_print.append((m, gap, lo, hi))
    rows_for_print.sort(key=lambda x: x[1])
    print(f"{'Model':<22} {'rel_gap':>9} {'CI95':>20}")
    for m, gap, lo, hi in rows_for_print:
        ci = f"[{lo:+.3f},{hi:+.3f}]"
        marker = "  <-- smallest" if (m, gap, lo, hi) == rows_for_print[0] else ""
        print(f"  {m:<20} {gap:>+8.3f}  {ci:>20}{marker}")
    print(f"  → {gap_csv.relative_to(ROOT)}")

    # ----- Figure -----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        x_positions = np.arange(len(TIER_ORDER))
        for m in models:
            ys = [per_model_tier_tpc[(m, t)] for t in TIER_ORDER]
            errs_low = [per_model_tier_tpc[(m, t)] - per_model_tier_ci[(m, t)][0]
                        for t in TIER_ORDER]
            errs_high = [per_model_tier_ci[(m, t)][1] - per_model_tier_tpc[(m, t)]
                         for t in TIER_ORDER]
            ax.errorbar(x_positions, ys,
                        yerr=[errs_low, errs_high],
                        marker="o", capsize=3, label=m)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(TIER_ORDER)
        ax.set_xlabel("Subject tier (pre-registered)")
        ax.set_ylabel("Aggregate TPC (tokens / Korean char)")
        ax.set_title("P7 — entity-density gradient across 7 models (conversation n=1000)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig_path = FIG_DIR / "04_p7_subject_tiers.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"\n  figure: {fig_path.relative_to(ROOT)}")
    except ImportError:
        print("\n  [skip] matplotlib not installed; skipping figure.")

    print("\nDone.")


if __name__ == "__main__":
    main()
