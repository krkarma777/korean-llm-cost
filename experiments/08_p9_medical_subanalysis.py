"""P9 medical sub-analysis — per-config + length-stratified + confound.

Reads ``results/raw/07_tpc_medical_n1000_per_text.csv`` (7 models × 1000
stems = 7000 rows, written by ``07_tpc_medical_n1000.py``) and produces
the post-headline sub-analysis pre-specified in
``notes/05_next_steps.md``. No new tokenizer calls — purely re-grouping
the existing per-text CSV.

Six analyses
------------
(a) Per-(model, config) aggregate TPC + 95% bootstrap CI
    7 models × 4 configs = 28 cells, each n=250.

(b) Per-(model, length-bucket) aggregate TPC + 95% bootstrap CI
    7 models × 3 buckets (short / medium / long), bucket boundaries
    pre-registered in ``length_buckets`` (commit 749313e).

(c) EXAONE per-config KPR/GPT verdicts
    For each of the 4 configs, compute EXAONE_TPC / GPT-4o_TPC at
    the same n=250 stems, then feed through
    ``medical_predictions.classify_kpr_gpt`` to get an A/C/B/buffer/
    outside verdict per config. Print a summary line that says
    whether verdict 'B' (the headline result at full n=1000) is
    robust across all 4 configs, or whether any config falls in a
    different band — which would be a sub-finding requiring its own
    Discussion paragraph.

(d) Cross-config Friedman test
    Treat models as blocks, configs as treatments. Tests whether the
    config-level TPC ranking is consistent across the 7 models
    (i.e., does every model agree on which config tokenizes most
    efficiently?). p < 0.05 means significant disagreement.

(e) Cross-bucket Friedman test
    Same idea, swapping configs for length buckets. Tests whether the
    bucket-level TPC ranking is consistent across the 7 models.

(f) Confound analysis — config × bucket joint distribution
    Quantifies how aligned config and length are in this corpus
    (large diagonal dominance ⇒ length effect and config effect
    cannot be cleanly separated; small ⇒ they can).

Outputs
-------
- ``results/raw/08_p9_per_config_aggregate.csv``
- ``results/raw/08_p9_length_stratified_aggregate.csv``
- ``results/raw/08_p9_exaone_per_config_verdicts.csv``
- ``results/raw/08_p9_friedman_results.csv``
- ``results/raw/08_p9_confound_table.csv``
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from korean_llm_cost import medical_predictions as mp  # noqa: E402
from korean_llm_cost.length_buckets import (  # noqa: E402
    BUCKET_ORDER,
    classify_length_bucket,
)

PER_TEXT_CSV = ROOT / "results" / "raw" / "07_tpc_medical_n1000_per_text.csv"
OUT_DIR = ROOT / "results" / "raw"

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
    print("P9 medical sub-analysis — per-config + length-stratified + confound")
    print(f"  source : {PER_TEXT_CSV.relative_to(ROOT)}")
    print(f"  pre-reg: medical_predictions.py @ de53094, "
          f"length_buckets.py @ 749313e")
    print("=" * 76)

    if not PER_TEXT_CSV.exists():
        print(f"[error] per-text CSV not found at {PER_TEXT_CSV}.")
        print("  Run experiments/07_tpc_medical_n1000.py first.")
        sys.exit(1)

    with PER_TEXT_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Index by (model, config) and (model, bucket).
    by_model_config: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    by_model_bucket: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    config_bucket_count: dict[tuple[str, str], int] = defaultdict(int)
    config_bucket_count_seen: set[tuple[str, int]] = set()

    for r in rows:
        m = r["model_label"]
        config = r["config"]
        n_chars = int(r["n_chars"])
        n_tokens = int(r["n_tokens"])
        bucket = classify_length_bucket(n_chars)
        by_model_config[(m, config)].append((n_tokens, n_chars))
        by_model_bucket[(m, bucket)].append((n_tokens, n_chars))
        # Confound table: count each text once (text_id is unique
        # per stem; one row per model so dedupe by text_id).
        text_id = int(r["text_id"])
        if text_id not in config_bucket_count_seen:
            config_bucket_count[(config, bucket)] += 1
            config_bucket_count_seen.add(text_id)

    models = [m for m in MODEL_ORDER
              if any((m, c) in by_model_config for c in mp.KORMEDMCQA_CONFIGS)]
    configs = list(mp.KORMEDMCQA_CONFIGS)
    buckets = list(BUCKET_ORDER)

    # ----- (a) Per-config TPC + CI95 -----
    print("\n[a] Per-(model, config) aggregate TPC + 95% bootstrap CI")
    print("-" * 76)
    print(f"{'Model':<22} " + " ".join(f"{c:>11}" for c in configs))
    print("-" * (22 + 12 * len(configs)))

    per_config_csv = OUT_DIR / "08_p9_per_config_aggregate.csv"
    per_config_tpc: dict[tuple[str, str], float] = {}
    per_config_ci: dict[tuple[str, str], tuple[float, float]] = {}
    with per_config_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_label", "config", "n",
                    "tokens_total", "chars_total",
                    "tpc", "tpc_ci_low", "tpc_ci_high"])
        for m in models:
            row_cells = []
            for cfg in configs:
                pairs = by_model_config[(m, cfg)]
                tok = np.asarray([p[0] for p in pairs], dtype=np.int64)
                chr_ = np.asarray([p[1] for p in pairs], dtype=np.int64)
                tpc = aggregate_tpc(tok, chr_)
                lo, hi = bootstrap_tpc_ci(tok, chr_, seed=42)
                per_config_tpc[(m, cfg)] = tpc
                per_config_ci[(m, cfg)] = (lo, hi)
                w.writerow([m, cfg, len(tok),
                            int(tok.sum()), int(chr_.sum()),
                            round(tpc, 4), round(lo, 4), round(hi, 4)])
                row_cells.append(f"{tpc:.4f}".rjust(11))
            print(f"{m:<22} " + " ".join(row_cells))
    print(f"  → {per_config_csv.relative_to(ROOT)}")

    # ----- (b) Length-stratified TPC + CI95 -----
    print("\n[b] Per-(model, length-bucket) aggregate TPC + 95% bootstrap CI")
    print(f"  buckets: short ≤ 50, medium 51-150, long > 150")
    print("-" * 76)
    print(f"{'Model':<22} " + " ".join(f"{b:>11}" for b in buckets))
    print("-" * (22 + 12 * len(buckets)))

    bucket_csv = OUT_DIR / "08_p9_length_stratified_aggregate.csv"
    per_bucket_tpc: dict[tuple[str, str], float] = {}
    per_bucket_ci: dict[tuple[str, str], tuple[float, float]] = {}
    with bucket_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_label", "bucket", "n",
                    "tokens_total", "chars_total",
                    "tpc", "tpc_ci_low", "tpc_ci_high"])
        for m in models:
            row_cells = []
            for buck in buckets:
                pairs = by_model_bucket[(m, buck)]
                tok = np.asarray([p[0] for p in pairs], dtype=np.int64)
                chr_ = np.asarray([p[1] for p in pairs], dtype=np.int64)
                if len(tok) == 0:
                    per_bucket_tpc[(m, buck)] = float("nan")
                    per_bucket_ci[(m, buck)] = (float("nan"), float("nan"))
                    row_cells.append("nan".rjust(11))
                    continue
                tpc = aggregate_tpc(tok, chr_)
                lo, hi = bootstrap_tpc_ci(tok, chr_, seed=42)
                per_bucket_tpc[(m, buck)] = tpc
                per_bucket_ci[(m, buck)] = (lo, hi)
                w.writerow([m, buck, len(tok),
                            int(tok.sum()), int(chr_.sum()),
                            round(tpc, 4), round(lo, 4), round(hi, 4)])
                row_cells.append(f"{tpc:.4f}".rjust(11))
            print(f"{m:<22} " + " ".join(row_cells))
    print(f"  → {bucket_csv.relative_to(ROOT)}")

    # ----- (c) EXAONE per-config verdicts -----
    print("\n[c] EXAONE per-config KPR/GPT verdicts (vs medical headline 'B')")
    print("-" * 76)
    exa_label = "EXAONE 3.5 7.8B"
    gpt_label = "GPT-4o"
    if exa_label not in models or gpt_label not in models:
        print("[skip] EXAONE or GPT-4o missing — cannot compute per-config verdicts.")
        return

    verdict_csv = OUT_DIR / "08_p9_exaone_per_config_verdicts.csv"
    verdicts: list[tuple[str, float, str]] = []
    with verdict_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "exaone_tpc", "gpt_tpc",
                    "exaone_kpr_gpt", "verdict"])
        for cfg in configs:
            exa = per_config_tpc[(exa_label, cfg)]
            gpt = per_config_tpc[(gpt_label, cfg)]
            kpr_gpt = exa / gpt
            verdict = mp.classify_kpr_gpt(kpr_gpt)
            verdicts.append((cfg, kpr_gpt, verdict))
            w.writerow([cfg, round(exa, 4), round(gpt, 4),
                        round(kpr_gpt, 4), verdict])
    print(f"EXAONE per-config verdicts:")
    for cfg, kpr, vd in verdicts:
        print(f"  {cfg:<8} KPR/GPT {kpr:.4f} → {vd!r}")

    # Summary line — captures whether verdict 'B' is robust
    verdict_counter = Counter(vd for _, _, vd in verdicts)
    n_B = verdict_counter.get("B", 0)
    n_total = len(verdicts)
    if n_B == n_total:
        summary_line = (
            f"  → {n_B}/{n_total} 'B' band: medical verdict 'B' "
            f"robust across configs"
        )
    else:
        # Compose "X/N 'B', Y/N 'OTHER'" listing all unique verdicts
        # in a deterministic order matching pre-registered band order.
        order = ["A", "between-A-and-C", "C", "between-C-and-B", "B",
                 "below-all", "above-all"]
        parts = [f"{verdict_counter[v]}/{n_total} {v!r}"
                 for v in order if verdict_counter.get(v)]
        non_B_configs = [cfg for cfg, _, vd in verdicts if vd != "B"]
        if non_B_configs:
            non_B_text = (
                f" (config{'s' if len(non_B_configs) > 1 else ''} "
                f"{', '.join(non_B_configs)})"
            )
        else:
            non_B_text = ""
        if n_B >= 1:
            summary_line = (
                f"  → {', '.join(parts)}: verdict 'B' present but with "
                f"non-'B' config{'s' if len(non_B_configs) > 1 else ''} "
                f"({', '.join(non_B_configs)}) — sub-finding"
            )
        else:
            summary_line = (
                f"  → {', '.join(parts)}: verdict 'B' NOT robust at "
                f"per-config granularity — sub-finding{non_B_text}"
            )
    print(summary_line)
    print(f"  → {verdict_csv.relative_to(ROOT)}")

    # ----- (d) + (e) Friedman tests -----
    print("\n[d/e] Friedman tests — model TPC ranking consistency across "
          "configs / buckets")
    print("-" * 76)
    from scipy.stats import friedmanchisquare

    friedman_csv = OUT_DIR / "08_p9_friedman_results.csv"
    with friedman_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["test", "n_blocks_models", "n_treatments",
                    "treatments", "chi2", "df", "pvalue", "interpretation"])

        # (d) cross-config: 4 treatments (configs), 7 blocks (models)
        config_groups = [
            [per_config_tpc[(m, cfg)] for m in models] for cfg in configs
        ]
        fr_cfg = friedmanchisquare(*config_groups)
        cfg_interp = (
            "Configs differ in TPC (model rankings disagree)"
            if fr_cfg.pvalue < 0.05 else
            "No significant per-config TPC difference (model rankings consistent)"
        )
        w.writerow(["cross-config", len(models), len(configs),
                    ",".join(configs),
                    round(float(fr_cfg.statistic), 4),
                    len(configs) - 1, float(fr_cfg.pvalue),
                    cfg_interp])
        print(f"  cross-config Friedman ({len(configs)} configs, "
              f"{len(models)} models as blocks):")
        print(f"    chi2={float(fr_cfg.statistic):.3f}, "
              f"df={len(configs)-1}, p={float(fr_cfg.pvalue):.4e}")
        print(f"    {cfg_interp}")

        # (e) cross-bucket: 3 treatments (buckets), 7 blocks (models)
        # Skip if any bucket has any nan TPC (shouldn't happen with n=1000).
        bucket_groups = [
            [per_bucket_tpc[(m, b)] for m in models] for b in buckets
        ]
        if any(any(np.isnan(v) for v in g) for g in bucket_groups):
            print(f"  cross-bucket Friedman: skipped (some buckets empty)")
        else:
            fr_b = friedmanchisquare(*bucket_groups)
            buck_interp = (
                "Buckets differ in TPC (model rankings disagree)"
                if fr_b.pvalue < 0.05 else
                "No significant per-bucket TPC difference (model rankings consistent)"
            )
            w.writerow(["cross-bucket", len(models), len(buckets),
                        ",".join(buckets),
                        round(float(fr_b.statistic), 4),
                        len(buckets) - 1, float(fr_b.pvalue),
                        buck_interp])
            print(f"  cross-bucket Friedman ({len(buckets)} buckets, "
                  f"{len(models)} models as blocks):")
            print(f"    chi2={float(fr_b.statistic):.3f}, "
                  f"df={len(buckets)-1}, p={float(fr_b.pvalue):.4e}")
            print(f"    {buck_interp}")
    print(f"  → {friedman_csv.relative_to(ROOT)}")

    # ----- (f) Confound table — config × bucket joint distribution -----
    print("\n[f] Confound — config × bucket joint distribution (text counts)")
    print("-" * 76)
    print(f"  Total stems = {sum(config_bucket_count.values())} "
          f"(should be 1000)")
    print(f"{'config':<10} " + " ".join(f"{b:>9}" for b in buckets) + f"   {'total':>7}")
    print("-" * (10 + 10 * len(buckets) + 10))
    confound_csv = OUT_DIR / "08_p9_confound_table.csv"
    diag_dominant_count = 0
    expected_alignment = {
        "dentist": "short",
        "doctor":  "long",
        "nurse":   "short",  # pilot median 64, slight overlap with medium
        "pharm":   "medium",
    }
    with confound_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config"] + list(buckets) + ["total", "expected_bucket",
                                                  "expected_share_pct"])
        for cfg in configs:
            counts = [config_bucket_count.get((cfg, b), 0) for b in buckets]
            total = sum(counts)
            cells = [f"{c:>9d}" for c in counts]
            exp_b = expected_alignment[cfg]
            exp_n = config_bucket_count.get((cfg, exp_b), 0)
            exp_pct = exp_n / total * 100 if total else 0.0
            if exp_pct >= 50:
                diag_dominant_count += 1
            w.writerow([cfg] + counts + [total, exp_b, round(exp_pct, 2)])
            print(f"{cfg:<10} " + " ".join(cells) + f"   {total:>7d}")
    print()
    print("Bucket dominance per config (expected from pilot):")
    for cfg, exp_b in expected_alignment.items():
        n = config_bucket_count.get((cfg, exp_b), 0)
        total = sum(config_bucket_count.get((cfg, b), 0) for b in buckets)
        pct = n / total * 100 if total else 0.0
        marker = "(diag-dominant)" if pct >= 50 else "(dispersed)"
        print(f"  {cfg:<10} → {exp_b:<7} {n:>4} / {total} ({pct:.1f}%)  {marker}")
    n_configs = len(expected_alignment)
    if diag_dominant_count == n_configs:
        confound_summary = (
            f"  Confound: HIGH — every config is diag-dominant in its "
            f"expected bucket. Length and config are largely aligned in "
            f"this corpus; cross-bucket-within-config TPC differences "
            f"are the cleaner length-effect estimator."
        )
    elif diag_dominant_count >= n_configs - 1:
        confound_summary = (
            f"  Confound: MODERATE — {diag_dominant_count}/{n_configs} "
            f"configs diag-dominant. Length effect partially separable."
        )
    else:
        confound_summary = (
            f"  Confound: LOW — {diag_dominant_count}/{n_configs} configs "
            f"diag-dominant. Length and config are largely separable in "
            f"this corpus."
        )
    print()
    print(confound_summary)
    print(f"  → {confound_csv.relative_to(ROOT)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
