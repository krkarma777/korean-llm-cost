"""Phase 1 — n=1000 KorMedMCQA medical measurement.

Mirror of ``03_tpc_conversation_n1000.py`` adapted for the medical
category (KorMedMCQA, ``train`` split, 4-config stratified balanced
sample of 250 × 4 = 1000 stems). All sampling-protocol constants
(repo, configs, per-config count, seed, length filter, measurement
unit) come from ``medical_predictions``, the binding pre-registration
artifact (commit de53094).

Differences from the conversation run:

- Loader is ``corpus_loader.load_category("medical", n=1000)``, which
  performs HF dataset loading and 4-config stratified sampling.
- Per-text CSV carries five extra columns from the KorMedMCQA metadata
  in place of conversation's six: ``config``, ``year``, ``period``,
  ``q_number``, ``question_id``. This enables the per-config (P9)
  sub-analysis that mirrors P8's per-media check.
- Output filenames use the ``07_tpc_medical_n1000_*`` prefix.
- After the headline aggregate is computed, the script *automatically*
  feeds EXAONE's KPR/GPT into ``medical_predictions.classify_kpr_gpt``
  and prints the pre-registered verdict (one of 'A', 'between-A-and-C',
  'C', 'between-C-and-B', 'B', 'below-all', 'above-all'). The verdict
  is also written to the aggregate CSV as a stand-alone column on the
  EXAONE row, and to a single-row ``07_tpc_medical_n1000_verdict.csv``
  for downstream pipelines.

Outputs
-------
- ``results/raw/07_tpc_medical_n1000_per_text.csv``
- ``results/raw/07_tpc_medical_n1000_aggregate.csv``
- ``results/raw/07_tpc_medical_n1000_pairwise.csv``
- ``results/raw/07_tpc_medical_n1000_verdict.csv``
"""

from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from korean_llm_cost import medical_predictions as mp
from korean_llm_cost.corpus_loader import load_category
from korean_llm_cost.metrics import (
    CountSet,
    ModelResult,
    paired_wilcoxon_tpc,
)


# ----- Cross-category baselines (for delta tracking, P3 + (e)) -----

N1000_NEWS_TPC = {
    "gpt-4o": 0.7226,
    "claude-sonnet-4-5": 1.0831,
    "gemini-2.5-flash": 0.6961,
    "upstage/SOLAR-10.7B-Instruct-v1.0": 1.2055,
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct": 0.5468,
    "Qwen/Qwen2.5-7B-Instruct": 0.8226,
    "meta-llama/Llama-3.1-8B-Instruct": 0.7325,
}

N1000_CONV_TPC = {
    "gpt-4o": 0.6659,
    "claude-sonnet-4-5": 1.1547,
    "gemini-2.5-flash": 0.5894,
    "upstage/SOLAR-10.7B-Instruct-v1.0": 1.2408,
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct": 0.5235,
    "Qwen/Qwen2.5-7B-Instruct": 0.7492,
    "meta-llama/Llama-3.1-8B-Instruct": 0.6754,
}


# ----- .env loader (same as the conversation run) -----

def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if not v:
            continue
        if not os.environ.get(k):
            os.environ[k] = v


def init_tokenizers() -> dict:
    """Same tokenizer set as 03_tpc_conversation_n1000."""
    toks: dict[str, object] = {}
    from korean_llm_cost.tokenizers.openai_tok import OpenAITokenizer
    toks["GPT-4o"] = OpenAITokenizer("gpt-4o")

    if os.environ.get("ANTHROPIC_API_KEY"):
        from korean_llm_cost.tokenizers.anthropic_tok import AnthropicTokenizer
        toks["Claude Sonnet 4.5"] = AnthropicTokenizer("claude-sonnet-4-5")
    else:
        print("[skip] ANTHROPIC_API_KEY not set; Claude omitted.")

    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        from korean_llm_cost.tokenizers.google_tok import GoogleTokenizer
        toks["Gemini 2.5 Flash"] = GoogleTokenizer("gemini-2.5-flash")
    else:
        print("[skip] GOOGLE_API_KEY not set; Gemini omitted.")

    from korean_llm_cost.tokenizers.hf_tok import HFTokenizer
    toks["Solar 10.7B"] = HFTokenizer("upstage/SOLAR-10.7B-Instruct-v1.0")
    toks["EXAONE 3.5 7.8B"] = HFTokenizer(
        "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct", trust_remote_code=True
    )
    toks["Qwen 2.5 7B"] = HFTokenizer("Qwen/Qwen2.5-7B-Instruct")
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        toks["Llama 3.1 8B"] = HFTokenizer("meta-llama/Llama-3.1-8B-Instruct")
    else:
        print("[skip] HF_TOKEN not set; Llama 3.1 omitted.")
    return toks


# ----- Checkpointed CSV writer (medical metadata: 5 fields) -----

@dataclass
class CheckpointWriter:
    path: Path
    fieldnames: tuple[str, ...]

    def __post_init__(self) -> None:
        self._existing_rows: list[dict[str, str]] = []
        self.done: set[tuple[str, int]] = set()
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self._existing_rows.append(row)
                    self.done.add((row["model_name"], int(row["text_id"])))
        self._fh = self.path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
        if not self._existing_rows:
            self._writer.writeheader()
            self._fh.flush()

    def write(self, row: dict) -> None:
        key = (row["model_name"], int(row["text_id"]))
        if key in self.done:
            return
        self._writer.writerow(row)
        self._fh.flush()
        self.done.add(key)

    def close(self) -> None:
        self._fh.close()


# ----- Verdict helpers -----

def tier_for(kpr_gpt: float) -> str:
    if kpr_gpt <= 1.0 / 1.3:
        return "advantage"
    if kpr_gpt < 1.3:
        return "efficient"
    return "penalty"


def _delta_pct(now: float, ref: float | None) -> str:
    if ref is None:
        return "(no baseline)"
    return f"{(now - ref) / ref * 100:+.1f}%"


# ----- Main -----

def main() -> None:
    _load_env_file(ROOT / ".env")
    n_target = mp.TOTAL_SAMPLE  # 1000
    out_dir = ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f"Phase 1 — medical main run: n={n_target}")
    print(f"       KorMedMCQA, train split, {len(mp.KORMEDMCQA_CONFIGS)} configs × "
          f"{mp.PER_CONFIG_SAMPLE} (stratified)")
    print(f"       pre-registration: src/korean_llm_cost/medical_predictions.py")
    print("=" * 76)

    print("\n[1/5] Loading corpus...")
    load = load_category("medical", n=n_target, seed=mp.SAMPLE_SEED,
                         lo_chars=mp.LENGTH_FILTER_MIN,
                         hi_chars=mp.LENGTH_FILTER_MAX)
    sentences = load.sentences
    metadata = load.metadata or ()
    assert len(sentences) == len(metadata) == n_target, (
        f"sample/metadata length mismatch: {len(sentences)} / {len(metadata)} / {n_target}"
    )
    print(f"  source       : {load.source.name}")
    print(f"  hf id        : {load.source.hf_id}")
    print(f"  configs      : {load.source.hf_config}")
    print(f"  license      : {load.source.license}")
    print(f"  pipeline     : raw {load.raw_count} → norm {load.after_norm} → "
          f"len {load.after_length} → dedupe {load.after_dedupe} → sample {len(sentences)}")

    # Config balance sanity (must be exactly PER_CONFIG_SAMPLE per config)
    from collections import Counter
    cfg_counts = Counter(m["config"] for m in metadata)
    for cfg in mp.KORMEDMCQA_CONFIGS:
        actual = cfg_counts.get(cfg, 0)
        assert actual == mp.PER_CONFIG_SAMPLE, (
            f"config {cfg!r} has {actual} stems, expected {mp.PER_CONFIG_SAMPLE}"
        )
    print(f"  config balance: {dict(cfg_counts)}  [exact {mp.PER_CONFIG_SAMPLE} per config]")

    print("\n[2/5] Initializing tokenizers...")
    tokenizers = init_tokenizers()
    print(f"  ready        : {list(tokenizers.keys())}")

    per_text_path = out_dir / "07_tpc_medical_n1000_per_text.csv"
    writer = CheckpointWriter(
        path=per_text_path,
        fieldnames=(
            "model_label", "model_name", "model_version", "provider",
            "category", "text_id", "n_chars", "n_tokens",
            "config", "year", "period", "q_number", "question_id",
            "timestamp",
        ),
    )
    if writer.done:
        print(f"  resume       : {len(writer.done)} (model, text_id) pairs already in CSV")

    char_counts = [len(s) for s in sentences]

    print(f"\n[3/5] Measuring 7 models × {n_target} stems (checkpointed)...")
    model_to_counts: dict[str, CountSet] = {}

    for label, tok in tokenizers.items():
        info = tok.info
        already = sum(1 for (m, _) in writer.done if m == info.name)
        if already >= n_target:
            print(f"  {label:<22} skipped (all {n_target} already done)")
            with per_text_path.open(encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if r["model_name"] == info.name]
            rows.sort(key=lambda r: int(r["text_id"]))
            tokens = tuple(int(r["n_tokens"]) for r in rows)
            chars = tuple(int(r["n_chars"]) for r in rows)
            model_to_counts[label] = CountSet(tokens=tokens, chars=chars)
            continue

        t0 = time.time()
        tokens_for_this_model: list[int] = [0] * n_target
        if already:
            with per_text_path.open(encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r["model_name"] == info.name:
                        tokens_for_this_model[int(r["text_id"])] = int(r["n_tokens"])

        for i, text in enumerate(sentences):
            if (info.name, i) in writer.done:
                continue
            n_tokens = tok.count(text)
            tokens_for_this_model[i] = n_tokens
            meta = metadata[i] if i < len(metadata) else {}
            writer.write({
                "model_label": label,
                "model_name": info.name,
                "model_version": info.version,
                "provider": info.provider,
                "category": "medical",
                "text_id": i,
                "n_chars": char_counts[i],
                "n_tokens": n_tokens,
                "config": meta.get("config", ""),
                "year": meta.get("year", ""),
                "period": meta.get("period", ""),
                "q_number": meta.get("q_number", ""),
                "question_id": meta.get("question_id", ""),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            if (i + 1) % 100 == 0:
                pct = (i + 1) / n_target * 100
                elapsed = time.time() - t0
                print(f"    [{label}] {i+1}/{n_target} ({pct:.0f}%) — {elapsed:.0f}s elapsed",
                      file=sys.stderr)

        dt = time.time() - t0
        cs = CountSet(tokens=tuple(tokens_for_this_model), chars=tuple(char_counts))
        model_to_counts[label] = cs
        print(f"  {label:<22} TPC={cs.aggregate_tpc:.4f}  n={cs.n}  ({dt:.1f}s)")

    writer.close()
    print(f"  per-text     : {per_text_path}")

    print("\n[4/5] Aggregates + bootstrap CI + pairwise Wilcoxon...")

    agg_path = out_dir / "07_tpc_medical_n1000_aggregate.csv"
    results_for_print: list[tuple[str, ModelResult]] = []
    with agg_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model_label", "model_name", "version", "provider", "category",
            "n", "tokens_total", "chars_total",
            "tpc", "tpc_ci_low", "tpc_ci_high",
            "news_tpc", "delta_vs_news_pct",
            "conv_tpc", "delta_vs_conv_pct",
            "verdict",  # populated only on the EXAONE row
        ])
        for label, cs in model_to_counts.items():
            tok_obj = tokenizers[label]
            lo, hi = cs.bootstrap_tpc_ci(n_bootstrap=1000, seed=42)
            r = ModelResult(
                model_name=tok_obj.info.name,
                model_version=tok_obj.info.version,
                provider=tok_obj.info.provider,
                category="medical",
                counts=cs,
                tpc=cs.aggregate_tpc,
                tpc_ci_low=lo,
                tpc_ci_high=hi,
            )
            results_for_print.append((label, r))
            news = N1000_NEWS_TPC.get(r.model_version) or N1000_NEWS_TPC.get(r.model_name)
            conv = N1000_CONV_TPC.get(r.model_version) or N1000_CONV_TPC.get(r.model_name)
            d_news = ((r.tpc - news) / news * 100) if news else ""
            d_conv = ((r.tpc - conv) / conv * 100) if conv else ""
            w.writerow([label, r.model_name, r.model_version, r.provider, r.category,
                        r.counts.n, r.counts.total_tokens, r.counts.total_chars,
                        round(r.tpc, 4),
                        round(r.tpc_ci_low, 4), round(r.tpc_ci_high, 4),
                        news or "",
                        round(d_news, 2) if d_news != "" else "",
                        conv or "",
                        round(d_conv, 2) if d_conv != "" else "",
                        ""])  # verdict filled in [5/5] below
    print(f"  aggregate    : {agg_path}")

    pair_path = out_dir / "07_tpc_medical_n1000_pairwise.csv"
    labels = [lbl for lbl, _ in results_for_print]
    with pair_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_a", "model_b", "median_diff_tpc",
                    "wilcoxon_statistic", "pvalue"])
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a_label, a_r = results_for_print[i]
                b_label, b_r = results_for_print[j]
                stats = paired_wilcoxon_tpc(a_r.counts, b_r.counts)
                w.writerow([a_label, b_label,
                            round(stats["median_diff"], 4),
                            round(stats["statistic"], 2),
                            stats["pvalue"]])
    print(f"  pairwise     : {pair_path}")

    # ----- Headline summary (sorted by KO/GPT ascending) -----
    print("\n" + "=" * 76)
    print(f"Summary (medical n={n_target}) — sorted by KO/GPT")
    print("=" * 76)
    gpt = next((r for lbl, r in results_for_print if r.model_name == "gpt-4o"), None)
    if gpt is None:
        print("[warn] GPT-4o baseline missing — cannot compute KO/GPT.")
        return
    gpt_tpc = gpt.tpc
    sorted_results = sorted(results_for_print, key=lambda x: x[1].tpc / gpt_tpc)
    header = f"{'Model':<22} {'TPC':>7} {'CI95':>17} {'KO/GPT':>8} {'Tier':<10} {'Δ news':>8} {'Δ conv':>8}"
    print(header)
    print("-" * len(header))
    exa_kpr_gpt: float | None = None
    for label, r in sorted_results:
        ratio = r.tpc / gpt_tpc
        tier = tier_for(ratio)
        ci_str = f"[{r.tpc_ci_low:.3f},{r.tpc_ci_high:.3f}]"
        news = N1000_NEWS_TPC.get(r.model_version) or N1000_NEWS_TPC.get(r.model_name)
        conv = N1000_CONV_TPC.get(r.model_version) or N1000_CONV_TPC.get(r.model_name)
        d_news_str = _delta_pct(r.tpc, news)
        d_conv_str = _delta_pct(r.tpc, conv)
        print(f"{label:<22} {r.tpc:>7.4f} {ci_str:>17} {ratio:>7.2f}× {tier:<10} {d_news_str:>8} {d_conv_str:>8}")
        if r.model_name == "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct":
            exa_kpr_gpt = ratio

    # ----- Paper-narrative block (cluster structure + gap trajectory +
    # genre sensitivity). Composes (P1, Aux d, Aux e) into one Discussion-
    # ready block that runs *immediately before* the [5/5] verdict so the
    # full mechanism narrative lands on stdout in one pass.

    # (Reinforcement 1) — Cluster structure: count tiers in medical and
    # diff against conversation tier membership (P1).
    medical_tier_by_label: dict[str, str] = {}
    tier_counts = {"advantage": 0, "efficient": 0, "penalty": 0}
    for label, r in results_for_print:
        t = tier_for(r.tpc / gpt_tpc)
        medical_tier_by_label[label] = t
        tier_counts[t] += 1

    def _conv_tier(model_version: str, model_name: str) -> str | None:
        conv_gpt = N1000_CONV_TPC.get("gpt-4o")
        conv_self = N1000_CONV_TPC.get(model_version) or N1000_CONV_TPC.get(model_name)
        if conv_gpt is None or conv_self is None:
            return None
        return tier_for(conv_self / conv_gpt)

    cross_changes: list[str] = []
    for label, r in results_for_print:
        c_t = _conv_tier(r.model_version, r.model_name)
        m_t = medical_tier_by_label[label]
        if c_t is not None and c_t != m_t:
            cross_changes.append(f"{label}: {c_t}→{m_t}")
    print(f"\nCluster structure: {tier_counts['advantage']} advantage / "
          f"{tier_counts['efficient']} efficient / {tier_counts['penalty']} penalty")
    if not cross_changes:
        print(f"  Tier membership preserved from conv (P1-style claim holds).")
    else:
        print(f"  Tier changes vs conv: {', '.join(cross_changes)}  [P1 borderline]")

    # (Reinforcement 2) — Cluster gap with trajectory (Aux d).
    NEWS_CLUSTER_GAP = 0.36   # from notes/02 news n=1000
    CONV_CLUSTER_GAP = 0.61   # from notes/02 conversation n=1000
    non_baseline_ratios = [(lbl, r.tpc / gpt_tpc) for lbl, r in results_for_print
                           if r.model_name != "gpt-4o"]
    eff_max = max((rr for _, rr in non_baseline_ratios if rr < 1.3), default=None)
    pen_min = min((rr for _, rr in non_baseline_ratios if rr >= 1.3), default=None)
    cluster_gap = (pen_min - eff_max) if (eff_max is not None and pen_min is not None) else None
    if cluster_gap is not None:
        d_lo, d_hi = mp.CLUSTER_GAP_BAND
        d_status = "IN band" if d_lo <= cluster_gap <= d_hi else "OUT of band"
        delta_vs_conv = cluster_gap - CONV_CLUSTER_GAP
        if abs(delta_vs_conv) <= 0.05:
            traj = "stable"
        elif delta_vs_conv > 0:
            traj = "widened"
        else:
            traj = "narrowed"
        print(f"\nCluster gap: {eff_max:.2f}× → {pen_min:.2f}× = {cluster_gap:.2f}×  "
              f"(Aux (d) band [{d_lo}, {d_hi}]: {d_status})")
        print(f"  Trajectory: news {NEWS_CLUSTER_GAP:.2f}× → conv {CONV_CLUSTER_GAP:.2f}× "
              f"→ medical {cluster_gap:.2f}× ({traj})")
    else:
        print("\n[warn] Cluster gap could not be computed (one tier missing).")

    # (Reinforcement 3) — Genre sensitivity per cluster: efficient cluster
    # expected sensitive (large |Δ vs conv|), penalty cluster expected
    # insensitive (|Δ vs conv| < Aux (e) bound).
    penalty_models = ("claude-sonnet-4-5", "upstage/SOLAR-10.7B-Instruct-v1.0")
    eff_lines: list[str] = []
    pen_lines: list[str] = []
    for label, r in results_for_print:
        conv = N1000_CONV_TPC.get(r.model_version) or N1000_CONV_TPC.get(r.model_name)
        if conv is None:
            continue
        d = (r.tpc - conv) / conv * 100
        if r.model_name in penalty_models:
            ok = abs(d) < mp.PENALTY_GENRE_INSENSITIVITY_PCT
            pen_lines.append(f"  {label:<22}  Δ={d:+.1f}%   "
                             f"{'HELD (insensitive)' if ok else 'VIOLATED'}")
        else:
            eff_lines.append(f"  {label:<22}  Δ={d:+.1f}%")
    print(f"\nGenre sensitivity (medical vs conversation):")
    print(f"  Efficient cluster (genre-sensitive expected):")
    for line in eff_lines:
        print(line)
    print(f"  Penalty cluster (Aux (e): |Δ| < {mp.PENALTY_GENRE_INSENSITIVITY_PCT}% expected):")
    for line in pen_lines:
        print(line)

    # ----- Auto-verdict (the headline classifier output) -----
    print("\n[5/5] Pre-registered EXAONE verdict")
    print("-" * 76)
    verdict_path = out_dir / "07_tpc_medical_n1000_verdict.csv"
    if exa_kpr_gpt is None:
        print("[warn] EXAONE missing — cannot compute pre-registered verdict.")
        with verdict_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["exaone_kpr_gpt", "verdict", "band_a", "band_c", "band_b"])
            w.writerow(["", "EXAONE_MISSING",
                        f"{mp.HYPOTHESIS_A_ENTITY_DENSITY}",
                        f"{mp.HYPOTHESIS_C_BOTH_MECHANISMS}",
                        f"{mp.HYPOTHESIS_B_REGISTER_ONLY}"])
        return

    verdict = mp.classify_kpr_gpt(exa_kpr_gpt)
    print(f"  EXAONE KPR/GPT (medical) = {exa_kpr_gpt:.4f}")
    print(f"  Pre-registered bands:")
    print(f"    A (entity-density)        {mp.HYPOTHESIS_A_ENTITY_DENSITY}")
    print(f"    C (both mechanisms)       {mp.HYPOTHESIS_C_BOTH_MECHANISMS}")
    print(f"    B (register-only)         {mp.HYPOTHESIS_B_REGISTER_ONLY}")
    print(f"    outside-low  if < {mp.ALL_FALSIFIED_BELOW}")
    print(f"    outside-high if > {mp.ALL_FALSIFIED_ABOVE}")
    print(f"\n  >>> VERDICT: '{verdict}'")
    if verdict == "A":
        print("      Hypothesis (a) — entity-density penalty resurfaces.")
    elif verdict == "C":
        print("      Hypothesis (c) — both mechanisms operating jointly.")
    elif verdict == "B":
        print("      Hypothesis (b) — register-invariance is the only mechanism.")
    elif verdict == "between-A-and-C":
        print("      Buffer zone near (a). Discussion calls out the buffer.")
    elif verdict == "between-C-and-B":
        print("      Buffer zone near (b). Discussion calls out the buffer.")
    elif verdict in ("below-all", "above-all"):
        print("      All three pre-registered hypotheses falsified —")
        print("      paper Discussion proceeds to fourth-mechanism scenario.")
    print()

    # Update the EXAONE row in aggregate CSV with the verdict.
    with agg_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys()) if rows else []
    for row in rows:
        if row.get("model_version") == "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct":
            row["verdict"] = verdict
    with agg_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"  aggregate updated with verdict: {agg_path}")

    with verdict_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["exaone_kpr_gpt", "verdict",
                    "band_a_low", "band_a_high",
                    "band_c_low", "band_c_high",
                    "band_b_low", "band_b_high"])
        a_lo, a_hi = mp.HYPOTHESIS_A_ENTITY_DENSITY
        c_lo, c_hi = mp.HYPOTHESIS_C_BOTH_MECHANISMS
        b_lo, b_hi = mp.HYPOTHESIS_B_REGISTER_ONLY
        w.writerow([round(exa_kpr_gpt, 4), verdict,
                    a_lo, a_hi, c_lo, c_hi, b_lo, b_hi])
    print(f"  verdict CSV  : {verdict_path}")

    print("\nDone. CSVs in results/raw/.")


if __name__ == "__main__":
    main()
