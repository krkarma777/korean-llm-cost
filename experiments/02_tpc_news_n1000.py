"""Phase 1 — n=1000 KLUE-YNAT measurement (the real run).

Differences from ``01_tpc_news_pilot.py``:

- 1000 sentences instead of 100 (10× the data).
- Per-text rows are written to CSV *as we measure* (append mode + flush
  after each row), so an interrupted run can be resumed without re-doing
  any (model, text_id) pair.
- Resume detection: at start, scans the existing per-text CSV (if any)
  for completed (model_name, text_id) pairs and skips them.
- Reuses the rate-limit retry baked into anthropic_tok / google_tok at
  the wrapper level — no script-level retry needed.

Outputs
-------
- ``results/raw/02_tpc_news_n1000_per_text.csv``   — every (model, text_id)
- ``results/raw/02_tpc_news_n1000_aggregate.csv``  — per-model aggregates with bootstrap CI
- ``results/raw/02_tpc_news_n1000_pairwise.csv``   — per-pair Wilcoxon stats

Run
---
    uv run python experiments/02_tpc_news_n1000.py
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

from korean_llm_cost.corpus_loader import load_category
from korean_llm_cost.metrics import (
    CountSet,
    ModelResult,
    paired_wilcoxon_tpc,
)


# ----- n=10 sanity baseline (for Δ tracking) -----
N10_TPC = {
    "gpt-4o": 0.664,
    "claude-sonnet-4-5": 1.074,
    "gemini-2.5-flash": 0.648,
    "upstage/SOLAR-10.7B-Instruct-v1.0": 1.139,
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct": 0.566,
    "Qwen/Qwen2.5-7B-Instruct": 0.746,
    "meta-llama/Llama-3.1-8B-Instruct": 0.684,
}


# ----- .env loader (same as pilot, source of truth = .env) -----

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


# ----- Checkpointed per-text writer -----

@dataclass
class CheckpointWriter:
    """Append-mode CSV writer with resume support.

    On open, reads the existing file (if any) and exposes
    ``done`` — the set of (model_name, text_id) pairs already recorded.
    Subsequent ``write`` calls are skipped if the pair is in ``done``.
    Each write flushes to disk so interruption never loses data.
    """

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
        # Open in append mode; write header only if file is brand new.
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


# ----- Verdict helpers (same as pilot) -----

def tier_for(kpr_gpt: float) -> str:
    if kpr_gpt <= 1.0 / 1.3:
        return "advantage"
    if kpr_gpt < 1.3:
        return "efficient"
    return "penalty"


def consistency_verdict(now_tpc: float, n10_tpc: float | None) -> str:
    if n10_tpc is None:
        return "(no n=10 baseline)"
    delta = (now_tpc - n10_tpc) / n10_tpc * 100
    return f"{delta:+.1f}% vs n=10"


# ----- Main -----

def main() -> None:
    _load_env_file(ROOT / ".env")
    n_target = 1000
    out_dir = ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Phase 1 main run: n={n_target} on KLUE-YNAT (news)")
    print("=" * 72)

    print("\n[1/4] Loading corpus...")
    load = load_category("news", n=n_target, seed=42)
    sentences = load.sentences
    print(f"  source   : {load.source.name} ({load.source.hf_id}/{load.source.hf_config})")
    print(f"  pipeline : raw {load.raw_count} → norm {load.after_norm} → "
          f"len {load.after_length} → dedupe {load.after_dedupe} → sample {len(sentences)}")

    print("\n[2/4] Initializing tokenizers...")
    tokenizers = init_tokenizers()
    print(f"  ready    : {list(tokenizers.keys())}")

    # Set up checkpointed writer
    per_text_path = out_dir / "02_tpc_news_n1000_per_text.csv"
    writer = CheckpointWriter(
        path=per_text_path,
        fieldnames=("model_label", "model_name", "model_version", "provider",
                    "category", "text_id", "n_chars", "n_tokens", "timestamp"),
    )
    if writer.done:
        print(f"  resume   : {len(writer.done)} (model, text_id) pairs already in CSV")

    # Build char_counts once (deterministic for the corpus)
    char_counts = [len(s) for s in sentences]

    print(f"\n[3/4] Measuring 7 models × {n_target} sentences (checkpointed)...")
    # Per-model in-memory tokens dict for aggregation/CI/Wilcoxon
    model_to_counts: dict[str, CountSet] = {}

    for label, tok in tokenizers.items():
        info = tok.info
        # Skip already-completed model entirely if all text_ids are done
        already_done_for_model = sum(
            1 for (m, _) in writer.done if m == info.name
        )
        if already_done_for_model >= n_target:
            print(f"  {label:<22} skipped (all {n_target} already done)")
            # Re-load from CSV to populate model_to_counts
            with per_text_path.open(encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if r["model_name"] == info.name]
            rows.sort(key=lambda r: int(r["text_id"]))
            tokens = tuple(int(r["n_tokens"]) for r in rows)
            chars = tuple(int(r["n_chars"]) for r in rows)
            model_to_counts[label] = CountSet(tokens=tokens, chars=chars)
            continue

        t0 = time.time()
        tokens_for_this_model: list[int] = [0] * n_target
        # Pre-fill from any partially-done state
        existing_for_model = {
            int(tid): None
            for (m, tid) in writer.done if m == info.name
        }
        if existing_for_model:
            with per_text_path.open(encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r["model_name"] == info.name:
                        tokens_for_this_model[int(r["text_id"])] = int(r["n_tokens"])

        for i, text in enumerate(sentences):
            if (info.name, i) in writer.done:
                continue
            n_tokens = tok.count(text)
            tokens_for_this_model[i] = n_tokens
            writer.write({
                "model_label": label,
                "model_name": info.name,
                "model_version": info.version,
                "provider": info.provider,
                "category": "news",
                "text_id": i,
                "n_chars": char_counts[i],
                "n_tokens": n_tokens,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            # Lightweight progress every 100
            if (i + 1) % 100 == 0:
                pct = (i + 1) / n_target * 100
                elapsed = time.time() - t0
                print(f"    [{label}] {i+1}/{n_target} ({pct:.0f}%) — {elapsed:.0f}s elapsed",
                      file=sys.stderr)

        dt = time.time() - t0
        cs = CountSet(tokens=tuple(tokens_for_this_model), chars=tuple(char_counts))
        model_to_counts[label] = cs
        print(f"  {label:<22} TPC={cs.aggregate_tpc:.4f}  "
              f"n={cs.n}  ({dt:.1f}s)")

    writer.close()
    print(f"  per-text : {per_text_path}")

    print("\n[4/4] Aggregates + bootstrap CI + pairwise Wilcoxon...")

    # --- Aggregate CSV ---
    agg_path = out_dir / "02_tpc_news_n1000_aggregate.csv"
    results_for_print: list[tuple[str, ModelResult]] = []
    with agg_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_label", "model_name", "version", "provider", "category",
                    "n", "tokens_total", "chars_total",
                    "tpc", "tpc_ci_low", "tpc_ci_high",
                    "n10_tpc", "delta_vs_n10_pct"])
        for label, cs in model_to_counts.items():
            tok_obj = tokenizers[label]
            lo, hi = cs.bootstrap_tpc_ci(n_bootstrap=1000, seed=42)
            r = ModelResult(
                model_name=tok_obj.info.name,
                model_version=tok_obj.info.version,
                provider=tok_obj.info.provider,
                category="news",
                counts=cs,
                tpc=cs.aggregate_tpc,
                tpc_ci_low=lo,
                tpc_ci_high=hi,
            )
            results_for_print.append((label, r))
            n10 = N10_TPC.get(r.model_version) or N10_TPC.get(r.model_name)
            delta_pct = ((r.tpc - n10) / n10 * 100) if n10 else ""
            w.writerow([label, r.model_name, r.model_version, r.provider, r.category,
                        r.counts.n, r.counts.total_tokens, r.counts.total_chars,
                        round(r.tpc, 4),
                        round(r.tpc_ci_low, 4), round(r.tpc_ci_high, 4),
                        n10 or "",
                        round(delta_pct, 2) if delta_pct != "" else ""])
    print(f"  aggregate: {agg_path}")

    # --- Pairwise Wilcoxon ---
    pair_path = out_dir / "02_tpc_news_n1000_pairwise.csv"
    labels = [lbl for lbl, _ in results_for_print]
    with pair_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_a", "model_b", "median_diff_tpc", "wilcoxon_statistic", "pvalue"])
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a_label, a_r = results_for_print[i]
                b_label, b_r = results_for_print[j]
                stats = paired_wilcoxon_tpc(a_r.counts, b_r.counts)
                w.writerow([a_label, b_label,
                            round(stats["median_diff"], 4),
                            round(stats["statistic"], 2),
                            stats["pvalue"]])
    print(f"  pairwise : {pair_path}")

    # --- Summary print ---
    print("\n" + "=" * 76)
    print(f"Summary (n={n_target}) — sorted by KO/GPT")
    print("=" * 76)
    gpt = next((r for lbl, r in results_for_print if r.model_name == "gpt-4o"), None)
    if gpt is None:
        print("[warn] GPT-4o baseline missing from this run.")
        return
    gpt_tpc = gpt.tpc
    sorted_results = sorted(results_for_print, key=lambda x: x[1].tpc / gpt_tpc)
    header = f"{'Model':<22} {'TPC':>7} {'CI95':>17} {'KO/GPT':>8} {'Tier':<10} {'Δ vs n=10':<14}"
    print(header)
    print("-" * len(header))
    for label, r in sorted_results:
        ratio = r.tpc / gpt_tpc
        tier = tier_for(ratio)
        ci_str = f"[{r.tpc_ci_low:.3f},{r.tpc_ci_high:.3f}]"
        n10 = N10_TPC.get(r.model_version) or N10_TPC.get(r.model_name)
        v = consistency_verdict(r.tpc, n10)
        print(f"{label:<22} {r.tpc:>7.4f} {ci_str:>17} {ratio:>7.2f}× {tier:<10} {v:<14}")

    # Tier counts and gap
    non_baseline = [r for _, r in results_for_print if r.model_name != "gpt-4o"]
    n_pen = sum(1 for r in non_baseline if r.tpc / gpt_tpc >= 1.3)
    n_adv = sum(1 for r in non_baseline if r.tpc / gpt_tpc <= 1 / 1.3)
    n_eff = len(non_baseline) - n_pen - n_adv
    print(f"\nTier distribution: {n_adv} advantage / {n_eff} efficient / {n_pen} penalty")
    eff_max = max(r.tpc / gpt_tpc for _, r in results_for_print if r.tpc / gpt_tpc < 1.3)
    pen_min = min(
        (r.tpc / gpt_tpc for _, r in results_for_print if r.tpc / gpt_tpc >= 1.3),
        default=None,
    )
    if pen_min is not None:
        print(f"Cluster gap: {eff_max:.2f}× → {pen_min:.2f}× = {pen_min - eff_max:.2f}×")

    print("\nDone. CSV outputs in results/raw/.")


if __name__ == "__main__":
    main()
