"""Phase 1 pilot — 100-sentence × 7-model TPC measurement on KLUE-YNAT.

Purpose
-------
Verify that the corpus → metrics pipeline scales correctly from the
n=10 sanity sample to n=100, and that the cluster split (efficient vs
penalty) reproduces at this larger sample. If the per-model TPC values
change by more than ~10% from the n=10 baseline, investigate before
committing to a 1000-sentence full run.

Outputs
-------
- ``results/raw/01_tpc_news_pilot_per_text.csv`` — every (model, text_id)
  count row. ~700 rows = 7 models × 100 texts.
- ``results/raw/01_tpc_news_pilot_aggregate.csv`` — per-model aggregate
  TPC + bootstrap 95% CI + delta vs n=10 sanity.
- stdout — formatted summary table + consistency verdict.

Run
---
    export $(grep -v '^#' .env | xargs)   # ANTHROPIC_API_KEY, GOOGLE_API_KEY, HF_TOKEN
    uv run python experiments/01_tpc_news_pilot.py
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

# Allow running as a script from the repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from korean_llm_cost.corpus_loader import load_category
from korean_llm_cost.metrics import ModelResult, measure_with_ci


# ----- n=10 sanity baseline (from notebooks/01_sanity_check) -----
# Per-model Korean TPC observed at n=10. Used for consistency check.
N10_TPC = {
    "gpt-4o": 0.664,
    "claude-sonnet-4-5": 1.074,
    "gemini-2.5-flash": 0.648,
    "upstage/SOLAR-10.7B-Instruct-v1.0": 1.139,
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct": 0.566,
    "Qwen/Qwen2.5-7B-Instruct": 0.746,
    "meta-llama/Llama-3.1-8B-Instruct": 0.684,
}


# ----- .env loader (no extra dep) -----

def _load_env_file(path: Path) -> None:
    """Load KEY=VAL lines from .env into os.environ.

    .env is the source of truth — overwrites any existing empty/stale value
    in os.environ. Non-empty existing values are preserved (lets the user
    override .env via shell export if they want).
    """
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
        existing = os.environ.get(k, "")
        if not existing:  # missing or empty → load from .env
            os.environ[k] = v


# ----- Tokenizer factory (skips providers without env vars) -----

def init_tokenizers() -> dict:
    """Return a dict of available tokenizers, one per provider+model."""
    toks: dict[str, object] = {}

    # OpenAI: offline, no key needed.
    from korean_llm_cost.tokenizers.openai_tok import OpenAITokenizer
    toks["GPT-4o"] = OpenAITokenizer("gpt-4o")

    # Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        from korean_llm_cost.tokenizers.anthropic_tok import AnthropicTokenizer
        toks["Claude Sonnet 4.5"] = AnthropicTokenizer("claude-sonnet-4-5")
    else:
        print("[skip] ANTHROPIC_API_KEY not set; Claude omitted from this run.")

    # Google
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        from korean_llm_cost.tokenizers.google_tok import GoogleTokenizer
        toks["Gemini 2.5 Flash"] = GoogleTokenizer("gemini-2.5-flash")
    else:
        print("[skip] GOOGLE_API_KEY not set; Gemini omitted.")

    # HF (open models — no token needed)
    from korean_llm_cost.tokenizers.hf_tok import HFTokenizer
    toks["Solar 10.7B"] = HFTokenizer("upstage/SOLAR-10.7B-Instruct-v1.0")
    toks["EXAONE 3.5 7.8B"] = HFTokenizer(
        "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct", trust_remote_code=True
    )
    toks["Qwen 2.5 7B"] = HFTokenizer("Qwen/Qwen2.5-7B-Instruct")

    # HF (gated — needs HF_TOKEN)
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        toks["Llama 3.1 8B"] = HFTokenizer("meta-llama/Llama-3.1-8B-Instruct")
    else:
        print("[skip] HF_TOKEN not set; Llama 3.1 omitted.")

    return toks


# ----- Verdict helpers -----

def tier_for(kpr_gpt: float) -> str:
    if kpr_gpt <= 1.0 / 1.3:
        return "advantage"
    if kpr_gpt < 1.3:
        return "efficient"
    return "penalty"


def consistency_verdict(now_tpc: float, n10_tpc: float | None) -> str:
    if n10_tpc is None:
        return "(no n=10 baseline)"
    delta = (now_tpc - n10_tpc) / n10_tpc
    sign = "+" if delta >= 0 else ""
    if abs(delta) < 0.05:
        return f"{sign}{delta*100:+.1f}% vs n=10 — consistent"
    if abs(delta) < 0.10:
        return f"{sign}{delta*100:+.1f}% vs n=10 — within tolerance"
    return f"{sign}{delta*100:+.1f}% vs n=10 — investigate"


# ----- Main -----

def main() -> None:
    _load_env_file(ROOT / ".env")

    print("=" * 72)
    print("Phase 1 pilot: 100 sentences × 7 models on KLUE-YNAT (news)")
    print("=" * 72)

    print("\n[1/4] Loading corpus...")
    load = load_category("news", n=100, seed=42)
    sentences = load.sentences
    print(f"  source   : {load.source.name} ({load.source.hf_id}/{load.source.hf_config})")
    print(f"  license  : {load.source.license}")
    print(f"  pipeline : raw {load.raw_count} → norm {load.after_norm} → "
          f"len-filter {load.after_length} → dedupe {load.after_dedupe} → sample {len(sentences)}")

    print("\n[2/4] Initializing tokenizers...")
    tokenizers = init_tokenizers()
    print(f"  ready    : {list(tokenizers.keys())}")

    print("\n[3/4] Measuring (this can take 30s–2min for API providers)...")
    results: list[ModelResult] = []
    for label, tok in tokenizers.items():
        t0 = time.time()
        r = measure_with_ci(tok, sentences, category="news", n_bootstrap=1000, seed=42)
        dt = time.time() - t0
        print(f"  {label:<22} TPC={r.tpc:.4f}  "
              f"CI=[{r.tpc_ci_low:.4f}, {r.tpc_ci_high:.4f}]  ({dt:.1f}s)")
        results.append(r)

    print("\n[4/4] Writing CSV outputs + summary...")
    out_dir = ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-text rows
    per_text_path = out_dir / "01_tpc_news_pilot_per_text.csv"
    with per_text_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "version", "provider", "category", "text_id", "n_chars", "n_tokens"])
        for r in results:
            for i, (n_t, n_c) in enumerate(zip(r.counts.tokens, r.counts.chars)):
                w.writerow([r.model_name, r.model_version, r.provider, r.category, i, n_c, n_t])
    print(f"  wrote    : {per_text_path}  ({sum(r.counts.n for r in results)} rows)")

    # Aggregate rows
    agg_path = out_dir / "01_tpc_news_pilot_aggregate.csv"
    with agg_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "version", "provider", "category", "n",
                    "tokens_total", "chars_total", "tpc",
                    "tpc_ci_low", "tpc_ci_high",
                    "n10_tpc", "delta_vs_n10_pct"])
        for r in results:
            n10 = N10_TPC.get(r.model_version) or N10_TPC.get(r.model_name)
            delta_pct = ((r.tpc - n10) / n10 * 100) if n10 else ""
            w.writerow([r.model_name, r.model_version, r.provider, r.category,
                        r.counts.n, r.counts.total_tokens, r.counts.total_chars,
                        round(r.tpc, 4),
                        round(r.tpc_ci_low, 4), round(r.tpc_ci_high, 4),
                        n10 or "",
                        round(delta_pct, 2) if delta_pct != "" else ""])
    print(f"  wrote    : {agg_path}")

    # Summary table
    print("\n" + "=" * 72)
    print("Summary — sorted by KPR-equivalent (KO TPC / GPT-4o TPC)")
    print("=" * 72)
    gpt = next((r for r in results if r.model_name == "gpt-4o"), None)
    if gpt is None:
        raise RuntimeError("GPT-4o baseline missing — cannot compute KO/GPT.")
    gpt_tpc = gpt.tpc

    # Sort by ko_ratio (kpr/gpt proxy since en TPC not measured here).
    results_sorted = sorted(results, key=lambda r: r.tpc / gpt_tpc)

    header = f"{'Model':<22} {'TPC':>7} {'CI95':>17} {'KO/GPT':>8} {'Tier':<10} {'Δ vs n=10':<22}"
    print(header)
    print("-" * len(header))
    for r in results_sorted:
        ratio = r.tpc / gpt_tpc
        tier = tier_for(ratio)
        ci_str = f"[{r.tpc_ci_low:.3f},{r.tpc_ci_high:.3f}]"
        n10 = N10_TPC.get(r.model_version) or N10_TPC.get(r.model_name)
        verdict = consistency_verdict(r.tpc, n10)
        label_lookup = {
            "gpt-4o": "GPT-4o",
            "claude-sonnet-4-5": "Claude Sonnet 4.5",
            "gemini-2.5-flash": "Gemini 2.5 Flash",
            "upstage/SOLAR-10.7B-Instruct-v1.0": "Solar 10.7B",
            "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct": "EXAONE 3.5 7.8B",
            "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5 7B",
            "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B",
        }
        label = label_lookup.get(r.model_name, r.model_name)
        print(f"{label:<22} {r.tpc:>7.4f} {ci_str:>17} {ratio:>7.2f}× {tier:<10} {verdict:<22}")

    # Tier counts
    non_baseline = [r for r in results if r.model_name != "gpt-4o"]
    n_pen = sum(1 for r in non_baseline if r.tpc / gpt_tpc >= 1.3)
    n_adv = sum(1 for r in non_baseline if r.tpc / gpt_tpc <= 1 / 1.3)
    n_eff = len(non_baseline) - n_pen - n_adv
    print()
    print(f"Tier distribution (excl. baseline): {n_adv} advantage / {n_eff} efficient / {n_pen} penalty")

    # Cluster gap
    eff_max_ratio = max(r.tpc / gpt_tpc for r in results if r.tpc / gpt_tpc < 1.3)
    pen_min_ratio = min((r.tpc / gpt_tpc for r in results if r.tpc / gpt_tpc >= 1.3), default=None)
    if pen_min_ratio is not None:
        gap = pen_min_ratio - eff_max_ratio
        print(f"Cluster gap (efficient max → penalty min): {eff_max_ratio:.2f}× → {pen_min_ratio:.2f}× = {gap:.2f}×")

    print("\nDone. Inspect results/raw/01_tpc_news_pilot_aggregate.csv for full numbers.")


if __name__ == "__main__":
    main()
