"""Phase 1 — n=1000 AI Hub conversation measurement.

Mirror of ``02_tpc_news_n1000.py`` adapted for the conversation
category (AI Hub Topic-based Korean Daily Conversation, validation
split, media-balanced 200×5 sample).

Differences from the news run:

- Loader is ``corpus_loader.load_category("conversation", ...)``,
  which performs in-place zip reading and media-balanced sampling.
- Per-text CSV carries five extra columns from the AI Hub metadata:
  ``subject``, ``speaker_sex``, ``speaker_age``, ``media``,
  ``session_id``. These enable the P7 subject-level sub-analysis
  and the P8 per-media check from ``notes/10_predictions_conversation.md``.
- Output filenames use the ``03_tpc_conversation_n1000_*`` prefix.

Outputs
-------
- ``results/raw/03_tpc_conversation_n1000_per_text.csv``
- ``results/raw/03_tpc_conversation_n1000_aggregate.csv``
- ``results/raw/03_tpc_conversation_n1000_pairwise.csv``
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


# ----- News n=1000 baseline (for Δ-from-news tracking, P3 + P5) -----
N1000_NEWS_TPC = {
    "gpt-4o": 0.7226,
    "claude-sonnet-4-5": 1.0831,
    "gemini-2.5-flash": 0.6961,
    "upstage/SOLAR-10.7B-Instruct-v1.0": 1.2055,
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct": 0.5468,
    "Qwen/Qwen2.5-7B-Instruct": 0.8226,
    "meta-llama/Llama-3.1-8B-Instruct": 0.7325,
}


# ----- .env loader (same as the news run) -----

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
    """Same tokenizer set as 02_tpc_news_n1000."""
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


# ----- Checkpointed CSV writer (same shape as the news script,
#       with extra metadata columns for conversation) -----

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


def delta_from_news(now_tpc: float, news_tpc: float | None) -> str:
    if news_tpc is None:
        return "(no news baseline)"
    delta = (now_tpc - news_tpc) / news_tpc * 100
    return f"{delta:+.1f}% vs news"


# ----- Main -----

def main() -> None:
    _load_env_file(ROOT / ".env")
    n_target = 1000
    out_dir = ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f"Phase 1 — conversation main run: n={n_target}")
    print("       AI Hub Topic-based Korean Daily Conversation, validation, 5 messengers × 200")
    print("=" * 76)

    print("\n[1/4] Loading corpus...")
    load = load_category("conversation", n=n_target, seed=42)
    sentences = load.sentences
    metadata = load.metadata or ()
    assert len(sentences) == len(metadata) == n_target, (
        f"sample/metadata length mismatch: {len(sentences)} / {len(metadata)} / {n_target}"
    )
    print(f"  source       : {load.source.name}")
    print(f"  local path   : {load.source.hf_config}")
    print(f"  license      : {load.source.license}")
    print(f"  pipeline     : raw {load.raw_count} → norm {load.after_norm} → "
          f"len {load.after_length} → dedupe {load.after_dedupe} → sample {len(sentences)}")

    print("\n[2/4] Initializing tokenizers...")
    tokenizers = init_tokenizers()
    print(f"  ready        : {list(tokenizers.keys())}")

    per_text_path = out_dir / "03_tpc_conversation_n1000_per_text.csv"
    writer = CheckpointWriter(
        path=per_text_path,
        fieldnames=(
            "model_label", "model_name", "model_version", "provider",
            "category", "text_id", "n_chars", "n_tokens",
            "subject", "speaker_sex", "speaker_age", "media", "session_id",
            "timestamp",
        ),
    )
    if writer.done:
        print(f"  resume       : {len(writer.done)} (model, text_id) pairs already in CSV")

    char_counts = [len(s) for s in sentences]

    print(f"\n[3/4] Measuring 7 models × {n_target} utterances (checkpointed)...")
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
                "category": "conversation",
                "text_id": i,
                "n_chars": char_counts[i],
                "n_tokens": n_tokens,
                "subject": meta.get("subject", ""),
                "speaker_sex": meta.get("speaker_sex", ""),
                "speaker_age": meta.get("speaker_age", ""),
                "media": meta.get("media", ""),
                "session_id": meta.get("session_id", ""),
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

    print("\n[4/4] Aggregates + bootstrap CI + pairwise Wilcoxon...")

    agg_path = out_dir / "03_tpc_conversation_n1000_aggregate.csv"
    results_for_print: list[tuple[str, ModelResult]] = []
    with agg_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model_label", "model_name", "version", "provider", "category",
            "n", "tokens_total", "chars_total",
            "tpc", "tpc_ci_low", "tpc_ci_high",
            "news_tpc", "delta_vs_news_pct",
        ])
        for label, cs in model_to_counts.items():
            tok_obj = tokenizers[label]
            lo, hi = cs.bootstrap_tpc_ci(n_bootstrap=1000, seed=42)
            r = ModelResult(
                model_name=tok_obj.info.name,
                model_version=tok_obj.info.version,
                provider=tok_obj.info.provider,
                category="conversation",
                counts=cs,
                tpc=cs.aggregate_tpc,
                tpc_ci_low=lo,
                tpc_ci_high=hi,
            )
            results_for_print.append((label, r))
            news = N1000_NEWS_TPC.get(r.model_version) or N1000_NEWS_TPC.get(r.model_name)
            delta_pct = ((r.tpc - news) / news * 100) if news else ""
            w.writerow([label, r.model_name, r.model_version, r.provider, r.category,
                        r.counts.n, r.counts.total_tokens, r.counts.total_chars,
                        round(r.tpc, 4),
                        round(r.tpc_ci_low, 4), round(r.tpc_ci_high, 4),
                        news or "",
                        round(delta_pct, 2) if delta_pct != "" else ""])
    print(f"  aggregate    : {agg_path}")

    pair_path = out_dir / "03_tpc_conversation_n1000_pairwise.csv"
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

    # ----- Summary print -----
    print("\n" + "=" * 76)
    print(f"Summary (conversation n={n_target}) — sorted by KO/GPT")
    print("=" * 76)
    gpt = next((r for lbl, r in results_for_print if r.model_name == "gpt-4o"), None)
    if gpt is None:
        print("[warn] GPT-4o baseline missing — cannot compute KO/GPT.")
        return
    gpt_tpc = gpt.tpc
    sorted_results = sorted(results_for_print, key=lambda x: x[1].tpc / gpt_tpc)
    header = f"{'Model':<22} {'TPC':>7} {'CI95':>17} {'KO/GPT':>8} {'Tier':<10} {'Δ vs news':<14}"
    print(header)
    print("-" * len(header))
    for label, r in sorted_results:
        ratio = r.tpc / gpt_tpc
        tier = tier_for(ratio)
        ci_str = f"[{r.tpc_ci_low:.3f},{r.tpc_ci_high:.3f}]"
        news_tpc = N1000_NEWS_TPC.get(r.model_version) or N1000_NEWS_TPC.get(r.model_name)
        v = delta_from_news(r.tpc, news_tpc)
        print(f"{label:<22} {r.tpc:>7.4f} {ci_str:>17} {ratio:>7.2f}× {tier:<10} {v:<14}")

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

    # Demographic + media check (P6, P8) — quick sanity, not full sub-analysis.
    if metadata:
        from collections import Counter
        sex_ct = Counter(m.get("speaker_sex", "") for m in metadata)
        age_ct = Counter(m.get("speaker_age", "") for m in metadata)
        media_ct = Counter(m.get("media", "") for m in metadata)
        n = len(metadata)
        print(f"\nDemographic (P6 check):")
        print(f"  sex   : {dict(sex_ct.most_common())}")
        print(f"  age   : {dict(age_ct.most_common())}")
        female_pct = sex_ct.get("여성", 0) / n * 100
        twenties_pct = sex_ct.get("20대", 0) / n * 100  # this is wrong key; use age dict
        twenties_pct = age_ct.get("20대", 0) / n * 100
        print(f"  female-share : {female_pct:.1f}%  (P6 expects 70–85%)")
        print(f"  20s-share    : {twenties_pct:.1f}%  (P6 expects 70–85%)")
        print(f"\nMedia balance (P8 sanity — full check via per-text CSV):")
        for media, ct in sorted(media_ct.items()):
            print(f"  {media:<12}: {ct}")

    print("\nDone. CSVs in results/raw/.")


if __name__ == "__main__":
    main()
