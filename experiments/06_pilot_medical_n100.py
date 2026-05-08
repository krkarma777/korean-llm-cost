"""Phase 1 — KorMedMCQA pilot inspector (n=100, 25 per config).

Corpus-side verification before the n=1000 main run. Loads 100 stems
via ``load_category("medical", n=100)`` (which delegates to the
stratified per-config sampler in ``corpus_loader``) and reports:

- 4-config balance check (must be exactly 25 per config)
- Combined length distribution (mean / median / p10 / p90 / max)
- Per-config length distribution
- 8 sample stems (2 per config) for visual entity-density inspection
- Pre-registered escape-hatch check (median ≤ 50 chars → switch to
  Option B requires an amendment commit *before* the n=1000 run)
- Auxiliary prediction (f) check: median in [60, 130] and p90 ≤ 350

Does *not* run any tokenizers — pilot is corpus verification only.
Tokenizer timing estimates are reported analytically based on the
n=1000 conversation run for comparison.

This script is intentionally read-only: no CSV / figure outputs,
nothing committed. The HF dataset cache lives outside the repo
(``~/.cache/huggingface/``), so no data leaks into git.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from korean_llm_cost import medical_predictions as mp  # noqa: E402
from korean_llm_cost.corpus_loader import load_category  # noqa: E402


def main() -> None:
    print("=" * 76)
    print("KorMedMCQA pilot inspector — n=100, 25 per config")
    print(f"  source repo : {mp.KORMEDMCQA_REPO}")
    print(f"  configs     : {mp.KORMEDMCQA_CONFIGS}")
    print(f"  split       : {mp.SPLIT_TO_SAMPLE_FROM}")
    print(f"  field       : {mp.MEASUREMENT_UNIT}  (Option A — stem only)")
    print(f"  filter      : [{mp.LENGTH_FILTER_MIN}, {mp.LENGTH_FILTER_MAX}] chars")
    print(f"  seed        : {mp.SAMPLE_SEED}")
    print(f"  pre-reg     : src/korean_llm_cost/medical_predictions.py")
    print("=" * 76)

    n_pilot = 100
    expected_per_config = n_pilot // len(mp.KORMEDMCQA_CONFIGS)
    print(f"\n[1/5] Loading {n_pilot} stems ({expected_per_config} per config)...")
    r = load_category("medical", n=n_pilot, seed=mp.SAMPLE_SEED,
                      lo_chars=mp.LENGTH_FILTER_MIN,
                      hi_chars=mp.LENGTH_FILTER_MAX)
    print(f"  pipeline    : raw {r.raw_count} → norm {r.after_norm} → "
          f"len {r.after_length} → dedupe {r.after_dedupe} → sample {len(r.sentences)}")
    print(f"  license     : {r.source.license}")

    # Config balance check
    assert r.metadata is not None
    print("\n[2/5] Config balance:")
    by_cfg: dict[str, list[int]] = {}
    for s, m in zip(r.sentences, r.metadata):
        by_cfg.setdefault(m["config"], []).append(len(s))
    for cfg in mp.KORMEDMCQA_CONFIGS:
        n_cfg = len(by_cfg.get(cfg, []))
        marker = "OK" if n_cfg == expected_per_config else "FAIL"
        print(f"  {cfg:<10} n={n_cfg:>3}   [{marker}]")
    if any(len(by_cfg.get(c, [])) != expected_per_config for c in mp.KORMEDMCQA_CONFIGS):
        print("  ! Imbalance detected — pilot fails. Investigate before n=1000.")
        sys.exit(1)

    # Length distribution
    all_lens = np.array([len(s) for s in r.sentences])
    print("\n[3/5] Length distribution (chars):")
    print(f"  combined (n={len(all_lens)}): "
          f"mean={all_lens.mean():.1f}  median={int(np.median(all_lens))}  "
          f"p10={int(np.quantile(all_lens, 0.1))}  p90={int(np.quantile(all_lens, 0.9))}  "
          f"max={all_lens.max()}")
    for cfg in mp.KORMEDMCQA_CONFIGS:
        a = np.array(by_cfg[cfg])
        print(f"  {cfg:<10} (n={len(a)}): "
              f"mean={a.mean():>5.1f}  median={int(np.median(a)):>3}  "
              f"p10={int(np.quantile(a, 0.1)):>3}  p90={int(np.quantile(a, 0.9)):>3}  "
              f"max={a.max():>3}")

    # Sample stems — 2 per config
    print("\n[4/5] Sample stems (2 per config, for entity-density visual check):")
    by_cfg_text: dict[str, list[tuple[int, str, dict]]] = {}
    for s, m in zip(r.sentences, r.metadata):
        by_cfg_text.setdefault(m["config"], []).append((len(s), s, m))
    for cfg in mp.KORMEDMCQA_CONFIGS:
        # 2 from each: shortest + longest (visualizes the range)
        items = sorted(by_cfg_text[cfg], key=lambda x: x[0])
        picks = [items[0], items[-1]] if len(items) >= 2 else items[:2]
        print(f"  --- {cfg} ---")
        for n_chars, text, meta in picks:
            qid = meta["question_id"]
            preview = text if len(text) <= 200 else text[:197] + "..."
            print(f"    [{qid}] ({n_chars} chars) {preview}")

    # Escape hatch + (f) auxiliary prediction
    median = int(np.median(all_lens))
    p90 = int(np.quantile(all_lens, 0.9))
    print("\n[5/5] Pre-registered checks:")
    escape_triggered = median <= mp.ESCAPE_HATCH_MIN_MEDIAN_CHARS
    print(f"  Escape hatch (Option A → B if median ≤ {mp.ESCAPE_HATCH_MIN_MEDIAN_CHARS}):")
    print(f"    pilot median = {median}  →  {'TRIGGERED — Option B amendment commit needed' if escape_triggered else 'NOT triggered — Option A holds'}")

    f_lo, f_hi = mp.STEM_LENGTH_MEDIAN_BAND
    in_f_band = f_lo <= median <= f_hi
    p90_ok = p90 <= mp.STEM_LENGTH_P90_MAX
    print(f"  Aux prediction (f) — median in [{f_lo}, {f_hi}] and p90 ≤ {mp.STEM_LENGTH_P90_MAX}:")
    print(f"    median = {median}  →  {'IN band' if in_f_band else 'OUT of band'}")
    print(f"    p90    = {p90}  →  {'within bound' if p90_ok else 'exceeds bound'}")
    if in_f_band and p90_ok:
        print(f"    Aux (f): pilot-consistent with full-run prediction.")
    else:
        print(f"    Aux (f): pilot deviates from prediction — check whether full n=1000 will too.")

    # Cross-corpus length comparison
    print("\nCross-corpus length comparison:")
    print(f"  news n=1000 (KLUE-YNAT)       : median≈22, p90≈37   (short headlines)")
    print(f"  conversation n=1000 (AI Hub)  : median≈20, p90≈36   (utterances)")
    print(f"  medical pilot n=100           : median={median}, p90={p90}   (multi-paragraph clinical vignettes)")
    print(f"  → medical stems are roughly {median/22:.0f}× the typical news/conversation length.")

    # Timing estimate (analytic, based on conversation n=1000 run)
    print("\nTiming estimate for full n=1000 measurement (per provider):")
    print(f"  HF tokenizers (Solar/EXAONE/Qwen/Llama): local, ~10ms/call ≈ 10s/model")
    print(f"  Anthropic count_tokens API: ~100-200ms/call ≈ 2-3 min")
    print(f"  Google count_tokens API: ~50-100ms/call ≈ 1-2 min")
    print(f"  OpenAI tiktoken: offline, ~1ms/call ≈ 1s")
    print(f"  Total wall time for n=1000 × 7 models ≈ 5-10 minutes (per the n=1000 conversation run).")
    print(f"  Medical stems are longer but per-call time is dominated by network latency, not size,")
    print(f"  so the projection is comparable to the conversation run.")


if __name__ == "__main__":
    main()
