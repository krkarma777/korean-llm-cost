"""Phase 1 corpus loader — Korean text by category.

Each category maps to one or more data sources. Sources can be public
HuggingFace datasets (no auth) or local files (e.g. AI Hub downloads
that the user has manually fetched and placed under ``data/corpora/``).

The ingest pipeline is identical regardless of source:

    raw → NFC-normalize → length-filter → exact-string dedupe → seeded sample

Categories planned for Phase 1
------------------------------
- ``"news"`` — currently from KLUE-YNAT (Yonhap News Agency topic headlines).
  Public HuggingFace dataset, CC-BY-SA 4.0. No auth required.
- ``"conversational"`` — pending. AI Hub 일상대화 corpus is the paper-quality
  source but requires a Korean academic affiliation registration on
  aihub.or.kr. Once downloaded, drop the JSON files under
  ``data/corpora/aihub_dialogue/`` and we'll wire the loader.
- ``"medical"`` — pending. Candidates: KorMedMCQA (HF), AI Hub medical text.

Why we start with news rather than conversational
--------------------------------------------------
The original Phase 1 plan called for conversational first. KLUE-YNAT (news)
is the only Phase 1 candidate that can be loaded without manual
registration, so we verify the loader infrastructure on news, then circle
back to conversational once AI Hub access is in place. The category
ordering does not affect Phase 1's analysis — all 3 categories will be
measured before any paper number is computed.
"""

from __future__ import annotations

import random
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

# The `datasets` import is intentionally inside the loader functions, not at
# module top, so importing this module never fails for environments that
# only need the future `data/corpora/` local-file path.


# ----- Source descriptors -----


@dataclass(frozen=True)
class CorpusSource:
    """Metadata for one data source. Recorded in results for reproducibility.

    The fields mirror what the paper's Methods section needs to cite,
    so we don't have to chase down provenance later.
    """

    name: str               # e.g., "KLUE-YNAT"
    hf_id: str | None       # HuggingFace dataset repo, or None if local
    hf_config: str | None   # config name within the repo (e.g., "ynat" inside "klue")
    license: str            # e.g., "CC-BY-SA 4.0"
    genre: str              # "news", "conversational", "medical", ...
    notes: str = ""         # any caveat worth recording


# ----- Per-source loaders (return list of raw strings before pipeline) -----


def _load_klue_ynat() -> tuple[CorpusSource, list[str]]:
    """KLUE-YNAT — Yonhap News Agency topic classification headlines.

    Schema (HuggingFace ``klue/ynat`` config):
      - ``title``: the news headline (Korean)
      - ``label``: integer category 0..6 (we don't use it for TPC)

    We use the ``train`` split because it has the largest pool (~45K
    headlines) — plenty of room for sampling 100 / 1000.
    """
    from datasets import load_dataset

    src = CorpusSource(
        name="KLUE-YNAT",
        hf_id="klue",
        hf_config="ynat",
        license="CC-BY-SA 4.0",
        genre="news",
        notes="Yonhap News Agency headlines. Source for Phase 1 'news' category.",
    )
    ds = load_dataset(src.hf_id, src.hf_config, split="train", trust_remote_code=False)
    return src, [item["title"] for item in ds]


# Registry: category → list of (loader_callable,) so we can have multiple
# sources per category later without changing call sites.
_SOURCES: dict[str, list[Callable[[], tuple[CorpusSource, list[str]]]]] = {
    "news": [_load_klue_ynat],
    # "conversational": [<AI Hub 일상대화 loader, pending>],
    # "medical":        [<KorMedMCQA loader, pending>],
}


# ----- Pipeline -----


def _normalize(text: str) -> str:
    """NFC normalization + strip. macOS file inputs often arrive in NFD."""
    return unicodedata.normalize("NFC", text).strip()


def _length_ok(text: str, *, lo: int = 5, hi: int = 500) -> bool:
    """5 ≤ length ≤ 500 chars. Filters out '.' headlines and concat dumps."""
    return lo <= len(text) <= hi


@dataclass(frozen=True)
class LoadResult:
    """What a load_category call returns. Carries provenance for reporting."""

    source: CorpusSource
    sentences: list[str]
    raw_count: int          # before any filtering
    after_norm: int         # after NFC + strip
    after_length: int       # after length filter
    after_dedupe: int       # after exact-string dedupe
    seed: int


def load_category(
    category: str,
    n: int = 100,
    *,
    seed: int = 42,
    lo_chars: int = 5,
    hi_chars: int = 500,
) -> LoadResult:
    """Load ``n`` deduplicated Korean sentences for ``category``.

    Parameters
    ----------
    category : ``"news"`` (others pending)
    n        : how many sentences to return (raises if fewer survive filter)
    seed     : deterministic sampling seed
    lo_chars : minimum char length kept (default 5 — drops 'ㅋ' / single chars)
    hi_chars : maximum char length kept (default 500 — drops concatenations)

    Returns
    -------
    LoadResult with sentences + per-stage counts (so the caller can sanity
    check how many rows survived each filter).
    """
    if category not in _SOURCES:
        available = ", ".join(_SOURCES) or "(none yet)"
        raise ValueError(
            f"Category {category!r} is not wired. Available: {available}. "
            "See docstring for category roadmap."
        )

    # Single source for now; multi-source merging is a future concern.
    [loader] = _SOURCES[category]
    source, raw_texts = loader()

    raw_count = len(raw_texts)
    normed = [_normalize(t) for t in raw_texts]
    normed = [t for t in normed if t]  # drop empties from strip
    after_norm = len(normed)

    filtered = [t for t in normed if _length_ok(t, lo=lo_chars, hi=hi_chars)]
    after_length = len(filtered)

    seen: set[str] = set()
    deduped: list[str] = []
    for t in filtered:
        if t in seen:
            continue
        seen.add(t)
        deduped.append(t)
    after_dedupe = len(deduped)

    if after_dedupe < n:
        raise RuntimeError(
            f"Only {after_dedupe} sentences survived filtering for "
            f"category {category!r} (asked for {n}). Loosen filters or add "
            f"another source."
        )

    rng = random.Random(seed)
    sample = rng.sample(deduped, n)

    return LoadResult(
        source=source,
        sentences=sample,
        raw_count=raw_count,
        after_norm=after_norm,
        after_length=after_length,
        after_dedupe=after_dedupe,
        seed=seed,
    )


# ----- Inspection helper (used by the W3 sample-verification step) -----


def describe(result: LoadResult) -> dict:
    """Summary stats for a LoadResult sample. For visual inspection."""
    lens = [len(t) for t in result.sentences]
    sorted_lens = sorted(lens)
    n = len(lens)
    return {
        "source": result.source.name,
        "license": result.source.license,
        "hf_id": result.source.hf_id,
        "hf_config": result.source.hf_config,
        "genre": result.source.genre,
        "pipeline": {
            "raw": result.raw_count,
            "after_norm": result.after_norm,
            "after_length": result.after_length,
            "after_dedupe": result.after_dedupe,
            "sample": n,
        },
        "char_lens": {
            "min": min(lens),
            "median": sorted_lens[n // 2],
            "mean": round(sum(lens) / n, 1),
            "max": max(lens),
        },
        "first_sentences": result.sentences[:10],
    }
