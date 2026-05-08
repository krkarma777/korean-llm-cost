"""Phase 1 corpus loader — Korean text by category.

Each category maps to one or more data sources. Sources can be public
HuggingFace datasets (no auth) or local files (e.g. AI Hub downloads
that the user has manually fetched and placed under ``data/conversation/``).

The ingest pipeline is identical regardless of source:

    raw → NFC-normalize → length-filter → exact-string dedupe → seeded sample

Categories wired in Phase 1
---------------------------
- ``"news"`` — KLUE-YNAT (Yonhap News Agency topic headlines).
  Public HuggingFace dataset, CC-BY-SA 4.0. No auth required.
- ``"conversation"`` — AI Hub Topic-based Korean Daily Conversation.
  Local zip files at ``data/conversation/korean/2.Validation/라벨링데이터/``
  (5 messengers: KAKAO/FACEBOOK/INSTAGRAM/BAND/NATEON). Read in-place
  via ``zipfile``, no extraction. Per-utterance metadata (subject,
  speaker_sex, speaker_age, media, session_id) is preserved alongside
  text. Sampling is **media-balanced** (n/5 from each messenger) so
  no single platform dominates the corpus.
- ``"medical"`` — pending.

License caveats for the conversation corpus are tracked in
``notes/07_caveats.md`` §7 (license metadata template artifact) and
§8 (annotator-level alphanumeric notation inconsistency).
"""

from __future__ import annotations

import io
import json
import random
import re
import unicodedata
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

# Path to the AI Hub validation labeled zips. Reading in-place; no extraction.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_AIHUB_CONV_VAL_LABELED = (
    _REPO_ROOT
    / "data"
    / "conversation"
    / "korean"
    / "2.Validation"
    / "라벨링데이터"
)
# Zip filename like "VL_05. NATEON.zip" → media name "NATEON".
_MEDIA_PATTERN = re.compile(r"V[LS]_\d+\.\s*([A-Z]+)\.zip")

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
# Note: "conversation" follows a different code path (with media-balanced
# sampling and per-utterance metadata) — see ``_load_aihub_conversation``.
_SOURCES: dict[str, list[Callable[[], tuple[CorpusSource, list[str]]]]] = {
    "news": [_load_klue_ynat],
    # "medical":   [<KorMedMCQA loader, pending>],
}


# ----- AI Hub conversation loader (in-place zip reading + metadata) -----


def _media_name_from_zip(zip_filename: str) -> str:
    """Extract the messenger name from a VL_NN. NAME.zip filename."""
    m = _MEDIA_PATTERN.search(zip_filename)
    return m.group(1) if m else zip_filename


def _iter_aihub_conversation() -> Iterator[dict]:
    """Yield one utterance dict per `lines[*]` entry across all VL_*.zip.

    Each yielded dict has keys::

        text          : str   norm_text from AI Hub (speaker prefix removed)
        subject       : str   topic label (e.g., "군대", "음식")
        speaker_sex   : str   "남성" / "여성" / ""
        speaker_age   : str   "20대" / "30대" / ...
        media         : str   "KAKAO" / "FACEBOOK" / "INSTAGRAM" / "BAND" / "NATEON"
        session_id    : str   filename stem of the source JSON (e.g. "NATEON_11_05")

    Reads zips in-place via ``zipfile``. No extraction to disk.
    """
    for zip_path in sorted(_AIHUB_CONV_VAL_LABELED.glob("VL_*.zip")):
        media = _media_name_from_zip(zip_path.name)
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                if not member.endswith(".json"):
                    continue
                with zf.open(member) as raw:
                    # AI Hub JSON files are UTF-8.
                    data = json.load(io.TextIOWrapper(raw, encoding="utf-8"))
                info_list = data.get("info") or []
                if not info_list:
                    continue
                info = info_list[0]
                ann = info.get("annotations") or {}
                subject = ann.get("subject") or ""
                session_id = Path(member).stem
                for line in ann.get("lines") or []:
                    nt = (line.get("norm_text") or "").strip()
                    if not nt:
                        continue
                    spk = line.get("speaker") or {}
                    yield {
                        "text": nt,
                        "subject": subject,
                        "speaker_sex": spk.get("sex") or "",
                        "speaker_age": spk.get("age") or "",
                        "media": media,
                        "session_id": session_id,
                    }


def _load_aihub_conversation(
    n: int, seed: int, *, lo: int, hi: int
) -> LoadResult:
    """Load conversation utterances with media-balanced sampling.

    Pipeline (same as the news path, but applied per-utterance dict
    instead of bare strings, and with the final sampling stratified
    over the ``media`` field so no single messenger dominates):

        raw utterances → NFC-normalize → length filter → dedupe by text
                       → group by media → sample n/M per media → shuffle
    """
    if not _AIHUB_CONV_VAL_LABELED.exists():
        raise FileNotFoundError(
            f"AI Hub conversation data not found at {_AIHUB_CONV_VAL_LABELED}. "
            "See data/README.md for the AI Hub download procedure."
        )

    src = CorpusSource(
        name="AI Hub Topic-based Korean Daily Conversation",
        hf_id=None,
        # Use this field to record the local sub-path so reproducibility
        # info is still complete even though it isn't a HF id.
        hf_config="2.Validation/라벨링데이터",
        license="AI Hub academic-research only (no redistribution)",
        genre="conversational",
        notes=(
            "AI Hub dataSetSn=543. Reading labeled JSONs in-place from VL_*.zip; "
            "per-file `licenses` JSON field is template noise — see "
            "notes/07_caveats.md §7."
        ),
    )

    # Stage 1: collect all utterances with metadata.
    raw_items: list[dict] = list(_iter_aihub_conversation())
    raw_count = len(raw_items)

    # Stage 2: NFC normalize text in-place. Drop empties.
    after_norm = 0
    normed_items: list[dict] = []
    for item in raw_items:
        t = unicodedata.normalize("NFC", item["text"]).strip()
        if not t:
            continue
        item = dict(item)
        item["text"] = t
        normed_items.append(item)
        after_norm += 1

    # Stage 3: length filter on text.
    filtered = [it for it in normed_items if lo <= len(it["text"]) <= hi]
    after_length = len(filtered)

    # Stage 4: exact-string dedupe by text.
    seen: set[str] = set()
    deduped: list[dict] = []
    for it in filtered:
        if it["text"] in seen:
            continue
        seen.add(it["text"])
        deduped.append(it)
    after_dedupe = len(deduped)

    # Stage 5: media-balanced sampling. n / M per media (with remainder
    # absorbed by the first few media in alphabetical order).
    by_media: dict[str, list[dict]] = {}
    for it in deduped:
        by_media.setdefault(it["media"], []).append(it)
    media_keys = sorted(by_media)
    if not media_keys:
        raise RuntimeError(
            "No media groups after filtering — AI Hub directory empty or unreadable."
        )

    per_media = n // len(media_keys)
    remainder = n - per_media * len(media_keys)
    rng = random.Random(seed)
    sampled: list[dict] = []
    for i, media in enumerate(media_keys):
        take = per_media + (1 if i < remainder else 0)
        pool = by_media[media]
        if len(pool) < take:
            raise RuntimeError(
                f"Only {len(pool)} utterances for media={media!r}; need {take}. "
                "Loosen filters or include the training split."
            )
        sampled.extend(rng.sample(pool, take))

    rng.shuffle(sampled)  # avoid the output being grouped by media

    sentences = [it["text"] for it in sampled]
    metadata = tuple(
        {k: v for k, v in it.items() if k != "text"} for it in sampled
    )

    return LoadResult(
        source=src,
        sentences=sentences,
        raw_count=raw_count,
        after_norm=after_norm,
        after_length=after_length,
        after_dedupe=after_dedupe,
        seed=seed,
        metadata=metadata,
    )


# ----- Pipeline -----


def _normalize(text: str) -> str:
    """NFC normalization + strip. macOS file inputs often arrive in NFD."""
    return unicodedata.normalize("NFC", text).strip()


def _length_ok(text: str, *, lo: int = 5, hi: int = 500) -> bool:
    """5 ≤ length ≤ 500 chars. Filters out '.' headlines and concat dumps."""
    return lo <= len(text) <= hi


@dataclass(frozen=True)
class LoadResult:
    """What a load_category call returns. Carries provenance for reporting.

    Attributes
    ----------
    metadata
        For sources that carry per-utterance side data (e.g. AI Hub
        conversation: subject / speaker_sex / speaker_age / media /
        session_id), this is a tuple aligned by index with
        ``sentences``. ``None`` for sources that don't have it
        (currently: news/KLUE-YNAT).
    """

    source: CorpusSource
    sentences: list[str]
    raw_count: int          # before any filtering
    after_norm: int         # after NFC + strip
    after_length: int       # after length filter
    after_dedupe: int       # after exact-string dedupe
    seed: int
    metadata: tuple[dict, ...] | None = None


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
    # Conversation has its own pipeline (in-place zip reads, metadata,
    # media-balanced sampling) — handle separately.
    if category == "conversation":
        return _load_aihub_conversation(n, seed, lo=lo_chars, hi=hi_chars)

    if category not in _SOURCES:
        available = ", ".join(["conversation", *_SOURCES]) or "(none yet)"
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
