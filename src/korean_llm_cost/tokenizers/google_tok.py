"""Google Gemini tokenizer via the Generative Language API ``count_tokens``.

Same calibration pattern as ``anthropic_tok``: probe with a single
character (``"."``), derive the per-message overhead constant (zero or
small), and subtract it from every count so the resulting numbers are
directly comparable to ``tiktoken``'s raw-content counts.

The Gemini ``count_tokens`` endpoint accepts a raw string and reports
``total_tokens`` for the model's tokenizer. In practice the overhead is
either 0 or a small constant — the calibration handles both cases
without us having to special-case anything.

Cost & balance
--------------
Gemini's ``count_tokens`` runs against the **free tier** of the
Generative Language API. Unlike Anthropic, Google does not require a
non-zero balance to access the endpoint — a free API key from
https://aistudio.google.com/apikey is sufficient. There are per-minute
rate limits but they are well above what Phase 1 needs.
"""

from __future__ import annotations

import os
import sys
import time

import google.generativeai as genai

from .base import Tokenizer, TokenizerInfo

# Rate-limit retry policy. Gemini's free tier is generous (1500 RPM at
# the time of writing) but corpus-scale runs occasionally hit per-second
# bursts. Exponential backoff matches the Anthropic wrapper: 60s → 120s
# → 240s, max 3 attempts.
_RETRY_BACKOFFS = (60, 120, 240)


def _is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "429" in msg or "quota" in msg or "rate" in msg or "exhausted" in msg:
        return True
    # google.api_core.exceptions.ResourceExhausted is the canonical type
    # but importing it here would force a heavier dep — string check covers it.
    return False

_DEFAULT_MODEL = "gemini-2.5-flash"

# Snapshot dates record the model variant the user requested. Google's
# count_tokens response does not carry a snapshot field, so we map
# common identifiers manually. Update when Google ships a new variant.
_MODEL_SNAPSHOTS: dict[str, str] = {
    "gemini-2.5-flash": "2025-06",     # Phase 1 default
    "gemini-2.5-pro": "2025-06",
    "gemini-2.0-flash": "2025-02",
    "gemini-1.5-flash": "2024-09",
    "gemini-1.5-pro": "2024-09",
}


class GoogleTokenizer(Tokenizer):
    """``count_tokens``-based wrapper for Gemini, with overhead calibration.

    Parameters
    ----------
    model :
        Public Gemini model identifier (see ``_MODEL_SNAPSHOTS``).
        Unknown names are accepted but recorded as ``version='unknown'``.
    api_key :
        Optional. If omitted, the ``GOOGLE_API_KEY`` env var is used.
        ``GEMINI_API_KEY`` is also checked as a fallback because both
        names appear in Google's docs.
    """

    def __init__(self, model: str = _DEFAULT_MODEL, *, api_key: str | None = None):
        key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not key:
            raise RuntimeError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) is not set. Either pass "
                "api_key=... or copy .env.example to .env and fill in the key."
            )
        genai.configure(api_key=key)
        self._model_name = model
        self._model = genai.GenerativeModel(model)
        self.info = TokenizerInfo(
            name=model,
            version=_MODEL_SNAPSHOTS.get(model, "unknown"),
            provider="google",
        )
        self._overhead = self._calibrate_overhead()

    def _api_count_with_retry(self, text: str) -> int:
        """Call ``count_tokens`` with rate-limit retry. Same policy as
        anthropic_tok — retry on 429/quota, propagate other errors.
        """
        for attempt, backoff in enumerate([0, *_RETRY_BACKOFFS]):
            if backoff > 0:
                print(
                    f"[google_tok] retry {attempt}/{len(_RETRY_BACKOFFS)} "
                    f"after {backoff}s backoff",
                    file=sys.stderr,
                )
                time.sleep(backoff)
            try:
                return self._model.count_tokens(text).total_tokens
            except Exception as e:
                if attempt < len(_RETRY_BACKOFFS) and _is_rate_limit_error(e):
                    continue
                raise
        raise RuntimeError("retry loop exited without return — should not happen")

    def _calibrate_overhead(self) -> int:
        """Probe with ``"."`` and back out any constant overhead.

        Same assumption as the Anthropic wrapper: a single printable
        ASCII character is exactly 1 content token under the Gemini
        tokenizer. If that breaks in a future model, calibrate with
        two probes of known-different lengths and solve the linear
        system.
        """
        return self._api_count_with_retry(".") - 1

    def count(self, text: str) -> int:
        if not text:
            return 0
        return self._api_count_with_retry(text) - self._overhead
