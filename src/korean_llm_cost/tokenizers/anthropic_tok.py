"""Anthropic (Claude) tokenizer via the Messages ``count_tokens`` API.

Why this looks different from ``openai_tok``
--------------------------------------------
Claude 3+ does NOT ship a public local tokenizer. The only sanctioned way
to count tokens is the ``count_tokens`` API endpoint, which counts tokens
for a *fully chat-formatted message* — including role markers and the
internal scaffolding Anthropic adds to every prompt. That count is **not
directly comparable** to ``tiktoken``'s raw-content counts.

To make TPC mean the same thing across providers, we calibrate the chat
overhead at ``__init__`` time and subtract it from every count:

1. Send a 1-character probe (``"."``). The endpoint returns
   ``input_tokens = overhead + 1``.
2. Subtract 1 to recover ``overhead``.
3. For every later ``count(text)`` call, return
   ``api_count(text) - overhead``.

The "1 character = 1 token" assumption holds for the BPE-style
tokenizers used by Claude 3+ on a printable ASCII probe. If a future
Claude release violates this, the wrapper will produce systematically
biased counts — re-derive ``overhead`` by sending two probes of known
different lengths and solving for the constant.

Cost & speed
------------
``count_tokens`` has **zero marginal cost** — Anthropic does not bill per
call. However, the API requires a *non-zero account balance* to access
at all (an empty-balance account gets HTTP 400 with
``"credit balance is too low"``), so a one-time minimum top-up at
console.anthropic.com is required even though counting itself is free.
Each call is also a network round-trip (~50–200 ms). For corpus-scale
runs we'll need a parallel batch path; the inherited ``count_batch`` is
sequential and fine for sanity-check / Phase 1 sample sizes.
"""

from __future__ import annotations

import os
import sys
import time

import anthropic

from .base import Tokenizer, TokenizerInfo

# Rate-limit retry policy. Corpus-scale sequential calls (1000+) can trip
# Anthropic's RPM ceiling. Exponential backoff: 60s → 120s → 240s, max 3
# attempts before the error propagates.
_RETRY_BACKOFFS = (60, 120, 240)


def _is_rate_limit_error(exc: BaseException) -> bool:
    if isinstance(exc, anthropic.RateLimitError):
        return True
    msg = str(exc).lower()
    return "rate" in msg or "429" in msg or "overload" in msg

_DEFAULT_MODEL = "claude-sonnet-4-5"

# Snapshot dates record the model variant the user requested. Anthropic's
# API does not currently include a snapshot field in count_tokens
# responses, so we map common identifiers manually.
_MODEL_SNAPSHOTS: dict[str, str] = {
    "claude-sonnet-4-5": "2025-09",
    "claude-sonnet-4-6": "2026-02",
    "claude-sonnet-4-7": "2026-04",  # Phase 1 target per project plan
    "claude-opus-4-5": "2025-09",
    "claude-opus-4-6": "2026-02",
    "claude-opus-4-7": "2026-04",
    "claude-3-5-sonnet-latest": "2024-10-22",
    "claude-3-5-haiku-latest": "2024-10-22",
}


class AnthropicTokenizer(Tokenizer):
    """``count_tokens``-based wrapper with chat-overhead calibration.

    Parameters
    ----------
    model :
        Public Claude model identifier (see ``_MODEL_SNAPSHOTS`` for
        recognized names). Unknown names are accepted but recorded as
        ``version='unknown'`` — fix the snapshot table before relying
        on those numbers in the paper.
    api_key :
        Optional. If omitted, the ``ANTHROPIC_API_KEY`` env var is used.
    """

    def __init__(self, model: str = _DEFAULT_MODEL, *, api_key: str | None = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Either pass api_key=... or "
                "copy .env.example to .env and fill in the key."
            )
        self._client = anthropic.Anthropic(api_key=key)
        self._model = model
        self.info = TokenizerInfo(
            name=model,
            version=_MODEL_SNAPSHOTS.get(model, "unknown"),
            provider="anthropic",
        )
        self._overhead = self._calibrate_overhead()

    def _api_count_with_retry(self, text: str) -> int:
        """Call ``messages.count_tokens`` with rate-limit retry.

        Retries up to 3 times on RateLimitError or 429-flavored errors,
        with backoffs from ``_RETRY_BACKOFFS``. Other errors propagate
        immediately. Logs each retry to stderr so corpus-scale runs leave
        a visible trail.
        """
        for attempt, backoff in enumerate([0, *_RETRY_BACKOFFS]):
            if backoff > 0:
                print(
                    f"[anthropic_tok] retry {attempt}/{len(_RETRY_BACKOFFS)} "
                    f"after {backoff}s backoff",
                    file=sys.stderr,
                )
                time.sleep(backoff)
            try:
                resp = self._client.messages.count_tokens(
                    model=self._model,
                    messages=[{"role": "user", "content": text}],
                )
                return resp.input_tokens
            except Exception as e:
                if attempt < len(_RETRY_BACKOFFS) and _is_rate_limit_error(e):
                    continue
                raise
        # Should be unreachable due to the loop's last iteration always
        # either returning or raising.
        raise RuntimeError("retry loop exited without return — should not happen")

    def _calibrate_overhead(self) -> int:
        """Measure the chat-template token overhead once at init.

        Probes with a single ``"."`` character, which Claude 3+ tokenizes
        as exactly 1 content token. Whatever the API reports beyond that
        is the per-message scaffolding cost.
        """
        return self._api_count_with_retry(".") - 1

    def count(self, text: str) -> int:
        if not text:
            return 0
        return self._api_count_with_retry(text) - self._overhead
