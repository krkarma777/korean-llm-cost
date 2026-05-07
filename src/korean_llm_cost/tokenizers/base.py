"""Abstract tokenizer interface for cross-model comparison.

Every provider wrapper (OpenAI, Anthropic, Google, HuggingFace) implements
`Tokenizer` so the rest of the pipeline (metrics, experiments) stays
provider-agnostic.

Design notes
------------
- We only require *token counts*, not the encoded ids. Some providers
  (Anthropic `count_tokens`, Google `countTokens`) only expose counts via
  remote APIs, so locking the interface to counts keeps every backend on
  equal footing.
- `TokenizerInfo` is mandatory metadata. Tokenizers change when models
  update (e.g., GPT-4 → GPT-4o switched encodings). Reviewers will ask
  which snapshot the numbers came from — the paper will reference these
  fields verbatim.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizerInfo:
    """Identification metadata for reproducibility.

    Attributes
    ----------
    name :
        Public model name as a user would request it from the API
        (e.g., ``"gpt-4o"``, ``"claude-sonnet-4-7"``, ``"solar-1-mini"``).
    version :
        A pin: dated snapshot (``"2024-11-20"``) for API models, or the
        underlying tokenizer name (``"o200k_base"``) when no dated
        snapshot exists. Must be specific enough that re-running the
        experiment in 12 months produces the same numbers.
    provider :
        One of ``"openai" | "anthropic" | "google" | "hf" | "naver" | ...``.
        Used for grouping in result tables and for selecting which API
        client to instantiate.
    """

    name: str
    version: str
    provider: str


class Tokenizer(ABC):
    """Minimal interface every tokenizer wrapper must satisfy.

    Subclasses set ``self.info`` in ``__init__`` and implement ``count``.
    ``count_batch`` has a default sequential implementation; override it
    when the underlying library exposes a batch API (e.g., HuggingFace
    fast tokenizers, Google's ``countTokens`` accepts arrays).
    """

    info: TokenizerInfo

    @abstractmethod
    def count(self, text: str) -> int:
        """Return the number of tokens billed for ``text``.

        Implementations should count tokens for the *raw* text as it
        would be sent in a user message — no chat template, no system
        prompt, no special tokens added by the provider's chat formatter.
        That keeps TPC comparable across providers.
        """
        ...

    def count_batch(self, texts: list[str]) -> list[int]:
        """Return token counts for each text. Default: sequential loop."""
        return [self.count(t) for t in texts]

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.info.name}@{self.info.version}>"
