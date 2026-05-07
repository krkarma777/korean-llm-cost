"""OpenAI tokenizer via ``tiktoken`` — fully offline, no API key required.

``tiktoken`` ships the encoding tables used by OpenAI models, so we can
count tokens for any GPT-3.5/4/4o variant locally. This is the cheapest
backend by far and the right one to use during sanity-checking.

Caveat: chat-completion requests add format tokens (``<|im_start|>``,
role tags, message separators) on top of the raw content tokens. We
deliberately count *only the content* — the paper compares tokenizer
efficiency, not chat-template overhead, and Anthropic / Google / HF
tokenizers don't expose the chat overhead in a comparable way.
"""

from __future__ import annotations

import tiktoken

from .base import Tokenizer, TokenizerInfo

# Dated snapshots pin which model variant each name refers to. tiktoken
# is encoding-based and doesn't carry these dates itself, so we record
# them here. Update when OpenAI ships a new dated snapshot.
_MODEL_SNAPSHOTS: dict[str, str] = {
    "gpt-4o": "2024-11-20",
    "gpt-4o-mini": "2024-07-18",
    "gpt-4-turbo": "2024-04-09",
    "gpt-4": "0613",
    "gpt-3.5-turbo": "0125",
}


class OpenAITokenizer(Tokenizer):
    """Wraps ``tiktoken`` for any OpenAI model name it recognizes.

    Examples
    --------
    >>> tok = OpenAITokenizer("gpt-4o")
    >>> tok.count("안녕하세요")
    4
    >>> tok.info.version
    '2024-11-20'
    """

    def __init__(self, model: str = "gpt-4o"):
        try:
            self._enc = tiktoken.encoding_for_model(model)
        except KeyError as e:
            raise ValueError(
                f"tiktoken does not recognize model {model!r}. "
                f"Pass a known OpenAI model name, e.g. 'gpt-4o'."
            ) from e

        self.info = TokenizerInfo(
            name=model,
            version=_MODEL_SNAPSHOTS.get(model, self._enc.name),
            provider="openai",
        )

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))
