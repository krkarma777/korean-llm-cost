"""HuggingFace ``AutoTokenizer`` wrapper for any model on the Hub.

Why this looks different from the API wrappers
------------------------------------------------
``AutoTokenizer.from_pretrained`` downloads only the *tokenizer files*
(``tokenizer.json``, sentencepiece ``.model`` files, special-tokens
maps) — never the model weights. Total per-model footprint: ~5–15 MB.
Files are cached at ``~/.cache/huggingface/hub/`` so the first call
incurs a one-time download and subsequent calls are instantaneous.

For fairness with ``tiktoken``-style raw counting, every ``encode()``
call uses ``add_special_tokens=False`` — no BOS/EOS, no chat template.
The chat-overhead calibration we do for the API wrappers is therefore
trivial here (overhead = 0 by construction), but we keep the field
populated so that the notebook's per-provider init line stays uniform
across providers.

Gated models
------------
Models like ``meta-llama/Llama-3.1-*`` require accepting a license on
HuggingFace and providing an HF token through the ``HF_TOKEN`` or
``HUGGING_FACE_HUB_TOKEN`` env var. Solar, Qwen, and Gemma are open
and need no token. ``transformers`` reads the token from the env
automatically; nothing extra to wire here.
"""

from __future__ import annotations

from transformers import AutoTokenizer

from .base import Tokenizer, TokenizerInfo


class HFTokenizer(Tokenizer):
    """Wraps any HuggingFace tokenizer for raw-content token counting.

    Parameters
    ----------
    model :
        HuggingFace Hub model identifier, e.g.
        ``"upstage/SOLAR-10.7B-Instruct-v1.0"``.
    revision :
        Optional commit SHA or branch name. Defaults to ``"main"``.
        Pinning a revision is important for paper reproducibility —
        the `info.version` field will record whatever you pass here.
    trust_remote_code :
        Some Korean models (e.g., EXAONE) ship custom tokenizer code.
        Set to ``True`` only for models you've audited — leaving it
        ``False`` is safer.

    Examples
    --------
    >>> tok = HFTokenizer("upstage/SOLAR-10.7B-Instruct-v1.0")
    >>> tok.count("안녕하세요")
    5  # representative; depends on actual SentencePiece merges
    """

    def __init__(
        self,
        model: str,
        *,
        revision: str | None = None,
        trust_remote_code: bool = False,
    ):
        self._tok = AutoTokenizer.from_pretrained(
            model,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        self.info = TokenizerInfo(
            name=model,
            version=revision or "main",
            provider="hf",
        )
        # Calibration mirrors the API wrappers; expected to be 0 here
        # because we explicitly disable special tokens. A non-zero value
        # would indicate a tokenizer that fails the "1 char = 1 token"
        # probe assumption — flag and investigate before trusting.
        self._overhead = self._calibrate_overhead()

    def _calibrate_overhead(self) -> int:
        return len(self._tok.encode(".", add_special_tokens=False)) - 1

    def count(self, text: str) -> int:
        if not text:
            return 0
        ids = self._tok.encode(text, add_special_tokens=False)
        return len(ids) - self._overhead
