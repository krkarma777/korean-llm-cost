"""Phase 1 metrics — TPC, KPR, ECPC, bootstrap CI, paired tests.

Design
------
The two natural levels of granularity are:

- **Per-text counts**: ``(n_tokens, n_chars)`` for each sentence. Needed for
  paired tests and bootstrap CI.
- **Aggregate TPC**: ``sum(tokens) / sum(chars)`` over a corpus. The
  headline number reported in the paper. Note this is *not* the mean of
  per-text TPCs — averaging per-text ratios over-weights short texts.

We carry both in the ``CountSet`` dataclass and derive everything from
there. The high-level ``measure(tokenizer, sentences)`` helper is the
one-line entry point experiments will call.

KPR and ECPC live here as functions — they take aggregate TPCs (not
counts), because both are derived ratios on top of TPC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ----- Per-corpus counts -----


@dataclass(frozen=True)
class CountSet:
    """Per-sample token + char counts for one (tokenizer, corpus) pair.

    Field invariants:
      - ``len(tokens) == len(chars)`` (one entry per sample)
      - all entries are non-negative
      - ``chars[i] > 0`` for all i (we drop empty strings before measuring)
    """

    tokens: tuple[int, ...]
    chars: tuple[int, ...]

    @property
    def n(self) -> int:
        return len(self.tokens)

    @property
    def total_tokens(self) -> int:
        return sum(self.tokens)

    @property
    def total_chars(self) -> int:
        return sum(self.chars)

    @property
    def aggregate_tpc(self) -> float:
        """Tokens per character, aggregated as sum(tokens)/sum(chars)."""
        return self.total_tokens / self.total_chars

    @property
    def per_text_tpc(self) -> tuple[float, ...]:
        return tuple(t / c for t, c in zip(self.tokens, self.chars))

    def bootstrap_tpc_ci(
        self,
        *,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        seed: int = 42,
    ) -> tuple[float, float]:
        """Bootstrap CI for aggregate TPC.

        Resamples *paired* (tokens, chars) entries with replacement and
        recomputes the aggregate ratio. The aggregate-style resample is
        important because per-text TPCs are not exchangeable across text
        lengths.
        """
        rng = np.random.default_rng(seed)
        toks = np.asarray(self.tokens, dtype=np.int64)
        chrs = np.asarray(self.chars, dtype=np.int64)
        n = len(toks)
        ratios = np.empty(n_bootstrap, dtype=np.float64)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, n)
            ratios[b] = toks[idx].sum() / chrs[idx].sum()
        alpha = (1 - ci) / 2
        return float(np.quantile(ratios, alpha)), float(np.quantile(ratios, 1 - alpha))


# ----- KPR / ECPC -----


def kpr(ko: CountSet, en: CountSet) -> float:
    """Korean Penalty Ratio = TPC_korean / TPC_english.

    Note: ``ko`` and ``en`` are *not* required to have aligned per-text
    samples. KPR is a corpus-level ratio. Per-text-aligned KPR is a
    separate analysis that requires paired ko↔en data.
    """
    return ko.aggregate_tpc / en.aggregate_tpc


def ecpc(tpc_value: float, price_per_1k_tokens: float) -> float:
    """Effective Cost per 1K Korean characters, in USD.

    Derivation:
        cost_per_char = TPC * price_per_token
                      = TPC * (price_per_1k_tokens / 1000)
        cost_per_1k_chars = cost_per_char * 1000
                          = TPC * price_per_1k_tokens
    """
    return tpc_value * price_per_1k_tokens


# ----- Paired tests -----


def paired_wilcoxon_tpc(a: CountSet, b: CountSet) -> dict:
    """Wilcoxon signed-rank test on per-text TPC differences.

    H0: TPC distributions are equal across the paired texts.

    Requires ``a.n == b.n`` and the two CountSets to share the same
    text ordering (i.e., ``a.tokens[i]`` and ``b.tokens[i]`` come from
    the same source sentence).

    Returns a dict with ``statistic``, ``pvalue``, and ``median_diff``
    (median of per-text TPC differences, signed: positive means ``a``
    has higher TPC).
    """
    if a.n != b.n:
        raise ValueError(f"Paired test requires equal n; got {a.n} vs {b.n}.")
    # Lazy import: scipy is in the `analysis` extra and may not be loaded
    # in environments that only need raw counting.
    from scipy.stats import wilcoxon

    a_tpcs = np.asarray(a.per_text_tpc)
    b_tpcs = np.asarray(b.per_text_tpc)
    diffs = a_tpcs - b_tpcs
    if np.allclose(diffs, 0):
        return {"statistic": 0.0, "pvalue": 1.0, "median_diff": 0.0}
    res = wilcoxon(a_tpcs, b_tpcs)
    return {
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "median_diff": float(np.median(diffs)),
    }


# ----- High-level helpers -----


def measure(tokenizer, sentences: Sequence[str]) -> CountSet:
    """Tokenize each sentence with ``tokenizer`` and return a CountSet.

    Drops zero-length sentences (defensive — corpus_loader already
    filters these, but safer to handle here too).
    """
    tokens: list[int] = []
    chars: list[int] = []
    for s in sentences:
        if not s:
            continue
        c = len(s)
        t = tokenizer.count(s)
        tokens.append(t)
        chars.append(c)
    return CountSet(tokens=tuple(tokens), chars=tuple(chars))


@dataclass(frozen=True)
class ModelResult:
    """One model's measurement on one corpus, with CI."""

    model_name: str
    model_version: str
    provider: str
    category: str
    counts: CountSet
    tpc: float           # aggregate
    tpc_ci_low: float    # 95% bootstrap
    tpc_ci_high: float

    def as_dict(self) -> dict:
        return {
            "model": self.model_name,
            "version": self.model_version,
            "provider": self.provider,
            "category": self.category,
            "n": self.counts.n,
            "tokens_total": self.counts.total_tokens,
            "chars_total": self.counts.total_chars,
            "tpc": round(self.tpc, 4),
            "tpc_ci_low": round(self.tpc_ci_low, 4),
            "tpc_ci_high": round(self.tpc_ci_high, 4),
        }


def measure_with_ci(
    tokenizer,
    sentences: Sequence[str],
    *,
    category: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> ModelResult:
    """One-shot: measure + bootstrap CI for one (tokenizer, corpus) pair."""
    counts = measure(tokenizer, sentences)
    lo, hi = counts.bootstrap_tpc_ci(n_bootstrap=n_bootstrap, seed=seed)
    return ModelResult(
        model_name=tokenizer.info.name,
        model_version=tokenizer.info.version,
        provider=tokenizer.info.provider,
        category=category,
        counts=counts,
        tpc=counts.aggregate_tpc,
        tpc_ci_low=lo,
        tpc_ci_high=hi,
    )
