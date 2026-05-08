"""Pre-registered hypothesis bands for the n=1000 KorMedMCQA medical measurement.

This module is the **pre-registration timestamp** for the medical-category
sub-analysis on KorMedMCQA. It is committed *before* any tokenizer is run
on the medical corpus. The bands and the sampling protocol are fixed once
committed; any later change must be tracked as a separate, justified
amendment in the git history.

Long-form rationale — per-hypothesis reasoning, dataset schema, sampling
justification, falsification logic, auxiliary predictions, and Discussion-
level narrative — is preserved in the gitignored
``notes/11_predictions_medical.md``. The binding pre-registration is *this
file*, by virtue of its commit timestamp; the notes file is human-readable
companion documentation.

Hypotheses (mutually exclusive bands)
-------------------------------------
The conversation-corpus P7 sub-analysis falsified the pre-registered
entity-density gradient in direction (see ``notes/02_phase1_results.md``,
``notes/07_caveats.md §11``). The medical category is the cross-validating
experiment that adjudicates between three competing mechanism stories for
EXAONE's vocabulary-extension advantage. The bands below are non-overlapping
by construction so that a measured EXAONE KPR/GPT on medical lands in at
most one band.

(a) Entity-density penalty resurfaces — KPR/GPT in [0.65, 0.75]
    Medical entities (한자 의약품명, 진단명, Latin scientific terms) are
    *unfamiliar* BPE territory, qualitatively distinct from the
    conversational "entity-rich" subjects (game / broadcast / movie) which
    were dominated by BPE-friendly English loanwords. With genuine
    unfamiliar-entity load, byte-level fallback should activate strongly on
    penalty-cluster tokenizers and EXAONE's Korean-vocab extension should
    pay off most. P7's reverse direction was a corpus artifact; real
    entity density behaves as originally hypothesized.

(c) Two mechanisms operating jointly — KPR/GPT in [0.76, 0.84]
    Vocabulary extension benefits both unfamiliar-BPE Korean
    (entity-density component) and colloquial Korean (register component).
    Medical activates the entity-density component but loses the register
    component, leaving EXAONE's net advantage at or modestly weaker than
    its news baseline (KPR/GPT 0.76× news). The two mechanisms partially
    counteract on the medical genre.

(b) Register-invariance is the only mechanism — KPR/GPT in [0.85, 0.95]
    EXAONE's only edge over English-trained BPEs is colloquial-register
    coverage. On formal medical Korean, EXAONE converges toward the
    efficient-cluster baseline; entity density was never a vocabulary-
    extension lever for Korean. P7's reverse-direction finding was the
    right mechanism after all.

Buffer and outside-band handling
--------------------------------
- (0.75, 0.76) — buffer zone, near-(a). Reported as "(a)-leaning" but not
  a clean (a) verdict; Discussion explicitly calls out the buffer.
- (0.84, 0.85) — buffer zone, near-(b). Reported as "(b)-leaning" but not
  a clean (b) verdict.
- KPR/GPT < 0.65 — *all three* hypotheses falsified, EXAONE is *more*
  efficient on medical than even (a) predicts. Possible mechanisms to
  consider: EXAONE's Korean training corpus may include disproportionate
  medical text; or 한자 entities trigger extra byte-fallback on penalty-
  cluster tokenizers beyond what (a) anticipates.
- KPR/GPT > 0.95 — *all three* hypotheses falsified, EXAONE essentially
  loses its advantage on medical. Possible mechanisms: medical's
  formal-register Latin/한자 mix may be uniformly hard for *every*
  tokenizer (cluster gap collapses); or EXAONE's vocabulary extension
  doesn't cover medical Korean specifically.

A landing in either outside-band region is itself a publishable finding,
not a measurement failure. The Discussion would dedicate a paragraph to
articulating the fourth-mechanism scenario.
"""

from __future__ import annotations

# ----- Mutually exclusive hypothesis bands (KPR/GPT, EXAONE on medical) -----

# Inclusive bands. The open intervals between adjacent bands are buffers.
HYPOTHESIS_A_ENTITY_DENSITY: tuple[float, float] = (0.65, 0.75)
HYPOTHESIS_C_BOTH_MECHANISMS: tuple[float, float] = (0.76, 0.84)
HYPOTHESIS_B_REGISTER_ONLY: tuple[float, float] = (0.85, 0.95)

BUFFER_A_C: tuple[float, float] = (0.75, 0.76)  # near-(a), ambiguous
BUFFER_C_B: tuple[float, float] = (0.84, 0.85)  # near-(b), ambiguous

# Catastrophic-falsification thresholds (all three hypotheses out).
ALL_FALSIFIED_BELOW: float = 0.65
ALL_FALSIFIED_ABOVE: float = 0.95


def classify_kpr_gpt(value: float) -> str:
    """Classify a measured EXAONE KPR/GPT against the pre-registered bands.

    Returns one of:
      'a'                  — entity-density penalty resurfaces (clean)
      'a-buffer'           — buffer near-(a), ambiguous
      'c'                  — two mechanisms (clean)
      'c-buffer'           — buffer above (c) and below (b), ambiguous
      'b'                  — register-invariance only (clean)
      'all-falsified-low'  — < 0.65, new mechanism required
      'all-falsified-high' — > 0.95, new mechanism required
    """
    if value < ALL_FALSIFIED_BELOW:
        return "all-falsified-low"
    if value > ALL_FALSIFIED_ABOVE:
        return "all-falsified-high"
    a_lo, a_hi = HYPOTHESIS_A_ENTITY_DENSITY
    c_lo, c_hi = HYPOTHESIS_C_BOTH_MECHANISMS
    b_lo, b_hi = HYPOTHESIS_B_REGISTER_ONLY
    if a_lo <= value <= a_hi:
        return "a"
    if c_lo <= value <= c_hi:
        return "c"
    if b_lo <= value <= b_hi:
        return "b"
    if a_hi < value < c_lo:
        return "a-buffer"
    if c_hi < value < b_lo:
        return "c-buffer"
    raise AssertionError(f"unreachable classification for value={value}")


# ----- Auxiliary predictions (paired with EXAONE-band classification) -----

# (d) Cluster gap (KPR/GPT, efficient_max → penalty_min) on medical.
#     News 0.36×, conversation 0.61×. Band covers both narrowing and modest
#     widening on medical.
CLUSTER_GAP_BAND: tuple[float, float] = (0.40, 0.65)

# (e) Claude / Solar genre insensitivity persists from conversation:
#     |TPC_medical − TPC_conversation| / TPC_conversation < 5%, both models.
PENALTY_GENRE_INSENSITIVITY_PCT: float = 5.0

# (f) Stem-length distribution after the corpus pipeline.
STEM_LENGTH_MEDIAN_BAND: tuple[int, int] = (60, 130)
STEM_LENGTH_P90_MAX: int = 350


# ----- Sampling protocol (frozen at pre-registration) -----

KORMEDMCQA_REPO: str = "sean0042/KorMedMCQA"
KORMEDMCQA_CONFIGS: tuple[str, ...] = ("dentist", "doctor", "nurse", "pharm")

# Stratified balanced sample: 250 stems per config, drawn from the 'train'
# split. Parallels the 200-utterance × 5-messenger media-balanced sample
# used for the conversation corpus.
PER_CONFIG_SAMPLE: int = 250
TOTAL_SAMPLE: int = PER_CONFIG_SAMPLE * len(KORMEDMCQA_CONFIGS)
SPLIT_TO_SAMPLE_FROM: str = "train"

SAMPLE_SEED: int = 42

# Pipeline matches news / conversation runs:
#   1. NFC normalize
#   2. length filter — 1000-char ceiling is wider than conversation's 500
#      because doctor stems can legitimately reach 600+ chars (long clinical
#      vignettes, not data noise).
#   3. exact-string dedupe
#   4. seeded random sample, PER_CONFIG_SAMPLE per config
LENGTH_FILTER_MIN: int = 5
LENGTH_FILTER_MAX: int = 1000


# ----- Measurement-unit decision (frozen at pre-registration) -----

# Option A — measure the `question` (stem) field only.
#
# Including A–E choices would inject repetitive structural patterns
# (parenthesized English equivalents, label markers, parallel grammatical
# forms) that inflate absolute TPC with structural artifacts and dilute the
# medical-content signal. Stems are the genuine medical text; choices are
# answer variants with low marginal information value about Korean
# tokenization on medical content.
MEASUREMENT_UNIT: str = "question"

# Escape hatch: if the 100-stem pilot reveals median stem length ≤
# ESCAPE_HATCH_MIN_MEDIAN_CHARS, switch to Option B (stem + 5 choices joined
# by single spaces, identical preprocessing pipeline). Any switch to Option
# B must be a *separate amendment commit* before the n=1000 run.
ESCAPE_HATCH_MIN_MEDIAN_CHARS: int = 50


# ----- Self-checks (catch typos at import time) -----

assert HYPOTHESIS_A_ENTITY_DENSITY[1] < HYPOTHESIS_C_BOTH_MECHANISMS[0]
assert HYPOTHESIS_C_BOTH_MECHANISMS[1] < HYPOTHESIS_B_REGISTER_ONLY[0]
assert ALL_FALSIFIED_BELOW == HYPOTHESIS_A_ENTITY_DENSITY[0]
assert ALL_FALSIFIED_ABOVE == HYPOTHESIS_B_REGISTER_ONLY[1]
assert TOTAL_SAMPLE == 1000
assert classify_kpr_gpt(0.70) == "a"
assert classify_kpr_gpt(0.80) == "c"
assert classify_kpr_gpt(0.90) == "b"
assert classify_kpr_gpt(0.755) == "a-buffer"
assert classify_kpr_gpt(0.845) == "c-buffer"
assert classify_kpr_gpt(0.60) == "all-falsified-low"
assert classify_kpr_gpt(1.00) == "all-falsified-high"
# Boundary inclusivity at exact band edges
assert classify_kpr_gpt(0.65) == "a"
assert classify_kpr_gpt(0.75) == "a"
assert classify_kpr_gpt(0.76) == "c"
assert classify_kpr_gpt(0.84) == "c"
assert classify_kpr_gpt(0.85) == "b"
assert classify_kpr_gpt(0.95) == "b"
