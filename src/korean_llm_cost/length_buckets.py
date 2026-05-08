"""Pre-registered length-bucket boundaries for the P9 medical sub-analysis.

This module is the **pre-registration timestamp** for the length-stratified
component of the P9 medical sub-analysis. It is committed *before* any
length-stratified statistic is computed against the n=1000 KorMedMCQA
per-text CSV. The bucket boundaries are fixed once committed; any later
change must be tracked as a separate, justified amendment in the git
history (mirroring the ``subject_groups.py`` and ``medical_predictions.py``
pre-registration pattern used elsewhere in this project).

Bucket rationale
----------------
The two boundaries (50 and 150 chars) are derived from the KorMedMCQA
n=100 pilot length distribution (`experiments/06_pilot_medical_n100.py`,
results in `notes/11_predictions_medical.md`):

  - ≤ 50 chars   → 'short'   — dominated by dentist (median 49)
                                and nurse (median 64) stems.
  - 51-150 chars → 'medium'  — dominated by pharm stems
                                (median 60, p90 141).
  - > 150 chars  → 'long'    — dominated by doctor stems
                                (median 181, p90 334).

Boundaries chosen so that each bucket roughly aligns with one config's
natural length distribution. This makes per-bucket counts well-populated
(neither bucket has < ~150 samples in the n=1000 stratified sample) and
makes the bucket-vs-config confound analysis readable.

Acknowledged confound — length and config are largely aligned
-------------------------------------------------------------
KorMedMCQA's per-config length distributions cluster: short stems are
mostly dentist/nurse, long stems are mostly doctor. This means a
"length effect" and a "config effect" cannot be fully separated in this
single corpus. The P9 sub-analysis must report both directions of the
2-way table:

  (i)  cross-bucket TPC differences *within* each config
       (controls config; isolates length effect)
  (ii) cross-config TPC differences *within* each bucket
       (controls length; isolates config effect)
  (iii) the joint config × bucket distribution itself
       (quantifies the alignment so reviewers can see how much
       residual signal remains after the controls)

If after both controls a residual effect persists in only one direction,
the sub-analysis identifies which factor dominates. If both residuals
shrink, length and config are largely confounded in this corpus and the
sub-analysis reports that limitation honestly. Either outcome is a
publishable conclusion; the inability to fully separate the two effects
in KorMedMCQA is a *corpus limitation* (paper Limitations §1 / §3),
not a measurement failure.

Pre-registration discipline applies the same logic that underwrote
``subject_groups.py`` (P7) and ``medical_predictions.py`` (medical
verdict): the boundaries are committed before the ratios are computed,
so post-hoc bucket selection cannot be a reviewer concern.
"""

from __future__ import annotations

from typing import Mapping

# ----- Bucket boundaries (frozen at pre-registration) -----

# Inclusive upper bounds for the short and medium buckets. The "long"
# bucket has no upper bound (the corpus pipeline already enforces a
# max length of medical_predictions.LENGTH_FILTER_MAX = 1000).
LENGTH_BUCKET_SHORT_MAX: int = 50
LENGTH_BUCKET_MEDIUM_MAX: int = 150

BUCKET_ORDER: tuple[str, ...] = ("short", "medium", "long")


def classify_length_bucket(n_chars: int) -> str:
    """Return the pre-registered length bucket for a stem character count.

    Boundaries (inclusive):
      - n_chars ≤ 50  → 'short'
      - 51 ≤ n_chars ≤ 150  → 'medium'
      - n_chars ≥ 151 → 'long'

    Raises ``ValueError`` for non-positive counts (corpus pipeline already
    filters anything ≤ 4 chars; defensive boundary on the lower side).
    """
    if n_chars <= 0:
        raise ValueError(f"n_chars must be positive, got {n_chars}")
    if n_chars <= LENGTH_BUCKET_SHORT_MAX:
        return "short"
    if n_chars <= LENGTH_BUCKET_MEDIUM_MAX:
        return "medium"
    return "long"


# ----- Self-checks (catch typos at import time) -----

# Boundary monotonicity
assert LENGTH_BUCKET_SHORT_MAX < LENGTH_BUCKET_MEDIUM_MAX

# Boundary inclusivity (50→short, 51→medium, 150→medium, 151→long)
assert classify_length_bucket(1) == "short"
assert classify_length_bucket(50) == "short"
assert classify_length_bucket(51) == "medium"
assert classify_length_bucket(150) == "medium"
assert classify_length_bucket(151) == "long"
assert classify_length_bucket(389) == "long"  # max stem from pilot

# Bucket order matches return values
assert set(BUCKET_ORDER) == {classify_length_bucket(c) for c in (1, 100, 200)}
