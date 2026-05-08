"""Pre-registered 3-tier subject classification for the P7 sub-analysis.

This module is the **pre-registration timestamp** for the entity-density
sub-analysis on the AI Hub Topic-based Korean Daily Conversation corpus
(n=1000, validation split). It is committed *before* any tier-level TPC
statistic is computed. The classification is fixed once committed; any
later change must be tracked as a separate, justified amendment in the
git history.

Rationale
---------
The full 20-subject inventory is partitioned into three tiers based on
expected named-entity density (proper nouns: titles, brand names, place
names, etc.) in conversational utterances. Compared to a binary
rich/light split, the 3-tier scheme:

1. eliminates "why these 4" selection-bias objections (whole inventory
   is classified, no cherry-picking),
2. yields balanced rich/light arms (8 vs 8) for higher statistical power,
3. supports a Discussion claim that entity-density is a *gradient*, not
   a binary feature — predicting a monotonic TPC ordering
   ``rich > neutral > light`` across all seven models if the
   entity-density mechanism for vocabulary-extension benefit holds.

Per-subject rationale (preserved from the pre-registration discussion):

    [entity-rich (8)]
    - 게임            — 게임 제목/캐릭터명/시리즈명 빈도 높음
    - 타 국가 이슈     — 국가명/지명/외국인명 entity-dense
    - 사회이슈         — 사건명/조직명/제도명 entity-dense
    - 회사/아르바이트   — 회사명/직무용어 entity-dense
    - 방송/연예        — 프로그램명/연예인명 entity-dense
    - 영화/만화        — 작품명/캐릭터명 entity-dense
    - 스포츠/레저      — 팀명/선수명/대회명 entity-dense
    - 군대            — 부대명/장비명/계급 entity-dense
                         (sample에서 K-9 등 phonetic 다수 확인)

    [entity-light (8)]
    - 계절/날씨        — 일반 어휘 위주
    - 식음료           — 일반 음식명 위주, 흔한 BPE 커버
    - 미용            — 일반 어휘 위주
    - 가족            — 호칭/관계 일반 어휘
    - 반려동물         — 일반 어휘 + 흔한 동물명
    - 건강            — 회화 맥락에선 "운동/식사" 같은 일상 건강
                         표현 위주 (의학 전문용어보다)
    - 주거와 생활      — 일반 어휘 위주
    - 연애/결혼        — 일반 어휘 위주

    [neutral (4)]
    - 교육            — 학교명/시험명 entity와 일반 어휘 혼재
    - 여행            — 지명 entity 다수, 단 일반 어휘도 많아 boundary
    - 상거래 전반      — 브랜드명 일부 + 거래 일반 어휘
    - 교통            — 지명/노선명 entity와 일반 이동 어휘 혼재

Subject-string note
-------------------
Strings below match the **exact** subject labels in the AI Hub source
(e.g., ``타 국가 이슈`` with internal spaces). The dataset's 20-subject
inventory was enumerated against
``results/raw/03_tpc_conversation_n1000_per_text.csv`` at
pre-registration time and matched 20/20.
"""

from __future__ import annotations

from typing import Mapping

# ----- Tier definitions (frozen at pre-registration) -----

ENTITY_RICH: tuple[str, ...] = (
    "게임",
    "타 국가 이슈",
    "사회이슈",
    "회사/아르바이트",
    "방송/연예",
    "영화/만화",
    "스포츠/레저",
    "군대",
)

ENTITY_LIGHT: tuple[str, ...] = (
    "계절/날씨",
    "식음료",
    "미용",
    "가족",
    "반려동물",
    "건강",
    "주거와 생활",
    "연애/결혼",
)

NEUTRAL: tuple[str, ...] = (
    "교육",
    "여행",
    "상거래 전반",
    "교통",
)

TIER_ORDER: tuple[str, ...] = ("entity-rich", "neutral", "entity-light")

SUBJECT_TO_TIER: Mapping[str, str] = {
    **{s: "entity-rich" for s in ENTITY_RICH},
    **{s: "entity-light" for s in ENTITY_LIGHT},
    **{s: "neutral" for s in NEUTRAL},
}

ALL_SUBJECTS: tuple[str, ...] = ENTITY_RICH + ENTITY_LIGHT + NEUTRAL


def tier_of(subject: str) -> str:
    """Return the pre-registered tier for a subject string.

    Raises ``KeyError`` if the subject is not in the 20-subject
    inventory. We intentionally do not have a default tier — an
    unrecognized subject indicates a data drift that must be
    investigated, not silently bucketed.
    """
    return SUBJECT_TO_TIER[subject]


# Self-check: catches typos at import time.
assert len(ENTITY_RICH) == 8
assert len(ENTITY_LIGHT) == 8
assert len(NEUTRAL) == 4
assert len(SUBJECT_TO_TIER) == 20
assert len(set(SUBJECT_TO_TIER)) == 20
