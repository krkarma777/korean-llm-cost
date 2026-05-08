"""ECPC snapshot — Effective Cost per 1K Korean characters, 7 models × 3 categories.

ECPC translates the unit-less TPC into actual USD cost, exposing
how the cluster gap in tokenization interacts with each provider's
pricing tier. The cluster gap measured in TPC may be larger or
smaller in ECPC depending on which model sits in which cluster.

Formula
-------
    ECPC ($/1K Korean chars) = TPC × price_per_1M_input_tokens / 1000

Equivalently, TPC × price_per_1K_input_tokens (since TPC is
tokens/char and we want cost-per-1K-chars).

rECPC-GPT = ECPC(M, C) / ECPC(GPT-4o, C) is the cost-domain analog
of rTPC-GPT, anchored to GPT-4o on the same corpus.

Data flow
---------
- TPC values: read from the existing per-category aggregate CSVs
  (no new tokenization).
- Prices: hardcoded in this file (snapshot 2026-05-08), each with
  source URL + access date + verification status.
- 1 model (EXAONE 3.5 7.8B) is UNVERIFIED — placeholder $0.20/1M.
  See notes/12_metric_definitions.md §Pricing snapshot for
  rationale and Future Work caveat.

Outputs
-------
- ``results/raw/09_ecpc_snapshot.csv`` — long-format,
  one row per (model, category) with TPC, price, ECPC, rECPC-GPT,
  and the source-verification fields.

Note on "rECPC-GPT" vs "KPR/GPT" naming
---------------------------------------
This script uses "rECPC-GPT" (the paper-facing name) directly.
The legacy "KPR/GPT" label is retained only in pre-registration
artifacts (``medical_predictions.classify_kpr_gpt``); see
``notes/12_metric_definitions.md`` for the cross-reference.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ----- Pricing snapshot (2026-05-08) -----
#
# ALL PRICES IN USD per 1,000,000 INPUT tokens.
#
# Sources verified by external research on 2026-05-08 against each
# provider's official documentation. One model (EXAONE 3.5 7.8B)
# could not be verified — placeholder used; clearly marked.

PRICING_2026_05_08: dict[str, dict] = {
    "gpt-4o": {
        "model_name_published": "gpt-4o",
        "price_usd_per_1m_input": 2.50,
        "source_url": "https://openai.com/api/pricing/",
        "access_date": "2026-05-08",
        "host": "OpenAI direct API",
        "verification": "VERIFIED",
        "notes": ("Standard tier; cached input $1.25/1M, Batch API 50% off. "
                  "Price stable since 2024 launch."),
    },
    "claude-sonnet-4-5": {
        "model_name_published": "claude-sonnet-4-5 (claude-sonnet-4-5-20250929)",
        "price_usd_per_1m_input": 3.00,
        "source_url": "https://platform.claude.com/docs/en/about-claude/pricing",
        "access_date": "2026-05-08",
        "host": "Anthropic direct API",
        "verification": "VERIFIED",
        "notes": ("Prompts <=200K tokens. Surcharge applies above 200K "
                  "(input becomes $6/1M)."),
    },
    "gemini-2.5-flash": {
        "model_name_published": "gemini-2.5-flash",
        "price_usd_per_1m_input": 0.30,
        "source_url": "https://ai.google.dev/gemini-api/docs/pricing",
        "access_date": "2026-05-08",
        "host": "Google AI / Gemini API (paid tier)",
        "verification": "VERIFIED",
        "notes": ("Text/image/video input; audio input is $1.00/1M. "
                  "Batch tier is half."),
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "model_name_published": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "price_usd_per_1m_input": 0.18,
        "source_url": "https://www.together.ai/pricing",
        "access_date": "2026-05-08",
        "host": "Together AI (serverless, Turbo tier)",
        "verification": "VERIFIED",
        "notes": ("Together also offers a cheaper 'Llama 3 8B Instruct Lite' "
                  "at $0.10/1M, but that is a different SKU "
                  "(lower-precision quantization)."),
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "model_name_published": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "price_usd_per_1m_input": 0.30,
        "source_url": "https://www.together.ai/pricing",
        "access_date": "2026-05-08",
        "host": "Together AI (serverless, Turbo tier)",
        "verification": "VERIFIED",
        "notes": ("Verified directly from Together AI pricing page fetch on "
                  "2026-05-08."),
    },
    "upstage/SOLAR-10.7B-Instruct-v1.0": {
        "model_name_published": ("Solar Mini (current commercial SKU; "
                                  "SOLAR-10.7B-Instruct-v1.0 is no longer "
                                  "separately priced)"),
        "price_usd_per_1m_input": 0.15,
        "source_url": "https://www.upstage.ai/pricing",
        "access_date": "2026-05-08",
        "host": "Upstage Console (direct)",
        "verification": "PROXY",
        "notes": ("Upstage no longer lists the open-weights v1.0 separately; "
                  "Solar Mini / Pro 2 / Pro 3 all at $0.15/1M input on the "
                  "current pricing page. Used as the closest verifiable proxy."),
    },
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct": {
        "model_name_published": "UNVERIFIED",
        "price_usd_per_1m_input": 0.20,  # PLACEHOLDER, NOT AUTHORITATIVE
        "source_url": "https://friendli.ai/docs/guides/serverless_endpoints/pricing",
        "access_date": "2026-05-08",
        "host": "Friendli AI (placeholder; 7.8B SKU not separately listed)",
        "verification": "UNVERIFIED",
        "notes": ("Friendli's serverless table only lists K-EXAONE-236B-A23B "
                  "at $0.20/1M input as of 2026-05-08; the 3.5-7.8B variant "
                  "is not separately priced and Artificial Analysis no longer "
                  "tracks it under Friendli. Placeholder $0.20 borrowed from "
                  "Friendli's listed EXAONE-family input rate. Alternative "
                  "placeholder: $0.18 (Together's Llama-3.1-8B-Turbo, "
                  "comparable 7-8B hosted model). Confirm via direct contact "
                  "with Friendli before publishing."),
    },
}


# ----- TPC sources: existing aggregate CSVs (no new tokenization) -----

CATEGORY_AGGREGATE = {
    "news":         "02_tpc_news_n1000_aggregate.csv",
    "conversation": "03_tpc_conversation_n1000_aggregate.csv",
    "medical":      "07_tpc_medical_n1000_aggregate.csv",
}

MODEL_ORDER = (
    "EXAONE 3.5 7.8B",
    "Gemini 2.5 Flash",
    "GPT-4o",
    "Llama 3.1 8B",
    "Qwen 2.5 7B",
    "Claude Sonnet 4.5",
    "Solar 10.7B",
)


def load_tpc_table() -> dict[tuple[str, str], tuple[str, float]]:
    """Read all per-category aggregate CSVs and return
    ``(category, model_label) -> (pricing_key, tpc)``.

    The ``pricing_key`` is the ``model_name`` field from the CSV
    (e.g., "gpt-4o", "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"), which
    matches the keys in PRICING_2026_05_08. The CSV's ``version``
    field is the model release date / git revision and is NOT used
    for pricing lookup.
    """
    raw_dir = ROOT / "results" / "raw"
    out: dict[tuple[str, str], tuple[str, float]] = {}
    for category, fname in CATEGORY_AGGREGATE.items():
        path = raw_dir / fname
        if not path.exists():
            print(f"[error] missing aggregate CSV: {path}", file=sys.stderr)
            sys.exit(1)
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out[(category, row["model_label"])] = (
                    row["model_name"], float(row["tpc"])
                )
    return out


def main() -> None:
    out_dir = ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("ECPC snapshot — 7 models × 3 categories, 2026-05-08")
    print("=" * 76)

    tpc_table = load_tpc_table()

    # Verify pricing keys match all model versions seen in the TPC table.
    seen_versions = {v for (_, _), (v, _) in tpc_table.items()}
    missing_prices = seen_versions - set(PRICING_2026_05_08)
    if missing_prices:
        print(f"[error] PRICING_2026_05_08 missing entries for: {missing_prices}",
              file=sys.stderr)
        sys.exit(1)

    # ----- Compute ECPC per (model, category) -----

    rows: list[dict] = []
    for category in CATEGORY_AGGREGATE:
        # Find GPT-4o ECPC for this category for rECPC-GPT.
        gpt_v, gpt_tpc = tpc_table[(category, "GPT-4o")]
        gpt_price = PRICING_2026_05_08[gpt_v]["price_usd_per_1m_input"]
        gpt_ecpc = gpt_tpc * gpt_price / 1000.0

        for model in MODEL_ORDER:
            if (category, model) not in tpc_table:
                print(f"[warn] missing TPC for ({category}, {model})")
                continue
            version, tpc = tpc_table[(category, model)]
            entry = PRICING_2026_05_08[version]
            price = entry["price_usd_per_1m_input"]
            ecpc = tpc * price / 1000.0
            r_ecpc_gpt = ecpc / gpt_ecpc
            rows.append({
                "category": category,
                "model_label": model,
                "model_version": version,
                "tpc": round(tpc, 4),
                "price_usd_per_1m_input": price,
                "ecpc_usd_per_1k_chars": round(ecpc, 6),
                "recpc_gpt": round(r_ecpc_gpt, 4),
                "host": entry["host"],
                "verification": entry["verification"],
                "source_url": entry["source_url"],
                "access_date": entry["access_date"],
            })

    # ----- Print human-readable matrix -----

    print(f"\nECPC per (model, category)  [USD per 1,000 Korean chars]")
    print(f"  Pricing snapshot: 2026-05-08; 1 model UNVERIFIED (EXAONE).")
    print("-" * 76)
    header = f"{'Model':<22} {'price$/1M':>10} " + " ".join(f"{c:>11}" for c in CATEGORY_AGGREGATE)
    print(header)
    print("-" * len(header))
    for model in MODEL_ORDER:
        # Take any category to retrieve the model's price (price doesn't
        # depend on category).
        version = None
        for (cat, m), (v, _) in tpc_table.items():
            if m == model:
                version = v
                break
        if version is None:
            continue
        price = PRICING_2026_05_08[version]["price_usd_per_1m_input"]
        flag = "?" if PRICING_2026_05_08[version]["verification"] == "UNVERIFIED" else " "
        cells = []
        for category in CATEGORY_AGGREGATE:
            row = next((r for r in rows
                        if r["model_label"] == model and r["category"] == category),
                       None)
            if row is None:
                cells.append(f"{'-':>11}")
            else:
                cells.append(f"${row['ecpc_usd_per_1k_chars']:>9.4f}")
        print(f"{model:<22} ${price:>8.2f}{flag} " + " ".join(cells))

    print(f"\nrECPC-GPT (cost-domain analog of rTPC-GPT)  [unitless]")
    print("-" * 76)
    header = f"{'Model':<22} " + " ".join(f"{c:>11}" for c in CATEGORY_AGGREGATE)
    print(header)
    print("-" * len(header))
    for model in MODEL_ORDER:
        cells = []
        for category in CATEGORY_AGGREGATE:
            row = next((r for r in rows
                        if r["model_label"] == model and r["category"] == category),
                       None)
            if row is None:
                cells.append(f"{'-':>11}")
            else:
                cells.append(f"{row['recpc_gpt']:>10.3f}×")
        print(f"{model:<22} " + " ".join(cells))

    # ----- Cluster gap in ECPC vs in TPC -----

    print(f"\nCluster gap evolution: TPC-gap vs ECPC-gap by category")
    print(f"  (ratio of penalty-cluster cost to efficient-cluster baseline,")
    print(f"   computed against GPT-4o for both metrics)")
    print("-" * 76)
    print(f"{'Category':<14} {'rTPC max':>10} {'rTPC ↔ rECPC of penalty cluster':>36}")
    print("-" * 76)
    for category in CATEGORY_AGGREGATE:
        cat_rows = [r for r in rows if r["category"] == category]
        if not cat_rows:
            continue
        # rTPC-GPT for this category from aggregate (pull from tpc table directly)
        gpt_v, gpt_tpc = tpc_table[(category, "GPT-4o")]
        for r in cat_rows:
            r["rtpc_gpt"] = r["tpc"] / gpt_tpc
        rtpc_max = max(r["rtpc_gpt"] for r in cat_rows)
        recpc_max = max(r["recpc_gpt"] for r in cat_rows)
        # who's at the top
        top_rtpc = next(r["model_label"] for r in cat_rows
                        if r["rtpc_gpt"] == rtpc_max)
        top_recpc = next(r["model_label"] for r in cat_rows
                         if r["recpc_gpt"] == recpc_max)
        print(f"{category:<14} {rtpc_max:>9.2f}×  "
              f"penalty-cluster top by rTPC: {top_rtpc} ({rtpc_max:.2f}×) | "
              f"by rECPC: {top_recpc} ({recpc_max:.2f}×)")

    # ----- Write CSV -----

    out_csv = out_dir / "09_ecpc_snapshot.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "category", "model_label", "model_version",
            "tpc", "price_usd_per_1m_input",
            "ecpc_usd_per_1k_chars", "recpc_gpt",
            "host", "verification", "source_url", "access_date",
        ])
        w.writeheader()
        for row in rows:
            row.pop("rtpc_gpt", None)  # transient, not in CSV schema
            w.writerow(row)
    print(f"\n  → {out_csv.relative_to(ROOT)}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
