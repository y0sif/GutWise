"""Filter IBS/GI-relevant entries from public HuggingFace medical datasets.

Downloads two HF datasets and extracts entries relevant to IBS and gastrointestinal health
using keyword matching, then reformats them to GutWise ChatML format.

Datasets:
  - lavita/ChatDoctor-HealthCareMagic-100k (Apache-2.0)
  - keivalya/MedQuad-MedicalQnADataset (public / academic)

Usage:
    uv run python scripts/collect/filter_hf_datasets.py
    uv run python scripts/collect/filter_hf_datasets.py --output-dir datasets/hf_filtered
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are GutWise, an IBS health education assistant. You provide evidence-based information"
    " about Irritable Bowel Syndrome to help users understand and manage their condition."
    " You are not a doctor and cannot diagnose or prescribe. Always recommend consulting a"
    " healthcare provider for personal medical decisions."
)

# Must match at least 1 primary keyword OR 2+ secondary keywords
PRIMARY_KEYWORDS = {"ibs", "irritable bowel", "fodmap", "bowel syndrome"}

SECONDARY_KEYWORDS = {
    "digestive",
    "gut",
    "abdominal",
    "bloating",
    "diarrhea",
    "constipation",
    "colon",
    "gastro",
    "bowel",
    "stool",
    "cramp",
    "nausea",
    "fiber",
    "probiotic",
    "laxative",
    "antispasmodic",
}

# Minimum character lengths for quality filtering
MIN_ANSWER_LEN = 100
MIN_QUESTION_LEN = 10

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "datasets" / "hf_filtered"
OUTPUT_FILE = OUTPUT_DIR / "hf_ibs_filtered.jsonl"


# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------


def _is_gi_relevant(question: str, answer: str) -> bool:
    """Return True if the entry is GI/IBS-relevant based on keyword matching.

    Matches if:
    - At least 1 primary keyword appears in question OR answer, OR
    - At least 2 secondary keywords appear across question + answer combined.
    """
    combined = (question + " " + answer).lower()

    # Check primary keywords (substring match)
    for kw in PRIMARY_KEYWORDS:
        if kw in combined:
            return True

    # Check secondary keywords — need 2+ matches
    secondary_count = sum(1 for kw in SECONDARY_KEYWORDS if kw in combined)
    return secondary_count >= 2


def _detect_topic(question: str, answer: str) -> str:
    """Heuristically detect the primary topic from question + answer text."""
    combined = (question + " " + answer).lower()

    topic_signals: list[tuple[str, list[str]]] = [
        ("fodmap", ["fodmap"]),
        ("diet", ["diet", "food", "eating", "fiber", "nutrition", "meal"]),
        (
            "symptoms",
            ["symptom", "bloating", "cramp", "diarrhea", "constipation", "nausea", "pain"],
        ),
        (
            "treatment",
            ["treatment", "medication", "medicine", "drug", "therapy", "antispasmodic", "laxative"],
        ),
        ("probiotics", ["probiotic", "bacteria", "microbiome", "gut flora"]),
        ("diagnosis", ["diagnos", "test", "colonoscopy", "endoscopy"]),
        ("lifestyle", ["stress", "exercise", "sleep", "lifestyle", "anxiety", "mental"]),
        ("pathophysiology", ["pathophysiology", "mechanism", "cause", "trigger", "gut-brain"]),
    ]

    for topic, signals in topic_signals:
        if any(sig in combined for sig in signals):
            return topic

    return "general_gi"


# ---------------------------------------------------------------------------
# Dataset-specific loaders and field extractors
# ---------------------------------------------------------------------------


def _extract_chatdoctor(row: dict) -> tuple[str, str] | None:
    """Extract (question, answer) from a ChatDoctor-HealthCareMagic row."""
    # Dataset schema: {"input": "...", "output": "...", "instruction": "..."}
    question = (row.get("input") or "").strip()
    answer = (row.get("output") or "").strip()

    if not question or not answer:
        return None
    return question, answer


def _extract_medquad(row: dict) -> tuple[str, str] | None:
    """Extract (question, answer) from a MedQuad row."""
    # Dataset schema: {"qtype": "...", "Question": "...", "Answer": "..."}
    question = (row.get("Question") or row.get("question") or "").strip()
    answer = (row.get("Answer") or row.get("answer") or "").strip()

    if not question or not answer:
        return None
    return question, answer


# ---------------------------------------------------------------------------
# Core filtering pipeline
# ---------------------------------------------------------------------------


def _passes_quality_filter(question: str, answer: str) -> bool:
    """Apply basic quality filters: minimum character lengths."""
    return len(question) >= MIN_QUESTION_LEN and len(answer) >= MIN_ANSWER_LEN


def _build_chatml_entry(
    question: str,
    answer: str,
    source_name: str,
    source_license: str,
) -> dict:
    """Build a GutWise ChatML-format entry."""
    topic = _detect_topic(question, answer)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "source": source_name,
            "source_license": source_license,
            "topic": topic,
            "conv_type": "factual_qa",
        },
    }


def filter_dataset(
    hf_path: str,
    split: str,
    extractor,
    source_name: str,
    source_license: str,
    trust_remote_code: bool = False,
) -> list[dict]:
    """Load a HuggingFace dataset, filter for GI relevance, and return ChatML entries."""
    logger.info("Loading %s (split=%s)...", hf_path, split)
    ds = load_dataset(hf_path, split=split, trust_remote_code=trust_remote_code)
    total = len(ds)
    logger.info("  Loaded %d rows", total)

    matched: list[dict] = []
    skipped_extraction = 0
    skipped_quality = 0
    skipped_relevance = 0

    for row in ds:
        pair = extractor(row)
        if pair is None:
            skipped_extraction += 1
            continue

        question, answer = pair

        if not _passes_quality_filter(question, answer):
            skipped_quality += 1
            continue

        if not _is_gi_relevant(question, answer):
            skipped_relevance += 1
            continue

        matched.append(_build_chatml_entry(question, answer, source_name, source_license))

    logger.info(
        "  Matched %d / %d  (skipped: %d extraction, %d quality, %d relevance)",
        len(matched),
        total,
        skipped_extraction,
        skipped_quality,
        skipped_relevance,
    )
    return matched


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASETS: list[dict] = [
    {
        "hf_path": "lavita/ChatDoctor-HealthCareMagic-100k",
        "split": "train",
        "extractor": _extract_chatdoctor,
        "source_name": "ChatDoctor-100k",
        "source_license": "Apache-2.0",
        "trust_remote_code": False,
    },
    {
        "hf_path": "keivalya/MedQuad-MedicalQnADataset",
        "split": "train",
        "extractor": _extract_medquad,
        "source_name": "MedQuad",
        "source_license": "academic-use",
        "trust_remote_code": False,
    },
]


def main(output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hf_ibs_filtered.jsonl"

    all_entries: list[dict] = []
    per_dataset_counts: dict[str, int] = {}

    for ds_cfg in DATASETS:
        name = ds_cfg["source_name"]
        logger.info("\n=== Processing: %s ===", name)
        entries = filter_dataset(
            hf_path=ds_cfg["hf_path"],
            split=ds_cfg["split"],
            extractor=ds_cfg["extractor"],
            source_name=name,
            source_license=ds_cfg["source_license"],
            trust_remote_code=ds_cfg["trust_remote_code"],
        )
        per_dataset_counts[name] = len(entries)
        all_entries.extend(entries)

    # Write output
    output_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in all_entries) + "\n",
        encoding="utf-8",
    )

    # Summary
    print("\n" + "=" * 60)
    print("HuggingFace IBS Filter — Summary")
    print("=" * 60)
    for name, count in per_dataset_counts.items():
        print(f"  {name:<35} {count:>5} entries")
    print("-" * 60)
    print(f"  {'TOTAL':<35} {len(all_entries):>5} entries")
    print("=" * 60)
    print(f"\nOutput written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter GI/IBS-relevant entries from public HuggingFace medical datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory to write filtered output (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    main(output_dir=args.output_dir)
