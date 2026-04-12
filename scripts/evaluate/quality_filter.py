"""Deduplication and quality filtering for GutWise validated training data.

Three-stage pipeline:
  1. Quality filters — min length, required disclaimers, topic keyword presence
  2. Exact dedup — SHA256 of user message content
  3. Near dedup — Jaccard similarity on user+assistant tokens (keeps longer response)

Usage:
    # Run on validated data (default output: datasets/validated/train_deduped.jsonl)
    uv run python scripts/evaluate/quality_filter.py \
        --input datasets/validated/validated.jsonl

    # Custom threshold and output
    uv run python scripts/evaluate/quality_filter.py \
        --input datasets/validated/validated.jsonl \
        --jaccard-threshold 0.75 \
        --output datasets/validated/train_final.jsonl

    # Dry run — report only, no output file
    uv run python scripts/evaluate/quality_filter.py \
        --input datasets/validated/validated.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Minimum lengths (in characters)
MIN_USER_CHARS = 10
MIN_ASSISTANT_CHARS = 100
MAX_ASSISTANT_CHARS = 5000

# At least one disclaimer-like phrase should appear in assistant responses
DISCLAIMER_PATTERNS = [
    "healthcare provider",
    "healthcare professional",
    "healthcare team",
    "consult",
    "doctor",
    "physician",
    "medical professional",
    "seek medical",
    "medical advice",
    "medical care",
    "professional guidance",
    "professional support",
    "gastroenterologist",
    "registered dietitian",
    "dietitian",
    "speak with",
    "speak to",
]

# IBS-relevant keywords — at least one should appear in assistant responses
IBS_KEYWORDS = [
    "ibs",
    "irritable bowel",
    "bowel",
    "gut",
    "digestive",
    "abdominal",
    "bloating",
    "diarrhea",
    "constipation",
    "fodmap",
    "gastrointestinal",
    "stool",
    "cramp",
    "symptom",
    "diet",
    "fiber",
    "probiotic",
    "microbiome",
    "motility",
    "visceral",
    "serotonin",
    "gastro",
    "colon",
    "intestin",
    "laxativ",
    "antispasmodic",
    "peppermint",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_content(entry: dict, role: str) -> str:
    """Extract content for a given role from a ChatML entry."""
    for msg in entry.get("messages", []):
        if msg.get("role") == role:
            return msg.get("content", "")
    return ""


def get_all_assistant_content(entry: dict) -> str:
    """Concatenate all assistant messages (for multi-turn)."""
    parts = []
    for msg in entry.get("messages", []):
        if msg.get("role") == "assistant":
            parts.append(msg.get("content", ""))
    return "\n".join(parts)


def get_all_user_content(entry: dict) -> str:
    """Concatenate all user messages (for multi-turn)."""
    parts = []
    for msg in entry.get("messages", []):
        if msg.get("role") == "user":
            parts.append(msg.get("content", ""))
    return "\n".join(parts)


def tokenize(text: str) -> set[str]:
    """Simple whitespace tokenizer returning lowercase word set."""
    return set(text.lower().split())


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


# ---------------------------------------------------------------------------
# Stage 1: Quality filters
# ---------------------------------------------------------------------------


def quality_filter(entry: dict) -> str | None:
    """Run quality checks. Returns rejection reason or None if OK."""
    user_text = get_all_user_content(entry)
    assistant_text = get_all_assistant_content(entry)

    if len(user_text) < MIN_USER_CHARS:
        return f"user_too_short ({len(user_text)} chars)"

    if len(assistant_text) < MIN_ASSISTANT_CHARS:
        return f"assistant_too_short ({len(assistant_text)} chars)"

    if len(assistant_text) > MAX_ASSISTANT_CHARS:
        return f"assistant_too_long ({len(assistant_text)} chars)"

    # Check for IBS relevance
    assistant_lower = assistant_text.lower()
    has_ibs_keyword = any(kw in assistant_lower for kw in IBS_KEYWORDS)
    if not has_ibs_keyword:
        return "no_ibs_keywords"

    # Check for disclaimer (safety refusals should always have one)
    has_disclaimer = any(p in assistant_lower for p in DISCLAIMER_PATTERNS)
    if not has_disclaimer:
        return "no_disclaimer"

    return None


# ---------------------------------------------------------------------------
# Stage 2: Exact dedup
# ---------------------------------------------------------------------------


def exact_dedup(entries: list[dict]) -> tuple[list[dict], int]:
    """Remove exact duplicates by SHA256 of user content. Returns (deduped, removed_count)."""
    seen: set[str] = set()
    result = []
    removed = 0

    for entry in entries:
        user_text = get_all_user_content(entry)
        h = hashlib.sha256(user_text.encode()).hexdigest()
        if h in seen:
            removed += 1
        else:
            seen.add(h)
            result.append(entry)

    return result, removed


# ---------------------------------------------------------------------------
# Stage 3: Near dedup
# ---------------------------------------------------------------------------


def near_dedup(entries: list[dict], threshold: float = 0.80) -> tuple[list[dict], int]:
    """Remove near-duplicates by Jaccard similarity on user tokens.

    When two entries are similar, keeps the one with the longer assistant response.
    """
    n = len(entries)
    user_tokens = [tokenize(get_all_user_content(e)) for e in entries]
    assistant_lengths = [len(get_all_assistant_content(e)) for e in entries]

    to_remove: set[int] = set()

    for i in range(n):
        if i in to_remove:
            continue
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            sim = jaccard_similarity(user_tokens[i], user_tokens[j])
            if sim >= threshold:
                # Keep the entry with longer assistant response
                if assistant_lengths[i] < assistant_lengths[j]:
                    to_remove.add(i)
                    break
                else:
                    to_remove.add(j)

    result = [e for idx, e in enumerate(entries) if idx not in to_remove]
    return result, len(to_remove)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def load_entries(path: Path) -> list[dict]:
    """Load entries from a JSONL file."""
    entries = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def run_pipeline(
    input_path: Path,
    output_path: Path | None,
    jaccard_threshold: float,
    dry_run: bool,
) -> None:
    entries = load_entries(input_path)
    if not entries:
        print("No entries loaded.", file=sys.stderr)
        return

    original_count = len(entries)
    print(f"Loaded {original_count} entries from {input_path}\n")

    # Stage 1: Quality filters
    quality_passed = []
    quality_reasons: Counter[str] = Counter()
    for entry in entries:
        reason = quality_filter(entry)
        if reason:
            category = reason.split(" (")[0]
            quality_reasons[category] += 1
        else:
            quality_passed.append(entry)

    quality_removed = original_count - len(quality_passed)
    print(f"Stage 1 — Quality filters: {len(quality_passed)} passed, {quality_removed} removed")
    if quality_reasons:
        for reason, count in quality_reasons.most_common():
            print(f"  {reason}: {count}")

    # Stage 2: Exact dedup
    exact_passed, exact_removed = exact_dedup(quality_passed)
    print(f"Stage 2 — Exact dedup: {len(exact_passed)} kept, {exact_removed} duplicates removed")

    # Stage 3: Near dedup
    near_passed, near_removed = near_dedup(exact_passed, threshold=jaccard_threshold)
    print(
        f"Stage 3 — Near dedup (threshold={jaccard_threshold}): "
        f"{len(near_passed)} kept, {near_removed} near-duplicates removed"
    )

    # Topic distribution
    topic_counts: Counter[str] = Counter()
    conv_type_counts: Counter[str] = Counter()
    for entry in near_passed:
        meta = entry.get("metadata", {})
        topic_counts[meta.get("topic", "unknown")] += 1
        conv_type_counts[meta.get("conv_type", "unknown")] += 1

    print(f"\n{'=' * 50}")
    print(
        f"FINAL: {len(near_passed)} / {original_count} entries retained "
        f"({len(near_passed) / max(original_count, 1):.1%})"
    )
    print(f"{'=' * 50}")

    print("\nTopic distribution:")
    for topic, count in topic_counts.most_common():
        print(f"  {topic:25s} {count:4d}")

    print("\nConversation type distribution:")
    for ct, count in conv_type_counts.most_common():
        print(f"  {ct:25s} {count:4d}")

    # Write output
    if not dry_run and output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Strip internal metadata fields before writing
        clean_entries = []
        for entry in near_passed:
            clean = {k: v for k, v in entry.items() if not k.startswith("_")}
            clean_entries.append(clean)

        with output_path.open("w") as f:
            for entry in clean_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"\nWritten to: {output_path}")

        # Also write report
        report = {
            "original_count": original_count,
            "quality_removed": quality_removed,
            "quality_reasons": dict(quality_reasons.most_common()),
            "exact_duplicates_removed": exact_removed,
            "near_duplicates_removed": near_removed,
            "final_count": len(near_passed),
            "retention_rate": round(len(near_passed) / max(original_count, 1), 4),
            "jaccard_threshold": jaccard_threshold,
            "topic_distribution": dict(topic_counts.most_common()),
            "conv_type_distribution": dict(conv_type_counts.most_common()),
        }
        report_path = output_path.parent / "dedup_report.json"
        with report_path.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"Report:     {report_path}")
    elif dry_run:
        print("\n(dry run — no files written)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplication and quality filtering for GutWise training data"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file (e.g., datasets/validated/validated.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: datasets/validated/train_deduped.jsonl)",
    )
    parser.add_argument(
        "--jaccard-threshold",
        type=float,
        default=0.80,
        help="Jaccard similarity threshold for near-dedup (default: 0.80)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report only, don't write output files",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = PROJECT_ROOT / "datasets" / "validated" / "train_deduped.jsonl"

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        jaccard_threshold=args.jaccard_threshold,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
