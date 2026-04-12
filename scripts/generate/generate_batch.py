"""Generate IBS health education Q&A prompts for Claude Code sub-agents.

Outputs prompts to stdout (no API calls). Claude Code spawns Haiku sub-agents
with the printed prompts, then results are validated back through this script.

Three modes:
    --info                          Show chunk stats
    --start N --count M             Output M prompts starting from chunk N
    --validate /path/to/results.jsonl  Validate sub-agent output

Usage:
    uv run python scripts/generate/generate_batch.py --info
    uv run python scripts/generate/generate_batch.py --start 0 --count 10
    uv run python scripts/generate/generate_batch.py --start 0 --count 10 --seed 42
    uv run python scripts/generate/generate_batch.py --validate /tmp/results.jsonl \
        --output datasets/generated/batch_0.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from scripts.generate.prompts import (
    CONV_TYPE_WEIGHTS,
    GENERATOR_SYSTEM_PROMPT,
    TRAINING_SYSTEM_PROMPT,
    build_prompt,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REFERENCE_DIR = PROJECT_ROOT / "reference"
CHUNKS_DIR = PROJECT_ROOT / "datasets" / "chunks"

REFERENCE_FILES = [
    REFERENCE_DIR / "ibs_reference.md",
    REFERENCE_DIR / "hallucination_blocklist.md",
    REFERENCE_DIR / "medication_whitelist.md",
]


# ---------------------------------------------------------------------------
# Reference context loading
# ---------------------------------------------------------------------------


def load_reference_context() -> str:
    """Load and concatenate all reference files into a single string."""
    sections = []
    for f in REFERENCE_FILES:
        if f.exists():
            sections.append(f"--- {f.name} ---\n{f.read_text()}")
        else:
            print(f"WARNING: Reference file not found: {f}", file=sys.stderr)
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Chunk loading
# ---------------------------------------------------------------------------


def load_chunks(chunks_dir: Path = CHUNKS_DIR) -> list[dict]:
    """Load all chunks from JSONL files, sorted by chunk_id for determinism."""
    chunks = []
    jsonl_files = sorted(chunks_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"ERROR: No .jsonl files found in {chunks_dir}", file=sys.stderr)
        sys.exit(1)

    for fp in jsonl_files:
        with fp.open() as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    chunks.append(chunk)
                except json.JSONDecodeError as e:
                    print(
                        f"WARNING: Skipping {fp.name} line {line_num}: {e}",
                        file=sys.stderr,
                    )

    # Deterministic ordering
    chunks.sort(key=lambda c: c.get("chunk_id", ""))
    return chunks


# ---------------------------------------------------------------------------
# Conversation type selection
# ---------------------------------------------------------------------------


def select_conv_type() -> str:
    """Randomly select a conversation type using the configured weights."""
    types = list(CONV_TYPE_WEIGHTS.keys())
    weights = list(CONV_TYPE_WEIGHTS.values())
    return random.choices(types, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# JSON output format instruction (replaces XML tag format)
# ---------------------------------------------------------------------------

_JSON_OUTPUT_INSTRUCTION = """\
You MUST output ONLY valid JSON — no markdown fencing, no commentary, no extra text.
Output exactly this structure:

{json_schema}

Rules:
- The "system" message content must be exactly: {system_prompt}
- For multi-turn conversations, include multiple user/assistant pairs in the messages array.
- All content fields must be non-empty strings.
- The assistant response must be at least 100 characters.
- Do not wrap the JSON in ```json``` or any other formatting.
"""

_SINGLE_TURN_SCHEMA = """\
{
  "messages": [
    {"role": "system", "content": "<system prompt>"},
    {"role": "user", "content": "<patient question>"},
    {"role": "assistant", "content": "<GutWise response>"}
  ],
  "metadata": {
    "chunk_id": "<chunk_id>",
    "topic": "<topic>",
    "conv_type": "<conv_type>",
    "source": "<source>",
    "source_license": "<license>"
  }
}"""

_MULTI_TURN_SCHEMA = """\
{
  "messages": [
    {"role": "system", "content": "<system prompt>"},
    {"role": "user", "content": "<patient question 1>"},
    {"role": "assistant", "content": "<GutWise response 1>"},
    {"role": "user", "content": "<follow-up question 2>"},
    {"role": "assistant", "content": "<GutWise response 2>"}
  ],
  "metadata": {
    "chunk_id": "<chunk_id>",
    "topic": "<topic>",
    "conv_type": "multi_turn",
    "source": "<source>",
    "source_license": "<license>"
  }
}"""


def build_json_instruction(chunk: dict, conv_type: str) -> str:
    """Build the JSON output format instruction with metadata pre-filled."""
    schema = _MULTI_TURN_SCHEMA if conv_type == "multi_turn" else _SINGLE_TURN_SCHEMA

    # Pre-fill metadata values so the sub-agent gets them right
    schema = schema.replace("<chunk_id>", chunk.get("chunk_id", "unknown"))
    schema = schema.replace("<topic>", chunk.get("topic", chunk.get("section", "")))
    schema = schema.replace("<conv_type>", conv_type)
    schema = schema.replace("<source>", chunk.get("source", ""))
    schema = schema.replace("<license>", chunk.get("source_license", chunk.get("license", "")))
    schema = schema.replace("<system prompt>", TRAINING_SYSTEM_PROMPT)

    return _JSON_OUTPUT_INSTRUCTION.format(
        json_schema=schema,
        system_prompt=repr(TRAINING_SYSTEM_PROMPT),
    )


# ---------------------------------------------------------------------------
# Mode: --info
# ---------------------------------------------------------------------------


def show_info() -> None:
    """Print chunk statistics."""
    chunks = load_chunks()
    topic_counts: Counter[str] = Counter()
    for c in chunks:
        topic_counts[c.get("topic", c.get("section", "unknown"))] += 1

    print(f"Total chunks: {len(chunks)}")
    print(f"Topics: {len(topic_counts)}")
    print()
    for topic, count in topic_counts.most_common():
        print(f"  {topic:30s} {count:4d}")
    print()
    print("Conversation type weights:")
    for ct, w in CONV_TYPE_WEIGHTS.items():
        print(f"  {ct:20s} {w:.0%}")


# ---------------------------------------------------------------------------
# Mode: --start N --count M (prompt output)
# ---------------------------------------------------------------------------


def output_prompts(start: int, count: int, seed: int | None = None) -> None:
    """Print generation prompts to stdout, delimited by markers."""
    if seed is not None:
        random.seed(seed)

    chunks = load_chunks()
    reference_context = load_reference_context()
    if not reference_context:
        print("ERROR: No reference context loaded — cannot generate safely", file=sys.stderr)
        sys.exit(1)

    end = min(start + count, len(chunks))
    if start >= len(chunks):
        print(
            f"ERROR: --start {start} is beyond chunk count ({len(chunks)})",
            file=sys.stderr,
        )
        sys.exit(1)

    selected = chunks[start:end]
    print(f"# Outputting {len(selected)} prompts (chunks {start}-{end - 1})", file=sys.stderr)

    for idx, chunk in enumerate(selected, start=start):
        conv_type = select_conv_type()
        chunk_text = chunk.get("text", chunk.get("content", ""))
        topic = chunk.get("topic", chunk.get("section", ""))
        chunk_id = chunk.get("chunk_id", "unknown")

        if not chunk_text:
            print(f"WARNING: Chunk {chunk_id} has no text, skipping", file=sys.stderr)
            continue

        # Build the generation prompt from prompts.py templates
        base_prompt = build_prompt(conv_type, chunk_text, reference_context)

        # Append JSON output instruction
        json_instruction = build_json_instruction(chunk, conv_type)
        full_prompt = (
            f"{GENERATOR_SYSTEM_PROMPT}\n\n"
            f"{base_prompt}\n\n"
            f"--- OUTPUT FORMAT ---\n"
            f"{json_instruction}"
        )

        # Print with delimiters
        print(f"=== CHUNK {idx} | {topic} | {conv_type} | {chunk_id} ===")
        print(full_prompt)
        print("=== END ===")


# ---------------------------------------------------------------------------
# Mode: --validate
# ---------------------------------------------------------------------------


def validate_entry(entry: dict) -> str | None:
    """Validate a single generated entry. Returns error string or None if valid."""
    if not isinstance(entry, dict):
        return "not a dict"

    messages = entry.get("messages")
    if not isinstance(messages, list):
        return "missing or invalid 'messages' key"
    if len(messages) < 2:
        return f"messages has {len(messages)} items, need >= 2"

    roles = {m.get("role") for m in messages if isinstance(m, dict)}
    if "user" not in roles:
        return "no 'user' role in messages"
    if "assistant" not in roles:
        return "no 'assistant' role in messages"

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return f"message {i} is not a dict"
        content = msg.get("content", "")
        if not content or not isinstance(content, str) or not content.strip():
            return f"message {i} has empty content"

    # Check assistant response length
    for msg in messages:
        if msg.get("role") == "assistant" and len(msg.get("content", "")) < 100:
            return (
                f"assistant response too short ({len(msg.get('content', ''))} chars, need >= 100)"
            )

    metadata = entry.get("metadata")
    if not isinstance(metadata, dict):
        return "missing or invalid 'metadata' dict"

    return None


def validate_results(input_path: Path, output_path: Path) -> None:
    """Validate a JSONL file of sub-agent results and save valid entries."""
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid = []
    invalid = []
    total = 0

    with input_path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                invalid.append((line_num, f"JSON parse error: {e}"))
                continue

            error = validate_entry(entry)
            if error:
                chunk_id = "?"
                if isinstance(entry, dict):
                    chunk_id = entry.get("metadata", {}).get("chunk_id", "?")
                invalid.append((line_num, f"[{chunk_id}] {error}"))
            else:
                valid.append(entry)

    # Append valid entries to output
    if valid:
        with output_path.open("a") as f:
            for entry in valid:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Print summary
    print("\nValidation Summary")
    print(f"  Total entries:   {total}")
    print(f"  Valid:           {len(valid)}")
    print(f"  Invalid:         {len(invalid)}")
    print(f"  Pass rate:       {len(valid) / total:.1%}" if total else "  Pass rate:       N/A")
    if valid:
        print(f"  Appended to:     {output_path}")
    if invalid:
        print("\nInvalid entries:")
        for line_num, reason in invalid:
            print(f"  line {line_num}: {reason}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate IBS Q&A prompts for Claude Code sub-agents (no API calls)."
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--info",
        action="store_true",
        help="Show chunk statistics",
    )
    mode.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start index for prompt output (requires --count)",
    )
    mode.add_argument(
        "--validate",
        type=Path,
        default=None,
        help="Path to results JSONL to validate",
    )

    p.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of prompts to output (required with --start)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible conversation type selection",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "generated" / "batch_0.jsonl",
        help="Output path for validated results (default: datasets/generated/batch_0.jsonl)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.info:
        show_info()
    elif args.start is not None:
        if args.count is None:
            print("ERROR: --count is required with --start", file=sys.stderr)
            sys.exit(1)
        output_prompts(start=args.start, count=args.count, seed=args.seed)
    elif args.validate is not None:
        validate_results(input_path=args.validate, output_path=args.output)


if __name__ == "__main__":
    main()
