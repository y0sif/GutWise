"""Chunk IBS source texts into topic-based segments for downstream Q&A generation.

Reads plain-text source files (with YAML frontmatter) from datasets/sources/,
classifies sections by medical topic using keyword matching, and writes
JSONL chunk files to datasets/chunks/ — one file per topic.

Usage:
    uv run python scripts/chunk/chunk_sources.py
    uv run python scripts/chunk/chunk_sources.py \
        --input-dir datasets/sources --output-dir datasets/chunks
    uv run python scripts/chunk/chunk_sources.py --min-words 150 --max-words 1000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Topic definitions & keyword mappings
# ---------------------------------------------------------------------------

TOPICS = [
    "diagnosis",
    "epidemiology",
    "pathophysiology",
    "diet",
    "pharmacology",
    "psychological",
    "lifestyle",
    "red_flags",
    "patient_education",
    "general",
]

# Each list is checked case-insensitively against section header + body text.
# Order matters: first match wins, so more specific topics come first.
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "diagnosis": [
        "rome iv",
        "rome criteria",
        "rome 4",
        "diagnos",
        "diagnostic",
        "subtype",
        "ibs-d",
        "ibs-c",
        "ibs-m",
        "ibs-u",
        "mixed ibs",
        "classification",
        "bristol stool",
        "bristol scale",
        "differential diagnosis",
        "celiac",
        "colonoscopy",
        "blood test",
        "stool test",
        "exclusion criteria",
        "screening",
        "biomarker",
    ],
    "red_flags": [
        "red flag",
        "warning sign",
        "alarm feature",
        "alarm symptom",
        "blood in stool",
        "rectal bleeding",
        "unexplained weight loss",
        "weight loss",
        "emergency",
        "urgent",
        "cancer",
        "colorectal cancer",
        "inflammatory bowel disease",
        "ibd",
        "fever",
        "night sweat",
        "family history of cancer",
        "onset after 50",
        "anemia",
        "when to see a doctor",
        "seek medical",
    ],
    "pharmacology": [
        "medication",
        "drug",
        "antispasmodic",
        "loperamide",
        "linaclotide",
        "lubiprostone",
        "eluxadoline",
        "rifaximin",
        "alosetron",
        "amitriptyline",
        "tricyclic",
        "ssri",
        "prescription",
        "dose",
        "dosage",
        "pharmaceutical",
        "peppermint oil",
        "antidiarrheal",
        "laxative",
        "osmotic laxative",
        "polyethylene glycol",
        "tegaserod",
        "plecanatide",
        "prucalopride",
        "antidepressant",
        "neuromodulator",
        "pharmacolog",
        "pharmacother",
        "otc",
        "over-the-counter",
        "bile acid",
    ],
    "diet": [
        "fodmap",
        "low fodmap",
        "diet",
        "dietary",
        "food",
        "eating",
        "fiber",
        "soluble fiber",
        "insoluble fiber",
        "lactose",
        "fructose",
        "gluten",
        "meal",
        "nutrition",
        "nutritional",
        "trigger food",
        "food intolerance",
        "elimination diet",
        "reintroduction",
        "fermentable",
        "oligosaccharide",
        "disaccharide",
        "monosaccharide",
        "polyol",
        "probiotic",
        "prebiotic",
        "bloating",
        "gas",
        "flatulence",
        "dietitian",
    ],
    "psychological": [
        "cbt",
        "cognitive behav",
        "cognitive-behav",
        "psychotherapy",
        "therapy",
        "hypnotherapy",
        "gut-directed hypnotherapy",
        "stress management",
        "stress",
        "anxiety",
        "depression",
        "mindfulness",
        "meditation",
        "relaxation",
        "psychological",
        "psychosocial",
        "mental health",
        "brain-gut psychotherapy",
        "catastrophizing",
        "coping",
        "biofeedback",
    ],
    "pathophysiology": [
        "gut-brain",
        "brain-gut",
        "gut brain axis",
        "visceral hypersensitiv",
        "visceral pain",
        "microbiome",
        "microbiota",
        "gut flora",
        "motility",
        "dysmotility",
        "serotonin",
        "5-ht",
        "nerve",
        "enteric nervous",
        "central sensitization",
        "peripheral sensitization",
        "inflammation",
        "low-grade inflammation",
        "mast cell",
        "intestinal permeability",
        "leaky gut",
        "post-infectious",
        "pathophysiology",
        "pathogenesis",
        "mechanism",
        "etiology",
        "aetiology",
    ],
    "epidemiology": [
        "prevalence",
        "incidence",
        "population",
        "demographic",
        "risk factor",
        "age",
        "gender",
        "sex difference",
        "epidemiol",
        "global burden",
        "healthcare cost",
        "economic burden",
        "comorbid",
        "co-morbid",
    ],
    "lifestyle": [
        "exercise",
        "physical activity",
        "yoga",
        "sleep",
        "sleep hygiene",
        "symptom diary",
        "food diary",
        "routine",
        "daily management",
        "self-management",
        "self-care",
        "lifestyle",
        "hydration",
        "water intake",
        "smoking",
        "alcohol",
    ],
    "patient_education": [
        "what is ibs",
        "functional disorder",
        "functional gastrointestinal",
        "chronic condition",
        "quality of life",
        "prognosis",
        "not dangerous",
        "benign condition",
        "patient education",
        "living with ibs",
        "long-term",
        "long term outlook",
        "misconception",
        "myth",
        "support group",
        "explain",
        "understanding ibs",
        "faq",
    ],
}

# Minimum keyword hits needed to classify as a topic (vs. general).
MIN_KEYWORD_HITS = 2

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SourceMeta:
    source: str = "Unknown"
    url: str = ""
    license: str = ""
    fetched: str = ""


@dataclass
class Chunk:
    chunk_id: str
    topic: str
    source: str
    source_url: str
    license: str
    text: str
    word_count: int


@dataclass
class Section:
    """A contiguous block of text under one heading (or the top of the file)."""

    heading: str
    body: str
    word_count: int = 0


@dataclass
class TopicCounter:
    counts: dict[str, int] = field(default_factory=lambda: {t: 0 for t in TOPICS})

    def next_id(self, topic: str, source_slug: str) -> str:
        self.counts[topic] += 1
        return f"{source_slug}_{topic}_{self.counts[topic]:03d}"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_frontmatter(text: str) -> tuple[SourceMeta, str]:
    """Extract YAML frontmatter and return (metadata, body_text)."""
    pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    match = pattern.match(text)
    if not match:
        return SourceMeta(), text

    yaml_block = match.group(1)
    body = text[match.end() :]

    meta = SourceMeta()
    for line in yaml_block.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip().strip("\"'")
        if key == "source":
            meta.source = value
        elif key == "url":
            meta.url = value
        elif key == "license":
            meta.license = value
        elif key == "fetched":
            meta.fetched = value

    return meta, body


def split_into_sections(text: str) -> list[Section]:
    """Split body text into sections delimited by markdown-style headers or
    uppercase section headers followed by a newline."""
    header_re = re.compile(
        r"^(?:#{1,4}\s+.+|[A-Z][A-Z\s,/&-]{4,})\s*$",
        re.MULTILINE,
    )

    positions: list[tuple[int, str]] = []
    for m in header_re.finditer(text):
        positions.append((m.start(), m.group().strip().lstrip("#").strip()))

    if not positions:
        # No headers found — treat entire text as one section.
        body = text.strip()
        return [Section(heading="", body=body, word_count=len(body.split()))] if body else []

    sections: list[Section] = []

    # Text before the first header.
    preamble = text[: positions[0][0]].strip()
    if preamble:
        sections.append(Section(heading="", body=preamble, word_count=len(preamble.split())))

    for i, (start, heading) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        # Body starts after the header line.
        header_end = text.index("\n", start) + 1 if "\n" in text[start:] else len(text)
        body = text[header_end:end].strip()
        if body:
            sections.append(Section(heading=heading, body=body, word_count=len(body.split())))

    return sections


# ---------------------------------------------------------------------------
# Topic classification
# ---------------------------------------------------------------------------


def classify_topic(heading: str, body: str) -> str:
    """Classify a section into one of the IBS topics using keyword matching.

    Returns the topic with the most keyword hits (minimum MIN_KEYWORD_HITS),
    or 'general' if no topic reaches the threshold.
    """
    combined = f"{heading}\n{body}".lower()

    scores: dict[str, int] = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in combined)
        if score >= MIN_KEYWORD_HITS:
            scores[topic] = score

    if not scores:
        return "general"

    return max(scores, key=lambda t: scores[t])


# ---------------------------------------------------------------------------
# Chunking logic
# ---------------------------------------------------------------------------


def split_section_into_paragraphs(section: Section) -> list[str]:
    """Split a section body into paragraphs (double-newline delimited)."""
    paragraphs = re.split(r"\n\s*\n", section.body)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_sections(
    sections: list[Section],
    min_words: int,
    max_words: int,
) -> list[tuple[str, str]]:
    """Convert sections into (topic, text) pairs respecting word limits.

    Strategy:
    - Classify each section by topic.
    - If a section is within [min_words, max_words], emit it as-is.
    - If too long, split at paragraph boundaries into sub-chunks.
    - If too short, try to merge with the next section if same topic.
    """
    classified: list[tuple[str, Section]] = []
    for sec in sections:
        topic = classify_topic(sec.heading, sec.body)
        classified.append((topic, sec))

    chunks: list[tuple[str, str]] = []
    i = 0

    while i < len(classified):
        topic, sec = classified[i]
        text = f"{sec.heading}\n\n{sec.body}".strip() if sec.heading else sec.body
        wc = len(text.split())

        if min_words <= wc <= max_words:
            chunks.append((topic, text))
            i += 1
            continue

        if wc > max_words:
            # Split at paragraph boundaries.
            paragraphs = split_section_into_paragraphs(sec)
            buf: list[str] = []
            if sec.heading:
                buf.append(sec.heading)
            buf_wc = len(sec.heading.split()) if sec.heading else 0

            for para in paragraphs:
                para_wc = len(para.split())
                if buf_wc + para_wc > max_words and buf_wc >= min_words:
                    chunks.append((topic, "\n\n".join(buf)))
                    buf = []
                    buf_wc = 0
                buf.append(para)
                buf_wc += para_wc

            if buf:
                # If remainder is too small, attach to last chunk of same topic or emit anyway.
                if buf_wc < min_words and chunks and chunks[-1][0] == topic:
                    prev_topic, prev_text = chunks.pop()
                    merged = prev_text + "\n\n" + "\n\n".join(buf)
                    chunks.append((prev_topic, merged))
                else:
                    chunks.append((topic, "\n\n".join(buf)))
            i += 1
            continue

        # wc < min_words — try merging with subsequent same-topic sections.
        merged_text = text
        merged_wc = wc
        j = i + 1
        while j < len(classified) and merged_wc < min_words:
            next_topic, next_sec = classified[j]
            if next_topic != topic:
                break
            addition = (
                f"{next_sec.heading}\n\n{next_sec.body}".strip()
                if next_sec.heading
                else next_sec.body
            )
            merged_text += "\n\n" + addition
            merged_wc = len(merged_text.split())
            j += 1

        chunks.append((topic, merged_text))
        i = j

    return chunks


# ---------------------------------------------------------------------------
# Source slug
# ---------------------------------------------------------------------------


def make_slug(source_name: str) -> str:
    """Turn a source name into a filesystem-safe lowercase slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", source_name.lower())
    return slug.strip("_") or "unknown"


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process_source_file(
    path: Path,
    min_words: int,
    max_words: int,
    counter: TopicCounter,
) -> list[Chunk]:
    """Read a single source file and return classified chunks."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  WARN: Could not read {path}: {e}", file=sys.stderr)
        return []

    if not text.strip():
        print(f"  WARN: Empty file {path}", file=sys.stderr)
        return []

    meta, body = parse_frontmatter(text)
    if not body.strip():
        print(f"  WARN: No body text in {path}", file=sys.stderr)
        return []

    slug = make_slug(meta.source)
    sections = split_into_sections(body)
    topic_chunks = chunk_sections(sections, min_words, max_words)

    results: list[Chunk] = []
    for topic, chunk_text in topic_chunks:
        wc = len(chunk_text.split())
        chunk_id = counter.next_id(topic, slug)
        results.append(
            Chunk(
                chunk_id=chunk_id,
                topic=topic,
                source=meta.source,
                source_url=meta.url,
                license=meta.license,
                text=chunk_text,
                word_count=wc,
            )
        )

    return results


def write_chunks(chunks: list[Chunk], output_dir: Path) -> None:
    """Write chunks to per-topic JSONL files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by topic.
    by_topic: dict[str, list[Chunk]] = {t: [] for t in TOPICS}
    for chunk in chunks:
        by_topic[chunk.topic].append(chunk)

    for topic in TOPICS:
        topic_chunks = by_topic[topic]
        if not topic_chunks:
            continue
        out_path = output_dir / f"{topic}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for chunk in topic_chunks:
                line = json.dumps(asdict(chunk), ensure_ascii=False)
                f.write(line + "\n")


def print_summary(chunks: list[Chunk]) -> None:
    """Print summary statistics to stdout."""
    if not chunks:
        print("\nNo chunks produced.")
        return

    by_topic: dict[str, list[Chunk]] = {t: [] for t in TOPICS}
    for chunk in chunks:
        by_topic[chunk.topic].append(chunk)

    total_words = sum(c.word_count for c in chunks)

    print("\n" + "=" * 60)
    print("Chunking Summary")
    print("=" * 60)
    print(f"{'Topic':<22} {'Chunks':>8} {'Avg Words':>10}")
    print("-" * 42)

    for topic in TOPICS:
        topic_chunks = by_topic[topic]
        if not topic_chunks:
            continue
        avg_wc = sum(c.word_count for c in topic_chunks) / len(topic_chunks)
        print(f"{topic:<22} {len(topic_chunks):>8} {avg_wc:>10.0f}")

    print("-" * 42)
    avg_total = total_words / len(chunks)
    print(f"{'TOTAL':<22} {len(chunks):>8} {avg_total:>10.0f}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk IBS source texts into topic-based segments.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("datasets/sources"),
        help="Directory containing source .txt files (default: datasets/sources)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/chunks"),
        help="Directory for output JSONL chunk files (default: datasets/chunks)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=200,
        help="Minimum words per chunk (default: 200)",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=800,
        help="Maximum words per chunk (default: 800)",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    min_words: int = args.min_words
    max_words: int = args.max_words

    if not input_dir.is_dir():
        print(f"ERROR: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    source_files = sorted(input_dir.rglob("*.txt"))
    if not source_files:
        print(f"ERROR: No .txt files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(source_files)} source file(s) in {input_dir}")
    print(f"Chunk size: {min_words}-{max_words} words")

    counter = TopicCounter()
    all_chunks: list[Chunk] = []

    for path in source_files:
        print(f"  Processing: {path.name}")
        chunks = process_source_file(path, min_words, max_words, counter)
        all_chunks.extend(chunks)
        print(f"    -> {len(chunks)} chunks")

    write_chunks(all_chunks, output_dir)
    print_summary(all_chunks)
    print(f"\nOutput written to {output_dir}/")


if __name__ == "__main__":
    main()
