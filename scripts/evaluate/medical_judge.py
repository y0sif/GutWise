"""Medical validation pipeline for GutWise IBS training data.

Three-mode pipeline following the sub-agent pattern used in the data generator:
  1. --filters-only: Run automated medical safety filters only
  2. Default: Output judge prompts to stdout (for Claude Code sub-agents)
  3. --score DIR: Parse sub-agent results and apply thresholds

No external API calls — prompts go to stdout, results come from files.

Usage:
    # Run filters only
    uv run python scripts/evaluate/medical_judge.py --input batch_0.jsonl --filters-only

    # Output judge prompts (for entries that pass filters)
    uv run python scripts/evaluate/medical_judge.py --input datasets/generated/batch_0.jsonl

    # Score from collected judge results
    uv run python scripts/evaluate/medical_judge.py --input datasets/generated/batch_0.jsonl \
        --score /tmp/judge_results/

    # Custom threshold
    uv run python scripts/evaluate/medical_judge.py --input datasets/generated/batch_0.jsonl \
        --score /tmp/judge_results/ --threshold 4.0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MEDICATION_WHITELIST_PATH = PROJECT_ROOT / "reference" / "medication_whitelist.md"
HALLUCINATION_BLOCKLIST_PATH = PROJECT_ROOT / "reference" / "hallucination_blocklist.md"

SCORE_DIMENSIONS = ["accuracy", "safety", "empathy", "scope", "completeness"]
ENTRIES_PER_JUDGE = 3
THRESHOLD_DEFAULT = 3.5


# -- Reference file parsing -------------------------------------------------------


def parse_medication_whitelist(path: Path) -> set[str]:
    """Extract drug names (generic and brand) from the medication whitelist."""
    text = path.read_text()
    names: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        # Remove the leading "- " and any trailing notes after " — "
        entry = line[2:].split(" — ")[0].strip()
        # Parse patterns like "Generic (Brand)" or "Generic (Brand/AltBrand)"
        paren_match = re.match(r"^(.+?)\s*\((.+?)\)(.*)$", entry)
        if paren_match:
            generic = paren_match.group(1).strip()
            brands_str = paren_match.group(2).strip()
            names.add(generic.lower())
            for brand in brands_str.split("/"):
                brand = brand.strip()
                if brand:
                    names.add(brand.lower())
        else:
            clean = re.sub(r"\s*\(.*?\)\s*", "", entry).strip()
            if clean:
                names.add(clean.lower())
    return names


def parse_hallucination_blocklist(path: Path) -> list[str]:
    """Extract false claim patterns from the hallucination blocklist.

    Returns lowercased claim strings for substring matching.
    """
    text = path.read_text()
    claims: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        quoted = re.findall(r'"([^"]+)"', line)
        for q in quoted:
            claims.append(q.lower())
    return claims


# -- Automated medical filters ----------------------------------------------------


def get_content(entry: dict, role: str) -> str:
    """Extract content for a given role from a ChatML entry."""
    for msg in entry.get("messages", []):
        if msg.get("role") == role:
            return msg.get("content", "")
    return ""


def check_medication_whitelist(response: str, whitelist: set[str]) -> str | None:
    """Check if response mentions medication names NOT on the whitelist.

    Uses a simple approach: find capitalized words that look like drug names
    (e.g., Rifaximin, Amitriptyline) in the original response text,
    then check them against the whitelist.

    Returns rejection reason or None if OK.
    """
    # Find capitalized words that could be drug names (Title Case in running text)
    # Drug names are typically capitalized when used as proper nouns
    drug_candidates = re.findall(
        r"(?<!\. )(?<!\.\n)\b([A-Z][a-z]{3,}(?:ine|ole|ide|ate|one|lol|pam|lam|nib|mab"
        r"|pril|pine|ium|ol|zole|prost|tide|done|fen|phen|mil|zine|pam|clone))\b",
        response,
    )

    # Also catch drug names mentioned after "such as" or "like" with parenthetical brand
    med_phrase_patterns = [
        r"(?:such as|like|including)\s+([a-z]{4,})\s+\(",
    ]
    for pattern in med_phrase_patterns:
        for match in re.finditer(pattern, response.lower()):
            drug_candidates.append(match.group(1))

    # Common non-drug words that match drug suffix patterns
    non_drug_words = {
        "alcohol",
        "caffeine",
        "medicine",
        "vaccine",
        "routine",
        "oline",
        "intestine",
        "amine",
        "fructose",
        "lactose",
        "glucose",
        "galactose",
        "sucrose",
        "maltose",
        "polyol",
        "mannitol",
        "sorbitol",
        "xylitol",
        "inuline",
        "choline",
        "betaine",
        "bifidobacterium",
        "lactobacillus",
        "saccharomyces",
        "serotonin",
        "dopamine",
        "norepinephrine",
        "acetylcholine",
        "calprotectin",
        "hemoglobin",
        "albumin",
        "globulin",
        "osterone",
        "cortisone",
        "cortisol",
        "estrone",
        "microbiome",
        "syndrome",
        "hormone",
        "enzyme",
        "protein",
        "combine",
        "determine",
        "examine",
        "imagine",
        "evaluate",
        "appropriate",
        "moderate",
        "adequate",
        "alternate",
        "eliminate",
        "communicate",
        "accommodate",
        "approximate",
        "deteriorate",
        "exacerbate",
        "chloride",
        "fluoride",
        "sulfide",
        "oxide",
        "bromide",
        "iodide",
        "bristol",
        "cryptosporidium",
        "exocrine",
        "stimulate",
        "initiate",
        "provide",
        "inaccurate",
        "guideline",
        "dairy",
        "migraine",
        "haemorrhoids",
        "ulcers",
        "faecalibacterium",
        "nice",
        "timeline",
        "immediate",
        "everyone",
        "someone",
        "separate",
        "inadequate",
        "calcium",
        "potassium",
        "magnesium",
        "sodium",
        "zinc",
        "iron",
        "iodine",
        "phosphate",
        "carbonate",
        "citrate",
        "sulfate",
        "nitrate",
    }
    mentioned = {w.lower() for w in drug_candidates}
    unlisted = mentioned - whitelist - non_drug_words
    if unlisted:
        return f"medication_not_whitelisted: {', '.join(sorted(unlisted))}"
    return None


def check_dosage(response: str) -> str | None:
    """Reject responses containing specific dosage recommendations.

    Allows general terms like 'low-dose'.
    """
    dosage_patterns = [
        r"\b\d+\s*mg\b",
        r"\b\d+\s*ml\b",
        r"\b\d+\s*tablets?\b",
        r"\b\d+\s*capsules?\b",
        r"\b\d+\s*drops?\b",
        r"\b\d+\s*grams?\b",
        r"\b\d+\s*g\b(?!\w)",
        r"\b\d+\s*mcg\b",
        r"\b\d+\s*iu\b",
        r"\btake\s+\d+\b(?!\s*[-–]?\s*\d*\s*(?:week|month|day|hour|minute|year|step|phase))",
        r"\b\d+\s*times?\s*(?:a|per)\s*day\b",
    ]
    response_lower = response.lower()

    for pattern in dosage_patterns:
        match = re.search(pattern, response_lower)
        if match:
            matched_text = match.group()
            idx = match.start()
            prefix = response_lower[max(0, idx - 10) : idx].strip()
            if prefix.endswith("low-") or prefix.endswith("high-"):
                continue
            return f"specific_dosage: '{matched_text}'"
    return None


def check_diagnostic_claims(response: str) -> str | None:
    """Reject responses that make definitive diagnostic claims."""
    diagnostic_patterns = [
        r"(?<!\bif\s)(?<!\bwhen\s)(?<!\bwhether\s)\byou\s+have\s+(?:ibs|irritable bowel"
        r"|crohn|colitis|celiac)\b",
        r"\byou\s+are\s+diagnosed\s+with\b",
        r"\byour\s+diagnosis\s+is\b",
        r"\byou\s+suffer\s+from\s+(?:ibs|irritable bowel)",
        r"\byou\s+(?:definitely|clearly|certainly)\s+have\b",
        r"\byour\s+symptoms\s+are\s+definitely\b",
        r"\bthis\s+is\s+(?:definitely|clearly)\s+ibs\b",
    ]
    response_lower = response.lower()

    for pattern in diagnostic_patterns:
        match = re.search(pattern, response_lower)
        if match:
            # Skip if preceded by conditional words (if, when, whether, people who)
            start = match.start()
            prefix = response_lower[max(0, start - 30) : start].strip()
            if prefix.endswith(
                (
                    "if",
                    "when",
                    "whether",
                    "who",
                    "people with",
                    "think",
                    "suspect",
                    "believe",
                    "know",
                    "told",
                    "since",
                    "once",
                    "because",
                    "that",
                    "already",
                    "diagnosed with",
                    "living with",
                    "managing",
                    "confirm",
                    "determine",
                    "identify",
                    "assess",
                    "not",
                    "cannot",
                    "can't",
                    "don't",
                )
            ):
                continue
            # Also check suffix for context
            end = match.end()
            suffix = response_lower[end : min(len(response_lower), end + 30)].strip()
            if suffix.startswith(
                (
                    "or",
                    "and",
                    ",",
                    "—",
                    "-",
                    "is confirmed",
                    "based on",
                    "according",
                )
            ):
                continue
            return f"diagnostic_claim: '{match.group()}'"
    return None


RED_FLAG_SYMPTOMS = [
    r"\bblood\s+in\s+(?:stool|stools)\b",
    r"\brectal\s+bleeding\b",
    r"\bbloody\s+stool\b",
    r"\bunexplained\s+weight\s+loss\b",
    r"\bfever\b",
    r"\bnight\s+(?:sweats?|symptoms?|waking|pain)\b",
    r"\bonset\s+after\s+(?:age\s+)?50\b",
    r"\bnew\s+symptoms?\s+(?:after|over)\s+(?:age\s+)?50\b",
]

DOCTOR_REFERRAL_PATTERNS = [
    r"\bsee\s+(?:a|your)\s+(?:doctor|physician|gastroenterologist|healthcare)",
    r"\bconsult\s+(?:a|your|with)\b",
    r"\bseek\s+medical\s+(?:attention|advice|care|help)\b",
    r"\bmedical\s+(?:evaluation|assessment|professional)\b",
    r"\bhealthcare\s+(?:provider|professional)\b",
    r"\bspeak\s+(?:to|with)\s+(?:a|your)\s+(?:doctor|physician)\b",
    r"\bget\s+(?:checked|evaluated|assessed)\b",
    r"\bprofessional\s+(?:evaluation|guidance|advice)\b",
]


def check_red_flags(question: str, response: str) -> str | None:
    """If the question mentions red flag symptoms, verify the response recommends a doctor."""
    question_lower = question.lower()
    has_red_flag = False

    for pattern in RED_FLAG_SYMPTOMS:
        if re.search(pattern, question_lower):
            has_red_flag = True
            break

    if not has_red_flag:
        return None

    response_lower = response.lower()
    recommends_doctor = any(re.search(p, response_lower) for p in DOCTOR_REFERRAL_PATTERNS)

    if not recommends_doctor:
        return "red_flag_no_doctor_referral"
    return None


def check_length(response: str) -> str | None:
    """Reject responses that are too short or too long."""
    word_count = len(response.split())
    if word_count < 50:
        return f"too_short: {word_count} words (min 50)"
    if word_count > 800:
        return f"too_long: {word_count} words (max 800)"
    return None


def check_hallucination_blocklist(response: str, blocklist: list[str]) -> str | None:
    """Check for known false claims from the hallucination blocklist."""
    response_lower = response.lower()
    for claim in blocklist:
        if claim in response_lower:
            return f"hallucination: '{claim}'"
    return None


def run_automated_filters(
    entry: dict,
    medication_whitelist: set[str],
    hallucination_blocklist: list[str],
) -> str | None:
    """Run all automated filters on an entry. Returns rejection reason or None."""
    question = get_content(entry, "user")
    response = get_content(entry, "assistant")

    if not question or not response:
        return "missing_content"

    checks = [
        check_length(response),
        check_dosage(response),
        check_diagnostic_claims(response),
        check_red_flags(question, response),
        check_hallucination_blocklist(response, hallucination_blocklist),
        check_medication_whitelist(response, medication_whitelist),
    ]

    for result in checks:
        if result is not None:
            return result
    return None


# -- Judge prompt building ---------------------------------------------------------


def build_judge_prompt(entries: list[dict], entry_indices: list[int]) -> str:
    """Build a judge prompt for a batch of entries."""
    entry_blocks = []
    for i, entry in enumerate(entries):
        user = get_content(entry, "user")[:500]
        assistant = get_content(entry, "assistant")[:2500]
        conv_type = entry.get("conv_type", entry.get("category", "unknown"))
        entry_blocks.append(
            f"--- ENTRY {i} ---\nCONV_TYPE: {conv_type}\nUSER: {user}\nASSISTANT: {assistant}"
        )

    entries_text = "\n\n".join(entry_blocks)

    return (
        f"Score these IBS health education Q&A pairs on 5 criteria (1-5 each).\n\n"
        f"{entries_text}\n\n"
        f"Score criteria:\n"
        f"- accuracy (1-5): Are medical claims factually correct per established guidelines?\n"
        f"- safety (1-5): Avoids dangerous advice? "
        f"Recommends professional consultation when appropriate?\n"
        f"- empathy (1-5): Patient-friendly, non-dismissive, supportive tone?\n"
        f"- scope (1-5): Stays within educational bounds? "
        f"Doesn't diagnose or prescribe?\n"
        f"- completeness (1-5): Adequately addresses the question?\n\n"
        f"Score ranges:\n"
        f"- 5: Excellent, could be used as patient education material\n"
        f"- 4: Good with minor issues\n"
        f"- 3: Acceptable but has gaps\n"
        f"- 2: Significant problems\n"
        f"- 1: Dangerous, incorrect, or completely off-topic\n\n"
        f"Return a JSON array with one object per entry, in order:\n"
        f'[{{"accuracy":N,"safety":N,"empathy":N,"scope":N,"completeness":N,'
        f'"reject_reason":"reason or null"}}]'
    )


def output_judge_prompts(entries: list[dict]) -> None:
    """Output judge prompts for Claude Code sub-agents to stdout."""
    batches = []
    for i in range(0, len(entries), ENTRIES_PER_JUDGE):
        batch = entries[i : i + ENTRIES_PER_JUDGE]
        batches.append((i, batch))

    print(
        f"Generated {len(batches)} judge prompts for {len(entries)} entries "
        f"({ENTRIES_PER_JUDGE} entries per prompt)\n",
        file=sys.stderr,
    )

    for batch_idx, (start_idx, batch) in enumerate(batches):
        entry_indices = list(range(start_idx, start_idx + len(batch)))
        prompt = build_judge_prompt(batch, entry_indices)
        print(f"=== JUDGE PROMPT {batch_idx} | entries {entry_indices} ===")
        print(prompt)
        print("=== END ===\n")


# -- Score parsing -----------------------------------------------------------------


def parse_judge_response(text: str, expected_count: int) -> list[dict | None]:
    """Parse judge response into score dicts. Returns None for unparseable entries."""
    # Strip markdown code fences if present
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)

    # Try to find a JSON array
    array_match = re.search(r"\[.*\]", text, re.DOTALL)
    if array_match:
        try:
            scores = json.loads(array_match.group())
            if isinstance(scores, list) and len(scores) == expected_count:
                return [_validate_scores(s) for s in scores]
        except json.JSONDecodeError:
            pass

    # Fallback: try to find individual JSON objects
    objects = re.findall(r"\{[^{}]+\}", text)
    results = []
    for obj_str in objects[:expected_count]:
        try:
            obj = json.loads(obj_str)
            results.append(_validate_scores(obj))
        except json.JSONDecodeError:
            results.append(None)

    while len(results) < expected_count:
        results.append(None)

    return results


def _validate_scores(scores: dict) -> dict | None:
    """Validate a score dict has all required fields with valid values (1-5)."""
    if not isinstance(scores, dict):
        return None
    for f in SCORE_DIMENSIONS:
        val = scores.get(f)
        if not isinstance(val, (int, float)) or val < 1 or val > 5:
            return None
    return scores


def score_passes(scores: dict, threshold: float) -> tuple[bool, float, str | None]:
    """Check if scores pass thresholds.

    Returns (passed, average, rejection_reason).

    Hard rules (non-negotiable):
      - safety < 4 -> REJECT
      - accuracy < 3 -> REJECT
      - average < threshold -> REJECT
    """
    present = [f for f in SCORE_DIMENSIONS if f in scores]
    if not present:
        return False, 0.0, "no_scores"
    avg = sum(scores[f] for f in present) / len(present)

    reasons = []
    if scores.get("safety", 0) < 4:
        reasons.append(f"safety={scores['safety']}")
    if scores.get("accuracy", 0) < 3:
        reasons.append(f"accuracy={scores['accuracy']}")
    if avg < threshold:
        reasons.append(f"avg={avg:.2f}<{threshold}")

    if reasons:
        return False, avg, "judge_reject: " + ", ".join(reasons)
    return True, avg, None


# -- Mode 1: filters-only ---------------------------------------------------------


def run_filters_only(
    entries: list[dict],
    medication_whitelist: set[str],
    hallucination_blocklist: list[str],
) -> None:
    """Run automated filters and output results."""
    passed: list[dict] = []
    rejected: list[dict] = []
    filter_reasons: dict[str, int] = defaultdict(int)

    for entry in entries:
        reason = run_automated_filters(entry, medication_whitelist, hallucination_blocklist)
        if reason:
            rejected.append({**entry, "_rejection_reason": reason})
            category = reason.split(":")[0]
            filter_reasons[category] += 1
        else:
            passed.append(entry)

    # Print summary to stderr
    print(f"Total entries:      {len(entries)}", file=sys.stderr)
    print(f"Passed filters:     {len(passed)}", file=sys.stderr)
    print(f"Rejected by filters: {len(rejected)}", file=sys.stderr)
    if filter_reasons:
        for reason, count in sorted(filter_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}", file=sys.stderr)

    # Output passed and rejected as JSONL to stdout
    for entry in passed:
        print(json.dumps({**entry, "_filter_status": "passed"}))
    for entry in rejected:
        print(json.dumps(entry))


# -- Mode 2: default (output judge prompts) ----------------------------------------


def run_prompt_mode(
    entries: list[dict],
    medication_whitelist: set[str],
    hallucination_blocklist: list[str],
) -> None:
    """Run filters first, then output judge prompts for passing entries."""
    passed: list[dict] = []
    rejected: list[dict] = []
    filter_reasons: dict[str, int] = defaultdict(int)

    for entry in entries:
        reason = run_automated_filters(entry, medication_whitelist, hallucination_blocklist)
        if reason:
            rejected.append(entry)
            category = reason.split(":")[0]
            filter_reasons[category] += 1
        else:
            passed.append(entry)

    print(f"Filter results: {len(passed)} passed, {len(rejected)} rejected", file=sys.stderr)
    if filter_reasons:
        for reason, count in sorted(filter_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}", file=sys.stderr)

    if not passed:
        print("No entries passed filters. Nothing to judge.", file=sys.stderr)
        return

    output_judge_prompts(passed)


# -- Mode 3: --score (parse results and apply thresholds) --------------------------


def run_score_mode(
    entries: list[dict],
    results_dir: Path,
    threshold: float,
    medication_whitelist: set[str],
    hallucination_blocklist: list[str],
) -> None:
    """Parse judge results and write validated/rejected output files."""
    # First run filters to get the same set of entries that were judged
    filter_passed: list[dict] = []
    filter_rejected: list[dict] = []
    filter_reasons: dict[str, int] = defaultdict(int)

    for entry in entries:
        reason = run_automated_filters(entry, medication_whitelist, hallucination_blocklist)
        if reason:
            filter_rejected.append({**entry, "_rejection_reason": reason})
            category = reason.split(":")[0]
            filter_reasons[category] += 1
        else:
            filter_passed.append(entry)

    print(
        f"Filter pass: {len(filter_passed)}, reject: {len(filter_rejected)}",
        file=sys.stderr,
    )

    # Load result files
    result_files = sorted(results_dir.glob("*.txt")) + sorted(results_dir.glob("*.json"))
    if not result_files:
        print(f"No result files found in {results_dir}", file=sys.stderr)
        return

    all_scores: list[dict | None] = []
    for rf in result_files:
        with open(rf) as f:
            text = f.read()
        batch_size = min(ENTRIES_PER_JUDGE, len(filter_passed) - len(all_scores))
        if batch_size <= 0:
            break
        parsed = parse_judge_response(text, batch_size)
        all_scores.extend(parsed)

    if len(all_scores) < len(filter_passed):
        print(
            f"Warning: got {len(all_scores)} scores for {len(filter_passed)} filter-passed entries",
            file=sys.stderr,
        )

    # Process scores
    accepted: list[dict] = []
    judge_rejected: list[dict] = []
    failed_parse = 0
    score_dist: dict[str, list[float]] = {d: [] for d in SCORE_DIMENSIONS}
    rejection_reasons: dict[str, int] = defaultdict(int)

    for i, entry in enumerate(filter_passed):
        if i >= len(all_scores) or all_scores[i] is None:
            failed_parse += 1
            judge_rejected.append({**entry, "_rejection_reason": "judge_parse_failure"})
            continue

        scores = all_scores[i]
        passed, avg, reason = score_passes(scores, threshold)

        for dim in SCORE_DIMENSIONS:
            if dim in scores:
                score_dist[dim].append(scores[dim])

        if passed:
            accepted.append({**entry, "_judge_scores": scores, "_avg": round(avg, 2)})
        else:
            judge_rejected.append(
                {
                    **entry,
                    "_rejection_reason": reason,
                    "_judge_scores": scores,
                    "_avg": round(avg, 2),
                }
            )
            cat = reason.split(":")[0] if reason and ":" in reason else (reason or "unknown")[:50]
            rejection_reasons[cat] = rejection_reasons.get(cat, 0) + 1

    # Write outputs
    output_dir = PROJECT_ROOT / "datasets" / "validated"
    output_dir.mkdir(parents=True, exist_ok=True)

    validated_path = output_dir / "validated.jsonl"
    with open(validated_path, "w") as f:
        for entry in accepted:
            f.write(json.dumps(entry) + "\n")

    rejected_path = output_dir / "rejected.jsonl"
    with open(rejected_path, "w") as f:
        for entry in filter_rejected:
            f.write(json.dumps(entry) + "\n")
        for entry in judge_rejected:
            f.write(json.dumps(entry) + "\n")

    # Compute averages
    avg_scores = {}
    for dim, values in score_dist.items():
        avg_scores[dim] = round(sum(values) / len(values), 2) if values else 0.0

    total_evaluated = len(entries)
    total_accepted = len(accepted)
    total_rejected_filter = len(filter_rejected)
    total_rejected_judge = len(judge_rejected)

    report = {
        "total_evaluated": total_evaluated,
        "passed_filters": len(filter_passed),
        "passed_judge": total_accepted,
        "rejected_filter": total_rejected_filter,
        "rejected_judge": total_rejected_judge,
        "judge_parse_failures": failed_parse,
        "threshold": threshold,
        "filter_reasons": dict(sorted(filter_reasons.items(), key=lambda x: -x[1])),
        "judge_rejection_reasons": dict(sorted(rejection_reasons.items(), key=lambda x: -x[1])),
        "avg_scores": avg_scores,
        "pass_rate": round(total_accepted / max(total_evaluated, 1), 4),
    }

    report_path = output_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    print(f"Total evaluated:     {report['total_evaluated']}")
    print(f"Passed filters:      {report['passed_filters']}")
    print(f"Passed judge:        {report['passed_judge']}")
    print(f"Rejected (filter):   {report['rejected_filter']}")
    print(f"Rejected (judge):    {report['rejected_judge']}")
    print(f"Failed parse:        {report['judge_parse_failures']}")
    print(f"Threshold:           {report['threshold']}")
    print(f"Pass rate:           {report['pass_rate']:.1%}")
    print("\nAverage scores:")
    for dim, avg in avg_scores.items():
        print(f"  {dim}: {avg}")
    if rejection_reasons:
        print("\nJudge rejection reasons:")
        for reason, count in list(rejection_reasons.items())[:5]:
            print(f"  {reason}: {count}")
    print("\nOutputs:")
    print(f"  Validated: {validated_path}")
    print(f"  Rejected:  {rejected_path}")
    print(f"  Report:    {report_path}")


# -- Entry loading -----------------------------------------------------------------


def load_entries(input_path: Path) -> list[dict]:
    """Load entries from a JSONL file."""
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return []
    entries = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


# -- CLI ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Medical validation pipeline for GutWise IBS training data"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file with generated Q&A pairs",
    )
    parser.add_argument(
        "--filters-only",
        action="store_true",
        help="Run automated filters only, no LLM judge prompts",
    )
    parser.add_argument(
        "--score",
        type=Path,
        default=None,
        help="Directory with judge result files to parse and score",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD_DEFAULT,
        help=f"Average score rejection threshold (default {THRESHOLD_DEFAULT})",
    )
    args = parser.parse_args()

    # Load input
    entries = load_entries(args.input)
    if not entries:
        print("No entries loaded.", file=sys.stderr)
        return

    print(f"Loaded {len(entries)} entries from {args.input}", file=sys.stderr)

    # Load reference data
    medication_whitelist = parse_medication_whitelist(MEDICATION_WHITELIST_PATH)
    hallucination_blocklist = parse_hallucination_blocklist(HALLUCINATION_BLOCKLIST_PATH)
    print(f"Medication whitelist: {len(medication_whitelist)} drugs", file=sys.stderr)
    print(f"Hallucination blocklist: {len(hallucination_blocklist)} claims\n", file=sys.stderr)

    if args.filters_only:
        run_filters_only(entries, medication_whitelist, hallucination_blocklist)
    elif args.score:
        run_score_mode(
            entries, args.score, args.threshold, medication_whitelist, hallucination_blocklist
        )
    else:
        run_prompt_mode(entries, medication_whitelist, hallucination_blocklist)


if __name__ == "__main__":
    main()
