# GutWise

Offline-first IBS health education assistant powered by fine-tuned Gemma 4.
Built for the Kaggle Gemma 4 Good Hackathon (Health & Sciences track).

## Quick Reference

```bash
# Install
uv sync --extra dev --extra collect --extra generate

# Lint + format
uv run ruff check .
uv run ruff format .

# Run tests
uv run pytest

# Run a script
uv run python scripts/collect/fetch_sources.py
```

## Architecture

```
Data Pipeline: Collect → Chunk → Generate → Validate → Train → Evaluate → Demo
```

### Pipeline Steps
1. `scripts/collect/` — Fetch open-access IBS sources (StatPearls, NICE, NHS, PubMed)
2. `scripts/chunk/` — Split sources into topic-based chunks
3. `scripts/generate/` — Synthetic Q&A generation via Claude with medical reference grounding
4. `scripts/evaluate/` — LLM judge (accuracy, safety, empathy, scope) + medical filters
5. `notebooks/training/` — QLoRA fine-tuning on Gemma 4
6. `notebooks/eval/` — Model evaluation and benchmarking
7. `app/` — Gradio demo with multimodal input

### Key Files
- `reference/ibs_reference.md` — Verified IBS medical facts (equivalent of api_reference.md)
- `reference/hallucination_blocklist.md` — Known medical myths and fake claims to reject
- `reference/medication_whitelist.md` — Verified IBS medications
- `scripts/generate/prompts.py` — Prompt templates for synthetic data generation
- `scripts/evaluate/medical_judge.py` — LLM judge adapted for medical content

### Datasets
- `datasets/sources/` — Raw collected source texts (guidelines, papers)
- `datasets/chunks/` — Chunked source material by topic
- `datasets/generated/` — Generated instruction pairs (pre-validation)
- `datasets/validated/` — Final validated training data
- Format: JSONL with ChatML messages array

## Data Quality Rules — CRITICAL

### Medical accuracy is non-negotiable
- Every generated Q&A pair MUST be grounded in source material from `reference/`
- The model must NEVER diagnose, prescribe specific dosages, or recommend stopping medications
- Safety pairs (10-15% of dataset) teach the model to defer to healthcare providers
- Red flag symptoms (bloody stool, weight loss, fever) must trigger "see a doctor" responses

### Validation pipeline
1. Source grounding check — every claim traceable to reference material
2. Medication whitelist check — reject unknown drug names
3. Safety pattern matching — reject responses that diagnose or prescribe
4. LLM judge scoring: accuracy (0-5), safety (0-5), empathy (0-5), scope (0-5), completeness (0-5)
5. Reject if safety < 4 or accuracy < 3

### Hallucination blocklist
- IBS "cures" (there is no cure, only management)
- Unproven treatments without disclaimers
- Specific dosage recommendations
- Diagnostic claims ("you have IBS-D")

## Hackathon Requirements

- **Deadline**: May 18, 2026
- **Model**: Gemma 4 (2B, 7B, 27B, or 27B-IT)
- **Track**: Health & Sciences
- **Submission**: Working demo + public repo + technical writeup + demo video
- **Judging**: Innovation 30%, Impact 30%, Technical Execution 25%, Accessibility 15%

## Conventions

- Python 3.10+, managed with `uv`
- Ruff for linting and formatting (line length 100)
- All scripts runnable with `uv run python scripts/...`
- Dataset files use JSONL format
- Commit messages: `feat:`, `fix:`, `chore:`, `docs:`, `data:`

## Pre-push Checklist

1. `uv run ruff check .` (zero warnings)
2. `uv run ruff format --check .`
3. `uv run pytest` (all pass)
