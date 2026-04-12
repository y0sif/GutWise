# GutWise Data Pipeline

## Overview

The pipeline follows the same proven architecture as OxideCoder/Arcwright,
adapted for medical content with additional safety validation.

```
1. Collect    -> Fetch open-access IBS sources
2. Chunk      -> Split by topic (diagnosis, diet, treatment, etc.)
3. Generate   -> Synthetic Q&A via Claude with reference grounding
4. Validate   -> LLM judge + medical safety filters
5. Train      -> QLoRA fine-tune Gemma 4
6. Evaluate   -> Medical accuracy + safety benchmarks
7. Demo       -> Gradio app with multimodal input
```

## Step 1: Collect

**Sources (ranked by license quality):**

| Source | License | Content |
|--------|---------|---------|
| StatPearls IBS | CC-BY 4.0 | Clinical reference |
| NICE CG61 | OGL v3.0 | Treatment guidelines |
| NHS IBS pages | OGL v3.0 | Patient education |
| MedlinePlus | Public Domain | Patient information |
| PubMed OA Subset | CC-BY | Research papers |

**Script**: `scripts/collect/fetch_sources.py`

## Step 2: Chunk

Split sources by medical topic:
- Diagnosis and subtypes
- Dietary management (FODMAP)
- Pharmacological treatments
- Psychological therapies
- Lifestyle management
- When to see a doctor (red flags)
- Living with IBS (daily management)

**Script**: `scripts/chunk/chunk_sources.py`

## Step 3: Generate

Use Claude to generate instruction pairs grounded in source chunks.
Every prompt includes:
- The source chunk
- `reference/ibs_reference.md` (verified facts)
- `reference/hallucination_blocklist.md` (things to reject)
- `reference/medication_whitelist.md` (valid medications)

**Conversation types to generate:**
- Simple factual Q&A (40%)
- Anxious patient tone ("I'm worried about...") (15%)
- Follow-up to doctor visit ("My doctor mentioned X...") (10%)
- Myth-busting ("Is it true that...?") (10%)
- Safety/refusal pairs (15%)
- Multi-turn conversations (10%)

**Script**: `scripts/generate/generate_batch.py`

## Step 4: Validate

### Automated checks:
1. Medication whitelist check
2. Dosage mention rejection
3. Diagnostic claim detection
4. Red flag symptom handling verification

### LLM Judge (5 dimensions):
- **Accuracy** (0-5): Medical claims correct per guidelines?
- **Safety** (0-5): Avoids dangerous advice? Recommends professional consultation?
- **Empathy** (0-5): Patient-friendly, non-dismissive tone?
- **Scope** (0-5): Stays within educational bounds?
- **Completeness** (0-5): Adequately addresses the question?

**Rejection thresholds**: safety < 4 OR accuracy < 3

**Script**: `scripts/evaluate/medical_judge.py`

## Step 5: Train

- **Base model**: Gemma 4 (7B or 27B-IT)
- **Method**: QLoRA (4-bit quantization)
- **Hyperparameters** (starting point, based on OxideCoder experience):
  - r=16, lora_alpha=32
  - lr=1e-4, 2-3 epochs
  - packing=False
- **Format**: ChatML messages array

**Notebook**: `notebooks/training/gutwise_finetune.ipynb`

## Step 6: Evaluate

### Custom IBS eval set:
- 50-100 IBS knowledge questions (MCQ)
- 30-50 safety scenarios (adversarial)
- 20-30 hallucination detection (trick questions)
- 20-30 conversational quality (multi-turn)

### Metrics:
- Accuracy on IBS knowledge quiz
- Safety score (% of correct refusals/escalations)
- Hallucination rate
- A/B comparison vs base Gemma 4

**Notebook**: `notebooks/eval/gutwise_eval.ipynb`

## Step 7: Demo

Gradio app with:
- Text chat (primary)
- Meal photo upload (FODMAP guidance via Gemma 4 vision)
- Symptom diary (structured input via function calling)
- Always-visible medical disclaimer

**App**: `app/main.py`

## Data Targets

| Category | Target Count | Notes |
|----------|-------------|-------|
| IBS-specific Q&A | 400-500 | Core domain |
| General GI context | 100-150 | Broader knowledge |
| Dietary guidance | 100-150 | FODMAP focus |
| Safety/refusal | 100-150 | "See a doctor" responses |
| Empathetic responses | 50-80 | Tone training |
| **Total** | **~800-1000** | Quality over quantity |
