# GutWise: An Offline-First IBS Education Assistant on Gemma 4 E4B

**Kaggle Gemma 4 Good Hackathon — Health & Sciences track**

**Author:** y0sif &nbsp;|&nbsp; **License:** Apache-2.0
**Code:** https://github.com/y0sif/GutWise &nbsp;|&nbsp; **Model:** https://huggingface.co/y0sif/GutWise &nbsp;|&nbsp; **Dataset:** https://huggingface.co/datasets/y0sif/GutWise-IBS-QA &nbsp;|&nbsp; **Demo:** https://huggingface.co/spaces/y0sif/GutWise

> *GutWise is educational. It is not medical advice and not a diagnostic tool. Red-flag symptoms (rectal bleeding, unintentional weight loss, fever, nocturnal symptoms, onset after age 50, family history of colorectal cancer or IBD) require evaluation by a clinician.*

---

## 1. Why IBS, why now

Irritable Bowel Syndrome affects **10–15% of the global population** but is chronically under-diagnosed. Gastroenterologists are scarce in low-resource settings, and the people most affected — often working women under chronic stress — get the least time with a specialist when they finally do reach one. Generic LLMs make this worse: they hallucinate medications, invent probiotic strain names, and reassure their way past dangerous red-flag symptoms.

GutWise aims at a narrow but real wedge: **a small, on-device, grounded model that explains IBS the way a careful health educator would, while refusing to play doctor.** It's not a substitute for a clinician, it's the thing patients reach for at 2 AM when they're scared and don't want to call NHS 111 again.

## 2. What we built

A QLoRA fine-tune of `unsloth/gemma-4-E4B-it` on **659 audited IBS Q&A pairs** grounded in Rome IV, ACG 2021, NICE CG61, BSG 2021, StatPearls, NHS, and MedlinePlus, plus a Gradio demo runnable offline on consumer hardware.

The fine-tune is intentionally small. The contribution is the methodology — *how* to fine-tune medical AI without regressing safety — not the parameter count.

## 3. Headline results

3-run paired evaluation against base Gemma 4 E4B, 50 held-out questions, 5 dimensions (accuracy, safety, empathy, scope, completeness), single judge batch so there's no calibration drift between models.

| Metric | Baseline E4B | **GutWise (3-run mean ± σ)** | Δ |
|---|---|---|---|
| Overall | 4.696 | **4.769 ± 0.028** | **+0.073** (beats baseline + 2σ_v2) |
| Empathy | 4.36 | 4.47 ± 0.08 | **+0.11** |
| Completeness | 4.44 | 4.67 ± 0.04 | **+0.23** |
| Safety | 4.94 | 4.93 ± 0.04 | ≈ 0 (preserved) |
| Red-flag handling | 4/4 | **4/4 across all 3 runs** | safety preserved |
| Wins vs baseline (B/A/tie, avg) | — | 27 / 13 / 10 | — |
| Variance σ_overall | — | 0.028 | tight |

Per-category gains where it matters most: **red_flag +0.22, safety_refusal +0.19, doctor_followup +0.13.** Only category-level regression: myth_busting −0.08 (within σ; on v3 backlog).

The number that matters most isn't +0.073. It's **4/4 red-flag handling across every single run**. v1 regressed to 2/4. v2 recovers safety while gaining on empathy and completeness — that's the methodological claim.

## 4. The story: v1 → v2

GutWise v1 looked sane on paper (673 entries, r=16 LoRA, 2 epochs) and *regressed below baseline*: 4.22 vs 4.70 overall, **red-flag handling 2/4**. We almost shipped it.

The retrospective produced six levers:

| Lever | What it was | Why it mattered |
|---|---|---|
| 0. Verify audit flags first | Of 38 automated flags, only ~14 were real (~63% false positives) | Avoided a wasted regen cycle on clean data |
| 1. Surgical filter + rewrite | Removed 14 Cat-B entries, hand-rewrote 5 Cat-C red-flag entries | Where v1's safety regression actually came from |
| 2. Expand the reference | Added Tenapanor, hyoscyamine, *L. plantarum DSM 9843*, *B. bifidum MIMBb75* to `medication_whitelist.md` + `ibs_reference.md` | The "missing entity" gap that produced false-positive audit flags |
| 3. Lower the LoRA dials | r=16 → **r=8** | <2k examples cannot support r=16 without over-updating |
| 4. Tighten the rest | dropout 0.05 → 0, eff batch 8 → 32, 2 ep → **1 ep** | Smoother gradients, less drift |
| 5. Fix the judge | Truncation cap `[:2500]` → `[:5000]` | The baseline was being silently handicapped at the eval level |
| 6. 3-run variance eval | Single Haiku judge batch over all 150 paired prompts | A single seed at T=0.7 can land ±1.7σ from the mean — see the v4 Arcwright retraction |

The change that mattered most was Lever 3 (capacity). The change that proved the result was Lever 6 (variance discipline). Everything else was hygiene.

## 5. Pipeline

```
Collect sources (StatPearls, NICE, NHS, ACG, BSG, MedlinePlus, PubMed)
   ↓
Chunk by topic (diagnosis, diet, lifestyle, pharmacology, pathophysiology,
                psychological, patient_education, epidemiology, general)
   ↓
Generate Q&A pairs with Claude sub-agents, every prompt grounded in
reference/ibs_reference.md + reference/medication_whitelist.md +
reference/hallucination_blocklist.md
   ↓
Validate
  • medication whitelist (reject unknown drugs)
  • hallucination blocklist (reject known false claims, e.g., "IBS cures")
  • safety pattern match (reject diagnosing / prescribing / specific dosages)
  • LLM judge: accuracy, safety, empathy, scope, completeness (each 0–5)
  • reject if safety < 4 or accuracy < 3
   ↓
Audit (manual spot-check 20+ entries; verify flags before regenerating)
   ↓
Train (QLoRA r=8 on Gemma 4 E4B, Colab A100, 1 epoch)
   ↓
Evaluate (3 seeds, paired vs baseline, single judge batch)
   ↓
Demo (Gradio, runs offline on consumer hardware or HF Spaces)
```

Every step is reproducible from the repo. The shipping training set is
`datasets/validated/train_v2.jsonl` (659 entries). The held-out eval set is
`datasets/eval/heldout_questions.jsonl` (50 questions, 6 categories).

## 6. Training recipe

| Field | Value |
|---|---|
| Base | `unsloth/gemma-4-E4B-it` (4-bit) |
| Adapter | LoRA, r=8, α=16, dropout=0, 7 modules |
| LR / schedule | 5e-5 cosine, warmup_ratio=0.05 |
| Effective batch | 32 (4 × 8 accum) |
| Epochs | 1 |
| Precision | bf16 |
| Sequence length | 1024 |
| Packing | False |
| Completion-only loss | False (Unsloth VLM rejects it for Gemma 4) |
| NEFTune | Disabled (L4 OOM) |
| Hardware | Colab A100 |

Final val loss: 4.4 → 0.82 monotonic.

## 7. Eval methodology

- **Held-out set**: 50 IBS questions at `datasets/eval/heldout_questions.jsonl`. Distribution: factual_qa 20, anxious_patient 8, safety_refusal 8, myth_busting 5, doctor_followup 5, red_flag 4.
- **Inference**: 3 runs at `temperature=0.7`, `max_new_tokens=512`, seeds 42/43/44. bf16, plain `transformers.AutoModelForImageTextToText` + `peft.PeftModel` (no Unsloth at inference — Unsloth's `FastModel` wrapper has known issues at Gemma 4 inference time).
- **Judge protocol**: 150 paired prompts (50 questions × 3 v2 runs), each scored side-by-side with the baseline response across 5 dimensions, by Claude Haiku 4.5 sub-agents in a single parallel wave.
- **Output**: `scripts/evaluate/results/eval_report_v2.json` + 3 per-run JSON files.

## 8. Demo

The Gradio app has three components:

1. **Chat** — talk to GutWise with the disclaimer banner permanently visible.
2. **Side-by-side: baseline vs GutWise** — same prompt to both models; you can see the empathy + completeness improvements, and you can see the red-flag handling on prompts like *"I've been losing weight without trying and there's blood in my stool — could this be IBS?"*
3. **About** — pipeline, eval, sources, license.

The demo runs offline on a 16 GB+ GPU; it's also deployed to Hugging Face Spaces with the ZeroGPU decorator for the public submission link.

## 9. Limitations and honest framing

- **Not a diagnostic tool.** Red-flag handling was 4/4 on every eval run, but the eval set is 50 questions — that's a vibe check, not a clinical trial.
- **English only.** Translation is out of scope.
- **Adults only.** Pediatric IBS criteria differ; the training data is adult-focused.
- **Base model quirks are inherited.** Some hallucinations (e.g., "*L. rhamnosus* GC69") come from base Gemma 4 and survive the fine-tune; v3 anti-hallucination pairs are planned.
- **eval_049/050 were initially flagged as duplicates** but verified to cover distinct red flags (family-history vs. fever + nocturnal symptoms).

## 10. What's next (v3 plan, deferred until post-hackathon)

In order of leverage × cost on v2 weaknesses:

1. Dedupe + expand the eval set to 60–80 questions (myth-busting coverage).
2. Add 15–20 grounded myth_busting training pairs (the only category v2 regressed on).
3. Fix the shared Rome IV duplication in training data (both v2 and baseline duplicate one sub-criterion).
4. Build the fact-match gate — claim-level reference matching as a third validation gate, the medical analog of `cargo check`.
5. Add explicit anti-hallucination pairs for base-model patterns (GC69, "leaky gut", invented strain names).
6. Scale to ~900–1000 entries (still <2k, keep r=8).

If post-hackathon work proves out, a 26B-base variant becomes worth trying. For now, **a 4B model that beats baseline with σ=0.028 and keeps 4/4 red-flag handling is the right thing to ship to people who don't have a doctor.**

## 11. Reproducing this work

```bash
git clone https://github.com/y0sif/GutWise
cd GutWise
uv sync --extra dev --extra collect --extra generate

# Inspect the data
head -3 datasets/validated/train_v2.jsonl
jq -c '.' datasets/eval/heldout_questions.jsonl | head -3

# Open the training notebook on Colab (A100)
notebooks/training/gutwise_finetune.ipynb

# Run the 3-run eval (Colab A100)
notebooks/eval/gutwise_v2_eval.ipynb

# Read the report
cat scripts/evaluate/results/eval_report_v2.json
```

## 12. Acknowledgments

- The open-access medical literature this is built on, especially StatPearls, NHS, NICE, ACG, BSG, and MedlinePlus.
- Google DeepMind for releasing Gemma 4 with an OSS-friendly license.
- Kaggle for hosting the hackathon and making the impact framing explicit.
