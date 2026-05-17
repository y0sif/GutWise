# GutWise

**Offline-first IBS education assistant — a Gemma 4 E4B fine-tune that beats baseline on a paired medical eval *while keeping 4/4 red-flag handling across every run*.**

Built for the [Kaggle Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon) (Health & Sciences track).

> **GutWise is educational, not medical advice.** It does not diagnose, prescribe, or replace a clinician. If you see red-flag symptoms (rectal bleeding, unintentional weight loss, fever, nocturnal symptoms, onset after age 50, family history of colorectal cancer/IBD), see a doctor.

---

## Why this matters

Irritable Bowel Syndrome affects **10–15% of the global population**, yet most people never receive a formal diagnosis. Gastroenterologists are scarce in low-resource settings, generic LLMs hallucinate medications and dosages, and red-flag symptoms get folded into reassuring chat. GutWise targets that gap with a small, on-device, grounded model.

## What GutWise is

- A **Gemma 4 E4B** model fine-tuned with **QLoRA (r=8)** on 659 audited IBS Q&A pairs grounded in **Rome IV, ACG 2021, NICE CG61, BSG 2021, StatPearls, NHS**, and **MedlinePlus**.
- Designed to run **offline on consumer hardware** (no API calls leaking patient text).
- Evaluated with a **3-run, single-judge-batch, paired-comparison** protocol against base Gemma 4 E4B.

## Results (v2, 3-run mean ± σ)

| Metric | Baseline E4B | **GutWise v2** | Δ |
|---|---|---|---|
| Overall (1–5 mean across 5 dims) | 4.696 | **4.769 ± 0.028** | **+0.073** (beats baseline + 2σ_v2) |
| Empathy | 4.36 | 4.47 ± 0.08 | **+0.11** |
| Completeness | 4.44 | 4.67 ± 0.04 | **+0.23** |
| Safety | 4.94 | 4.93 ± 0.04 | ≈ 0 |
| Red-flag handling | 4/4 | **4/4 across all 3 runs** | safety preserved |
| Wins vs baseline (B/A/tie, avg over runs) | — | **27 / 13 / 10** | — |
| Variance σ_overall | — | 0.028 | — |

Per-category gains where it matters most: **red_flag +0.22**, **safety_refusal +0.19**. Only category-level regression: myth_busting −0.08 (within σ; on v3 backlog).

Full numbers in [`scripts/evaluate/results/eval_report_v2.json`](scripts/evaluate/results/eval_report_v2.json).

## How v2 happened — the methodology (this is the contribution)

GutWise v1 regressed *below baseline* (4.22 vs 4.70) with **red-flag handling 2/4**. v2 is the disciplined fix:

1. **Audit before regenerate.** ~63% of automated audit flags on v1 were spurious; verifying claims against `reference/ibs_reference.md` first avoided a wasted regen cycle.
2. **Expand reference, then prompt with it.** Added Tenapanor, hyoscyamine, *L. plantarum DSM 9843*, *B. bifidum MIMBb75* to `reference/medication_whitelist.md` and `reference/ibs_reference.md`. Every generation prompt is grounded in these files.
3. **Hand-rewrite the dangerous five.** 5 sev-4+ entries that caused the v1 red-flag regression were rewritten to defer to a clinician.
4. **Lower the LoRA dials.** v1 used `r=16` on <2k entries and *over-updated*. v2 follows the Hyperparameter Drift Framework: `r=8`, `lr=5e-5`, `dropout=0`, eff batch 32, **1 epoch**.
5. **Eval discipline.** 3 seeds (42/43/44), 150 paired prompts in a single Haiku judge batch, baseline scored *in the same batch* to eliminate judge-calibration drift.

## Architecture

```
Collect (StatPearls, NICE, NHS, ACG, BSG, MedlinePlus, PubMed)
    ↓
Chunk by topic (9 topics: diagnosis, diet, lifestyle, pharmacology, ...)
    ↓
Generate Q&A pairs with Claude sub-agents (reference-grounded prompts)
    ↓
Validate (medication whitelist + hallucination blocklist + LLM judge)
    ↓
Train (QLoRA r=8 on Gemma 4 E4B, Colab A100, 1 epoch)
    ↓
Evaluate (paired vs baseline, 3 seeds, single judge batch)
    ↓
Demo (Gradio app, runs on-device or on Hugging Face Spaces)
```

## Project layout

```
GutWise/
├── app.py                              # Hugging Face Spaces entrypoint
├── app/main.py                         # Local entrypoint (same code path)
├── reference/                          # The grounding files
│   ├── ibs_reference.md                # Rome IV, ACG, NICE, BSG facts
│   ├── medication_whitelist.md         # 58 verified IBS-relevant drugs
│   └── hallucination_blocklist.md      # Known false claims to reject
├── datasets/
│   ├── eval/heldout_questions.jsonl    # 50-question held-out eval set
│   └── validated/train_v2.jsonl        # 659 audited training pairs
├── notebooks/
│   ├── training/gutwise_finetune.ipynb # The notebook that produced v2
│   └── eval/
│       ├── baseline_e4b_eval.ipynb     # Baseline Gemma 4 E4B
│       └── gutwise_v2_eval.ipynb       # 3-run v2 eval
├── scripts/
│   ├── collect/                        # Source fetchers
│   ├── chunk/                          # Topic chunking
│   ├── generate/                       # Q&A generation
│   ├── evaluate/                       # Medical judge + filters
│   └── publish/push_model.py           # HF Hub push for the v2 LoRA
└── competitions/gemma-4-good-hackathon/
    └── WRITEUP.md                      # Hackathon technical writeup
```

## Quick start

```bash
# Install
uv sync --extra dev

# Run the demo locally (loads Gemma 4 E4B + v2 LoRA from HF Hub)
uv run python -m app.main

# Or run the HF Spaces entrypoint
uv run python app.py
```

The first run downloads the model (~7 GB) from Hugging Face. On a 16 GB+ GPU it runs in bf16; on CPU it falls back to 4-bit quantization.

## Reproducing the eval

```bash
# 50 held-out IBS questions, 6 categories (factual, safety_refusal, red_flag, ...)
jq -c '.' datasets/eval/heldout_questions.jsonl | head -3

# Re-run the v2 eval on Colab — open the notebook
notebooks/eval/gutwise_v2_eval.ipynb     # NUM_RUNS=3, seeds 42/43/44
```

The judge protocol pairs each v2 response against the baseline response side-by-side, scored across 5 dimensions (accuracy, safety, empathy, scope, completeness). Output: `scripts/evaluate/results/eval_report_v2.json`.

## Hackathon submission

- **Track**: Health & Sciences
- **Writeup**: [`competitions/gemma-4-good-hackathon/WRITEUP.md`](competitions/gemma-4-good-hackathon/WRITEUP.md)
- **Model**: [`y0sif/GutWise-v2`](https://huggingface.co/y0sif/GutWise-v2) (LoRA adapter on `unsloth/gemma-4-E4B-it`)
- **Demo**: [Hugging Face Space](https://huggingface.co/spaces/y0sif/GutWise) (link goes live on submission day)
- **Video**: see writeup

## Safety and limitations

- **Not a diagnostic tool.** Red-flag symptoms are escalated to "see a doctor" responses; on the eval set this happened 4/4 across every run, but the model is not perfect and you must verify any clinical claim.
- **English only.** All training and eval are English. Other-language outputs are not evaluated.
- **Adults only.** Pediatric IBS has different criteria; the training data is adult-focused.
- **No dosing.** The model is instructed never to recommend specific drug dosages.
- **Base model inherits its own quirks.** Some hallucinations come from base Gemma 4 (e.g., the "*L. rhamnosus* GC69" pattern) and are inherited by the fine-tune; v3 anti-hallucination pairs are planned.

## Sources & licenses

- StatPearls (CC-BY 4.0)
- NHS / NICE CG61 (Open Government Licence v3.0)
- MedlinePlus (Public Domain)
- ACG 2021, BSG 2021, Rome IV — referenced under fair-use citation
- 24 PubMed abstracts (open-access)

GutWise itself is **Apache-2.0** ([LICENSE](LICENSE)).

## Development

```bash
uv sync --extra dev --extra collect --extra generate
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Citation

If you use GutWise, please cite the Kaggle writeup:

```
@misc{gutwise2026,
  author       = {y0sif},
  title        = {GutWise: An Offline-First IBS Education Assistant Built on Gemma 4 E4B},
  year         = {2026},
  howpublished = {Kaggle Gemma 4 Good Hackathon},
  url          = {https://www.kaggle.com/competitions/gemma-4-good-hackathon}
}
```
