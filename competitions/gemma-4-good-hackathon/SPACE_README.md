---
title: GutWise
emoji: 🩺
colorFrom: emerald
colorTo: green
sdk: gradio
sdk_version: "5.9.0"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Offline IBS education assistant — Gemma 4 E4B fine-tune
---

# GutWise

**Offline-first IBS education assistant — a Gemma 4 E4B fine-tune that beats baseline on a paired medical eval *while keeping 4/4 red-flag handling across every run*.**

Built for the [Kaggle Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon) (Health & Sciences track).

> **GutWise is educational, not medical advice.** It does not diagnose, prescribe, or replace a clinician. If you see red-flag symptoms (rectal bleeding, unintentional weight loss, fever, nocturnal symptoms, onset after age 50, family history of colorectal cancer/IBD), see a doctor.

---

## Try it

- **Chat** tab — talk to GutWise. The disclaimer banner is always visible.
- **Side-by-side** tab — compare GutWise against base Gemma 4 E4B on the same prompt. Try `"I've been losing weight without trying and there's blood in my stool. Could this be IBS?"` to see red-flag handling diverge.
- **About** tab — sources, eval methodology, license.

The Space runs on **ZeroGPU** (free). First query after idle has a cold-start (~30 s).

## Results (3-run paired eval)

| Metric | Baseline E4B | **GutWise** | Δ |
|---|---|---|---|
| Overall | 4.696 | **4.769 ± 0.028** | **+0.073** (beats baseline + 2σ) |
| Empathy | 4.36 | 4.47 ± 0.08 | **+0.11** |
| Completeness | 4.44 | 4.67 ± 0.04 | **+0.23** |
| Safety | 4.94 | 4.93 ± 0.04 | ≈ 0 (preserved) |
| Red-flag handling | 4/4 | **4/4 across all 3 runs** | safety preserved |

The number that matters most isn't +0.073. It's **4/4 red-flag handling on every run**. v1 regressed to 2/4. v2 recovers safety while gaining on empathy and completeness.

## What's under the hood

- **Base**: `unsloth/gemma-4-E4B-it`
- **Adapter**: LoRA r=8, α=16, 1 epoch on 659 audited IBS Q&A pairs
- **Grounding**: Rome IV, ACG 2021, NICE CG61, BSG 2021, StatPearls, NHS, MedlinePlus
- **Eval**: 50 held-out questions × 3 seeds × paired judging, single Haiku judge batch

## Links

- **Code**: https://github.com/y0sif/GutWise
- **Model (LoRA)**: https://huggingface.co/y0sif/GutWise
- **Kaggle writeup**: linked from the GitHub README on submission day

## Safety

- **English only.** Adult IBS only (pediatric criteria differ).
- **No dosing.** The model is instructed never to recommend specific drug dosages.
- **Hallucinations inherited from the base model are possible.** Always verify clinical claims against a clinician.

GutWise is **Apache-2.0**.
