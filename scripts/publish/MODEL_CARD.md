---
license: apache-2.0
base_model: unsloth/gemma-4-E4B-it
language:
  - en
tags:
  - gemma
  - gemma-4
  - lora
  - peft
  - medical
  - ibs
  - health-education
  - kaggle-gemma-4-good-hackathon
library_name: peft
pipeline_tag: text-generation
---

# GutWise — IBS Education Assistant (LoRA adapter)

A QLoRA adapter for [`unsloth/gemma-4-E4B-it`](https://huggingface.co/unsloth/gemma-4-E4B-it), fine-tuned on 659 audited Irritable Bowel Syndrome Q&A pairs grounded in Rome IV, ACG 2021, NICE CG61, BSG 2021, StatPearls, NHS, and MedlinePlus.

Built for the [Kaggle Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon) (Health & Sciences track).

> **Educational only — not medical advice, not a diagnostic tool.** Red-flag symptoms (rectal bleeding, unintentional weight loss, fever, nocturnal symptoms, onset after age 50, family history of colorectal cancer or IBD) require evaluation by a clinician.

## Quick start

```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer
import torch

base_id = "unsloth/gemma-4-E4B-it"
adapter_id = "y0sif/GutWise"

tok = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForImageTextToText.from_pretrained(
    base_id, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_id)

messages = [
    {"role": "system", "content": "You are GutWise, an IBS health education assistant. ..."},
    {"role": "user", "content": "What is the low-FODMAP diet?"},
]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9)
print(tok.decode(out[0, inputs.shape[-1]:], skip_special_tokens=True))
```

## Eval results (v2, 3-run mean ± σ)

| Metric | Baseline E4B | **GutWise** | Δ |
|---|---|---|---|
| Overall | 4.696 | **4.769 ± 0.028** | +0.073 |
| Accuracy | 4.78 | 4.78 ± 0.02 | 0 |
| Safety | 4.94 | 4.93 ± 0.04 | ≈ 0 |
| Empathy | 4.36 | 4.47 ± 0.08 | **+0.11** |
| Scope | 4.96 | 4.99 ± 0.01 | +0.03 |
| Completeness | 4.44 | 4.67 ± 0.04 | **+0.23** |
| Red-flag handling | 4/4 | **4/4 all 3 runs** | safety preserved |
| Wins vs baseline (B/A/tie, avg) | — | **27 / 13 / 10** | — |

Per-category gains: red_flag +0.22, safety_refusal +0.19, doctor_followup +0.13.
Single regression: myth_busting −0.08 (within σ).

Eval protocol: 50 held-out IBS questions × 3 seeds (42/43/44) = 150 paired prompts, judged side-by-side with the baseline in a single Haiku judge batch (no judge-calibration drift). Full numbers in the [GitHub repo](https://github.com/y0sif/GutWise/blob/main/scripts/evaluate/results/eval_report_v2.json).

## Training recipe

| Field | Value |
|---|---|
| Base model | `unsloth/gemma-4-E4B-it` (4-bit) |
| Adapter | LoRA, r=8, α=16, dropout=0, 7 target modules |
| LR / schedule | 5e-5 cosine, warmup_ratio=0.05 |
| Effective batch | 32 (4 × 8 accum) |
| Epochs | 1 |
| Precision | bf16 |
| Sequence length | 1024 |
| Packing | False |
| Completion-only loss | False (Unsloth VLM constraint) |
| NEFTune | Disabled (L4 OOM) |
| Hardware | Colab A100 |
| Dataset size | 659 audited IBS Q&A pairs |

## Intended use

- IBS *education* for patients and curious lay readers
- Demonstrating that small medical fine-tunes can ship safely with disciplined methodology

## Out of scope

- Diagnosis, prescribing, dosing
- Pediatric IBS (criteria differ)
- Non-English use
- Any clinical decision-making

## Known limitations

- Some hallucinations are inherited from base Gemma 4 (e.g., the "*L. rhamnosus* GC69" pattern — canonical strain is GG). v3 anti-hallucination pairs are planned.
- One category, myth_busting, regressed slightly (−0.08, within σ); v3 adds targeted training pairs.
- Held-out eval set had 2 duplicate red-flag questions (eval_049/050); fix on v3 backlog.

## Sources

- Rome IV Criteria
- ACG Clinical Guideline 2021 (Lacy et al.)
- NICE CG61 (Open Government Licence v3.0)
- BSG Guidelines on IBS 2021
- StatPearls (CC-BY 4.0)
- NHS IBS pages (OGL v3.0)
- MedlinePlus (Public Domain)
- 24 open-access PubMed abstracts

## Citation

```
@misc{gutwise2026,
  author       = {y0sif},
  title        = {GutWise — IBS Education Assistant on Gemma 4 E4B},
  year         = {2026},
  howpublished = {Hugging Face Hub},
  url          = {https://huggingface.co/y0sif/GutWise}
}
```

## License

Apache-2.0
