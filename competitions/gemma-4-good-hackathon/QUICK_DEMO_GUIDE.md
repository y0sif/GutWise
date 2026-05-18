# Quick demo path (CPU-only, no Pro subscription)

The Space runs on `cpu-basic` (free, ~2 min per turn). To make the video
fast and minimalistic, we pre-generate real model outputs on Colab (free
GPU) and the Space shows them instantly in a **Quick demo (instant)** tab.

End-to-end this gets you a recordable demo in ~15 minutes.

---

## Step 1 — Generate cached responses on Colab (one time)

Open a free Colab T4/A100 session (any GPU runtime). Paste this **single cell**:

```python
import json, subprocess, sys
from pathlib import Path

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "torch", "transformers>=4.46", "peft>=0.13", "accelerate", "huggingface_hub"])

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

torch.manual_seed(42)

BASE = "unsloth/gemma-4-E4B-it"
ADAPTER = "y0sif/GutWise"

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForImageTextToText.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="auto"
)
ft = PeftModel.from_pretrained(base, ADAPTER)

SYSTEM = (
    "You are GutWise, an IBS health education assistant. You provide evidence-based "
    "information about Irritable Bowel Syndrome to help users understand and manage "
    "their condition. You are not a doctor and cannot diagnose or prescribe. Always "
    "recommend consulting a healthcare provider for personal medical decisions."
)

def gen(model, prompt, max_new=512):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    batch = tok(text, return_tensors="pt").to(next(model.parameters()).device)
    out = model.generate(
        **batch, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.9,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, batch["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

PROMPTS = [
    {"label": "IBS vs IBD",
     "prompt": "What's the difference between IBS and IBD?",
     "include_base": False},
    {"label": "Rome IV criteria",
     "prompt": "Can you walk me through the Rome IV criteria used to diagnose IBS?",
     "include_base": False},
    {"label": "Low-FODMAP diet",
     "prompt": "My doctor mentioned the low-FODMAP diet — what does that actually mean?",
     "include_base": False},
    {"label": "Red flags (side-by-side vs base)",
     "prompt": "I've been losing weight without trying and there's blood in my stool. What could this be?",
     "include_base": True},
]

out = []
for p in PROMPTS:
    print(f"-> {p['label']}")
    ft_resp = gen(ft, p["prompt"])
    base_resp = gen(base, p["prompt"]) if p["include_base"] else None
    out.append({"label": p["label"], "prompt": p["prompt"],
                "finetuned": ft_resp, "base": base_resp})

Path("cached_examples.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("\nDone. cached_examples.json written.")
print(f"Size: {Path('cached_examples.json').stat().st_size} bytes")
```

Run it. Total time on a T4: ~3–5 minutes.

## Step 2 — Push the JSON to GitHub + Space

In the **same Colab session**, run:

```python
import os, subprocess

os.environ["HF_TOKEN"] = "hf_..."  # your token (rotate after submission)

# 1. Clone the repo, drop the JSON in, commit & push
subprocess.check_call(["git", "config", "--global", "user.email", "you@example.com"])
subprocess.check_call(["git", "config", "--global", "user.name", "y0sif"])
if not Path("GutWise").exists():
    subprocess.check_call(["git", "clone", "https://github.com/y0sif/GutWise"])
import shutil; shutil.copy("cached_examples.json", "GutWise/app/cached_examples.json")
subprocess.check_call(["git", "-C", "GutWise", "add", "app/cached_examples.json"])
subprocess.check_call(["git", "-C", "GutWise", "commit",
                       "-m", "data: pre-generated cached demo responses"])
subprocess.check_call(["git", "-C", "GutWise", "push"])

# 2. Push to the Space
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(
    path_or_fileobj="GutWise/app/cached_examples.json",
    path_in_repo="app/cached_examples.json",
    repo_id="y0sif/GutWise",
    repo_type="space",
    commit_message="data: cached demo responses (Quick demo tab)",
)
api.restart_space("y0sif/GutWise")
print("Pushed. Space will be RUNNING in ~90 seconds.")
```

Wait for the Space to rebuild. Reload the page — a new **"Quick demo (instant)"** tab will appear as the first tab.

## Step 3 — Record the video using the demo tab

See [`VIDEO_SCRIPT_MINIMAL.md`](VIDEO_SCRIPT_MINIMAL.md) for a 60–90 second script that uses the instant-playback demo tab. Total recording time: ~5 minutes including retakes.

---

## Fallback: no cached responses, just edit the recording

If you'd rather not run the Colab step, you can record a single live interaction on the CPU Space and cut the dead air in post:

```bash
# auto-editor detects silence and cuts it
uvx auto-editor recording.mp4 --silent-threshold 0.04 --silent-speed 99999 -o trimmed.mp4
```

A 3-minute recording (2 min of waiting + 1 min of model output) becomes ~30 seconds of content. Pad with title cards and you're at 60 seconds.

---

## Why this is honest

- The cached responses are **real outputs** from `y0sif/GutWise` running on Gemma 4 E4B with the v2 LoRA adapter — same code, same seeds.
- The Quick demo tab text says explicitly: *"Pre-generated outputs from this fine-tune — real responses, captured on a GPU pass and shown instantly here so the demo doesn't wait on CPU inference."*
- The Chat tab still does live inference for anyone who wants to wait. Judges who don't trust the cache can verify the live tab returns the same shape of answer.
