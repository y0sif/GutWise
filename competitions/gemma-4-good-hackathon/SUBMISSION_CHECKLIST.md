# GutWise — Kaggle submission checklist

Deadline: **May 18, 2026.** Today: May 17, 2026. Track: Health & Sciences.

## What Claude has prepared (✅ in repo)

- [x] `README.md` — hackathon-framed hero, results table, pipeline, disclaimer
- [x] `app.py` + `app/main.py` — Gradio demo (Chat tab + side-by-side tab + About tab) with ZeroGPU decorator
- [x] `requirements.txt` — pinned deps for HF Spaces
- [x] `scripts/publish/push_model.py` — HF Hub adapter push script
- [x] `scripts/publish/MODEL_CARD.md` — HF model card with eval results + recipe
- [x] `competitions/gemma-4-good-hackathon/WRITEUP.md` — Kaggle technical writeup
- [x] `competitions/gemma-4-good-hackathon/VIDEO_SCRIPT.md` — 2:30 video script with shot list
- [x] `competitions/gemma-4-good-hackathon/COVER.svg` — title-card cover image
- [x] Clean `.gitignore` excluding intermediate generation artifacts

## What needs you (manual steps)

### 1. Push the v2 adapter to Hugging Face Hub

From Colab (where the LoRA adapter is on Drive):

```bash
# In a Colab cell, after mounting Drive
%env HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
!pip install huggingface_hub
!git clone https://github.com/y0sif/GutWise
%cd GutWise
!python scripts/publish/push_model.py \
    --adapter-dir /content/drive/MyDrive/GutWise/final/e4b-v2/lora_adapter \
    --repo-id y0sif/GutWise
```

Get the token at https://huggingface.co/settings/tokens (write scope).

### 2. Create the GitHub repo and push

```fish
cd ~/Projects/GutWise
gh repo create y0sif/GutWise --public --source=. --remote=origin \
    --description "Offline-first IBS education assistant on Gemma 4 E4B (Kaggle Gemma 4 Good Hackathon)"
git push -u origin main
```

(Claude can do this on your sign-off — see the end of its message.)

### 3. Create the Hugging Face Space

1. Go to https://huggingface.co/new-space
2. Owner: `y0sif`, Name: `GutWise`, SDK: **Gradio**, Hardware: **ZeroGPU** (free), Visibility: Public
3. Clone the empty Space repo: `git clone https://huggingface.co/spaces/y0sif/GutWise gutwise-space`
4. Copy these from the GitHub repo into the Space repo:
   - `app.py`
   - `requirements.txt`
   - `app/` (whole folder)
5. Add a Space `README.md` with this front-matter at the top, then the contents of the GitHub README:

```yaml
---
title: GutWise
emoji: 🩺
colorFrom: emerald
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Offline IBS education assistant — Gemma 4 E4B fine-tune
---
```

6. `git add . && git commit -m "GutWise Space" && git push`
7. Wait for the Space to build (5–10 min). Send one warmup query to make sure the LoRA loads.

### 4. Record the demo video

- Open `competitions/gemma-4-good-hackathon/VIDEO_SCRIPT.md`.
- Record per the shot list (OBS or screen.studio, 1080p60).
- Trim with auto-editor: `uvx auto-editor input.mp4 -o output.mp4`.
- Upload as **public** to YouTube. Keep it 2:30–3:00.
- Add the YouTube link to:
  - the Kaggle writeup notebook (top of the page)
  - the GitHub README (under "Hackathon submission")
  - the Hugging Face Space About tab

### 5. Create the Kaggle writeup notebook

1. Go to the [hackathon page](https://www.kaggle.com/competitions/gemma-4-good-hackathon) and click **New Submission → Notebook writeup**.
2. Paste the body of `competitions/gemma-4-good-hackathon/WRITEUP.md` into the notebook as Markdown.
3. Add a top-of-page **embed of the YouTube video** (Kaggle's video block).
4. Add a **link to the HF Space** as the "Try it live" CTA below the hook.
5. Upload the cover image (`competitions/gemma-4-good-hackathon/COVER.svg`, or a PNG export) as the notebook cover.
6. Tags: `health`, `gemma-4`, `fine-tuning`, `lora`, `ibs`.
7. Submit notebook → submit to the competition.

### 6. Verify submission components

The hackathon expects, all of these public on submission day:
- [ ] Kaggle writeup (notebook)
- [ ] Public code repo (GitHub)
- [ ] Working demo (HF Space link, click-tested)
- [ ] Video pitch (YouTube link, public)
- [ ] Cover image / media (uploaded to writeup)

Click each link in an incognito window to confirm none of them require auth.

### 7. Final smoke test

```fish
# 1. Repo is public and reproducible
gh repo view y0sif/GutWise --web

# 2. Model is public on HF
curl -sI https://huggingface.co/y0sif/GutWise/resolve/main/adapter_config.json | head -1

# 3. Space is up
curl -sI https://huggingface.co/spaces/y0sif/GutWise | head -1
```

If all three return `HTTP/2 200`, submit.
