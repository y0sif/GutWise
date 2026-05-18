# Demo video — minimal version (60–75 seconds)

Use this script when you want a fast, no-frills submission video. Designed
around the **Quick demo (instant)** tab so there's no CPU wait on camera.

**Length**: 60–75 s. **Format**: screen capture + voiceover. No music, no
camera, no cuts beyond what OBS records. Plain microphone. One take.

Prereq: run the Colab cell from `QUICK_DEMO_GUIDE.md` so `cached_examples.json`
is populated and the **Quick demo (instant)** tab is visible on the Space.

---

## SCENE 1 — Hook (0:00–0:10)

**Visual**: title card or just the top of the Space page (the green
banner with "GutWise" and the disclaimer).

**Voiceover**:
> GutWise is a Gemma 4 fine-tune for IBS education. Submitted to the
> Gemma 4 Good Hackathon, Health and Sciences track. Apache 2.0, runs
> offline, grounded in NICE, ACG, and Rome IV.

---

## SCENE 2 — One real answer (0:10–0:25)

**Visual**: Quick demo tab, open the "Rome IV criteria" accordion.
Scroll the response so the viewer can see the structure (the 3
sub-criteria, the 6-month onset rule).

**Voiceover**:
> Here's GutWise answering on the Rome IV criteria. Three sub-criteria,
> three months of weekly pain, six months since onset. The structure
> comes from the actual guideline, not training noise. This is a
> pre-generated real output from the model — same Gemma 4 plus the LoRA
> adapter on Hugging Face, captured on GPU so the demo doesn't wait.

---

## SCENE 3 — The killer feature: red-flag handling (0:25–0:50)

**Visual**: Open the "Red flags (side-by-side vs base)" accordion. Both
panes visible. Base Gemma on the left, GutWise on the right.

**Voiceover**:
> Same red-flag prompt to base Gemma 4 and to GutWise. Bloody stool plus
> weight loss. Base Gemma starts discussing IBS subtypes. GutWise opens
> by escalating to a clinician. On our held-out eval, GutWise hit four
> out of four red-flag prompts across three independent runs. The first
> attempt regressed to two of four — recovering that is what this
> submission is really about.

---

## SCENE 4 — The numbers (0:50–1:05)

**Visual**: About tab. Hold on the results table for 10 seconds.

**Voiceover**:
> Three-run paired eval against base Gemma. Overall plus 0.07. Empathy
> plus 0.11. Completeness plus 0.23. Red-flag handling preserved at
> four of four. Six hundred and fifty-nine training pairs, every one
> grounded in real medical references.

---

## SCENE 5 — Close (1:05–1:15)

**Visual**: scroll to bottom of About tab — the URLs section.

**Voiceover**:
> Code, model, eval set, notebooks, and Kaggle writeup are all linked.
> Apache 2.0. Thanks for watching.

---

## Recording notes

- **One take**. The whole video is ~60 s of voiceover over Space screen
  capture. If you fluff a line, restart — it's only a minute.
- **Pre-open the accordions** on the Quick demo tab once before you hit
  record, then collapse them. That way they open snappily on click.
- **Captions**: turn on YouTube auto-captions and run one manual pass.
  Required for accessibility judging.
- **End frame**: hold the URLs on screen for the last 2 seconds.

## Upload

- YouTube → **Public** (not unlisted — the hackathon needs the link
  publicly viewable).
- Title: "GutWise — IBS education on Gemma 4 (Kaggle Gemma 4 Good
  Hackathon, Health & Sciences)".
- Description: paste the first paragraph of `WRITEUP.md` plus the three
  URLs (GitHub, HF model, HF Space).
- Pin a comment with the same links for accessibility.

## Where this video goes in your submission

1. Embed the YouTube link at the top of the Kaggle writeup notebook
2. Paste the link in the Hugging Face Space's "About" section (optional)
3. Paste the link in the GitHub README under "Hackathon submission"
4. That's it.
